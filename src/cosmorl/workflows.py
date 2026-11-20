"""High-level programmatic workflows for simulation, PPO training, and evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

from .config import (
    CosmologyConfig,
    EnvironmentConfig,
    ExperimentConfig,
    PolicyConfig,
    RuntimeConfig,
    SimulationConfig,
)
from .physics.cosmology import InitialConditions
from .physics.pm import Integrator, ParticleState, PeriodicMesh
from .rl.environment import FieldErrors, TimeStepEnvironment
from .rl.network import ActorCritic
from .rl.ppo import collect_episode, evaluate_episode, update_policy


@dataclass(frozen=True)
class SimulationResult:
    """Summary of a fixed-step Particle-Mesh simulation run."""

    device: str
    a_final: float
    num_particles: int
    pm_steps: int
    initial_delta_rms: float
    final_delta_rms: float
    mean_delta: float
    output_path: Path | None
    state: ParticleState


@dataclass(frozen=True)
class UpdateMetrics:
    """Per-update PPO training metrics."""

    update: int
    mean_reward: float
    mean_pm_steps: float
    ppo_loss: float


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed PPO training run."""

    checkpoint_path: Path
    next_seed: int
    history: tuple[UpdateMetrics, ...]


@dataclass(frozen=True)
class EvaluationEpisodeResult:
    """Comparison between a trained policy and a fixed-step baseline for one seed."""

    seed: int
    policy_steps: int
    fixed_steps: int
    policy_errors: FieldErrors | None
    fixed_errors: FieldErrors


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated evaluation results across seeds."""

    checkpoint_path: Path
    environment: EnvironmentConfig
    episodes: tuple[EvaluationEpisodeResult, ...]


def resolve_device(runtime: RuntimeConfig) -> torch.device:
    """Configure PyTorch CPU threads and resolve the execution device."""
    torch.set_num_threads(runtime.threads)
    if runtime.device == "auto":
        backend = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        backend = runtime.device
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested in runtime.device, but no CUDA device is available.")
    return torch.device(backend)


def build_simulation(
    simulation: SimulationConfig,
    cosmology: CosmologyConfig,
    device: torch.device,
) -> tuple[Integrator, InitialConditions]:
    """Construct the CLASS initial condition generator and periodic PM integrator."""
    initial = InitialConditions(
        size=simulation.resolved_particle_size,
        box_mpc=simulation.box_mpc,
        a_initial=simulation.a_initial,
        cosmology=cosmology,
        device=device,
    )
    mesh = PeriodicMesh(simulation.grid_size, initial.background.omega_m, device)
    return Integrator(mesh, initial.background), initial


def save_checkpoint(data: dict[str, Any], path: str | Path) -> Path:
    """Atomically save a PyTorch checkpoint to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(data, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a saved PyTorch checkpoint safely on CPU."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def run_simulation(config: ExperimentConfig) -> SimulationResult:
    """Run a fixed-step PM simulation defined by config.simulation."""
    device = resolve_device(config.runtime)
    sim = config.simulation
    integrator, initial = build_simulation(sim, config.cosmology, device)
    state = integrator.initialize(sim.a_initial, *initial.sample(sim.seed))
    initial_rms = state.delta.square().mean().sqrt().item()
    state = integrator.advance_fixed(state, sim.a_final, sim.steps)
    final_rms = state.delta.square().mean().sqrt().item()
    mean_delta = state.delta.mean().item()

    saved_path: Path | None = None
    if sim.output_path is not None:
        saved_path = save_checkpoint(
            {
                **sim.to_mesh_dict(),
                "cosmology": asdict(config.cosmology),
                "a": state.a,
                "seed": sim.seed,
                "positions": state.positions.cpu(),
                "momentum": state.momentum.cpu(),
                "density_contrast": state.delta.cpu(),
            },
            sim.output_path,
        )

    return SimulationResult(
        device=str(device),
        a_final=state.a,
        num_particles=len(state.positions),
        pm_steps=sim.steps,
        initial_delta_rms=initial_rms,
        final_delta_rms=final_rms,
        mean_delta=mean_delta,
        output_path=saved_path,
        state=state,
    )


def run_training(
    config: ExperimentConfig,
    *,
    on_update: Callable[[UpdateMetrics], None] | None = None,
) -> TrainingResult:
    """Train an O(3)-equivariant PPO time-step policy and save a checkpoint."""
    device = resolve_device(config.runtime)
    integrator, initial = build_simulation(config.simulation, config.cosmology, device)
    environment = TimeStepEnvironment(integrator, initial, config.environment)

    torch.manual_seed(config.training.seed)
    policy = ActorCritic.from_config(len(config.environment.actions), config.policy).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.ppo.learning_rate)

    next_seed = config.training.seed
    history: list[UpdateMetrics] = []
    for update in range(1, config.training.updates + 1):
        rollouts = [
            collect_episode(environment, policy, seed, gae_lambda=config.ppo.gae_lambda)
            for seed in range(next_seed, next_seed + config.training.episodes_per_update)
        ]
        next_seed += len(rollouts)
        loss = update_policy(policy, optimizer, rollouts, config.ppo)
        metrics = UpdateMetrics(
            update=update,
            mean_reward=sum(r.reward for r in rollouts) / len(rollouts),
            mean_pm_steps=sum(len(r.observations) for r in rollouts) / len(rollouts),
            ppo_loss=loss,
        )
        history.append(metrics)
        if on_update is not None:
            on_update(metrics)

    checkpoint_path = save_checkpoint(
        {
            "policy": policy.state_dict(),
            "policy_config": asdict(config.policy),
            "cosmology": asdict(config.cosmology),
            "simulation": config.simulation.to_mesh_dict(),
            "environment": asdict(config.environment),
            "ppo": asdict(config.ppo),
            "next_seed": next_seed,
        },
        config.training.output_path,
    )
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        next_seed=next_seed,
        history=tuple(history),
    )


def run_evaluation(config: ExperimentConfig) -> EvaluationResult:
    """Evaluate a saved PPO policy checkpoint against a fixed-step baseline."""
    device = resolve_device(config.runtime)
    eval_cfg = config.evaluation
    checkpoint = load_checkpoint(eval_cfg.checkpoint_path)

    env_data = checkpoint.get("environment") or checkpoint["episode"]
    env_cfg = EnvironmentConfig(**env_data)
    policy_cfg = PolicyConfig(**checkpoint["policy_config"])
    cosmology_cfg = CosmologyConfig(**checkpoint.get("cosmology", {}))

    sim_raw = checkpoint["simulation"]
    sim_cfg = SimulationConfig(
        grid_size=sim_raw.get("grid_size", sim_raw.get("grid")),
        particle_size=sim_raw.get("particle_size", sim_raw.get("particles")),
        box_mpc=sim_raw["box_mpc"],
        a_initial=sim_raw["a_initial"],
        a_final=env_cfg.a_final,
    )

    policy = ActorCritic.from_config(len(env_cfg.actions), policy_cfg)
    policy.load_state_dict(checkpoint["policy"])
    policy.to(device)
    policy.eval()

    integrator, initial = build_simulation(sim_cfg, cosmology_cfg, device)
    environment = TimeStepEnvironment(integrator, initial, env_cfg)

    first_seed = checkpoint["next_seed"] if eval_cfg.seed is None else eval_cfg.seed
    results: list[EvaluationEpisodeResult] = []
    for seed in range(first_seed, first_seed + eval_cfg.episodes):
        outcome = evaluate_episode(environment, policy, seed)
        assert environment.initial_state is not None
        baseline = integrator.advance_fixed(environment.initial_state, env_cfg.a_final, env_cfg.budget)
        results.append(
            EvaluationEpisodeResult(
                seed=seed,
                policy_steps=environment.steps_used,
                fixed_steps=env_cfg.budget,
                policy_errors=outcome.errors,
                fixed_errors=environment.errors(baseline),
            )
        )

    return EvaluationResult(
        checkpoint_path=Path(eval_cfg.checkpoint_path),
        environment=env_cfg,
        episodes=tuple(results),
    )
