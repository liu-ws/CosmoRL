"""Structured configuration schemas and TOML/JSON loader for CosmoRL."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field
from math import ceil, exp, isfinite, log
from pathlib import Path
from typing import Any, Literal

DeviceType = Literal["auto", "cpu", "cuda"]


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_positive_int(value: Any, name: str, *, minimum: int = 1) -> None:
    if not _is_strict_int(value) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")


def _require_positive_float(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}.")


def _require_non_negative_float(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}.")


def _check_unknown_keys(section: str, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in [{section}]: {', '.join(unknown)}.")


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution device and CPU threading settings."""

    device: DeviceType = "auto"
    threads: int = 4

    def __post_init__(self) -> None:
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"runtime.device must be 'auto', 'cpu', or 'cuda', got {self.device!r}.")
        _require_positive_int(self.threads, "runtime.threads")


@dataclass(frozen=True)
class CosmologyConfig:
    """Flat LambdaCDM cosmological parameters passed to CLASS."""

    h: float = 0.674
    omega_b: float = 0.0224
    omega_cdm: float = 0.120
    scalar_amplitude: float = 2.1e-9
    scalar_index: float = 0.965

    def __post_init__(self) -> None:
        _require_positive_float(self.h, "cosmology.h")
        _require_positive_float(self.omega_b, "cosmology.omega_b")
        _require_positive_float(self.omega_cdm, "cosmology.omega_cdm")
        _require_positive_float(self.scalar_amplitude, "cosmology.scalar_amplitude")
        if isinstance(self.scalar_index, bool) or not isinstance(self.scalar_index, (int, float)) or not isfinite(self.scalar_index):
            raise ValueError(f"cosmology.scalar_index must be finite, got {self.scalar_index!r}.")


@dataclass(frozen=True)
class SimulationConfig:
    """Particle-Mesh box geometry and fixed-step integration parameters."""

    grid_size: int = 16
    particle_size: int | None = None
    box_mpc: float = 100.0
    a_initial: float = 0.05
    a_final: float = 1.0
    steps: int = 64
    seed: int = 0
    output_path: str | None = None

    def __post_init__(self) -> None:
        _require_positive_int(self.grid_size, "simulation.grid_size", minimum=2)
        resolved_particles = self.grid_size if self.particle_size is None else self.particle_size
        _require_positive_int(resolved_particles, "simulation.particle_size", minimum=3)
        object.__setattr__(self, "particle_size", resolved_particles)
        _require_positive_float(self.box_mpc, "simulation.box_mpc")
        if not (isfinite(self.a_initial) and isfinite(self.a_final) and 0 < self.a_initial < self.a_final <= 1.0):
            raise ValueError("simulation requires 0 < a_initial < a_final <= 1.")
        _require_positive_int(self.steps, "simulation.steps")
        if not _is_strict_int(self.seed):
            raise ValueError(f"simulation.seed must be an integer, got {self.seed!r}.")
        if self.output_path is not None and not isinstance(self.output_path, str):
            raise ValueError("simulation.output_path must be a string path or omitted.")

    @property
    def resolved_particle_size(self) -> int:
        return self.grid_size if self.particle_size is None else self.particle_size

    def to_mesh_dict(self) -> dict[str, Any]:
        """Serialize box/grid geometry stored inside simulation and policy checkpoints."""
        return {
            "grid_size": self.grid_size,
            "particle_size": self.resolved_particle_size,
            "box_mpc": float(self.box_mpc),
            "a_initial": float(self.a_initial),
        }


@dataclass(frozen=True)
class EnvironmentConfig:
    """Fixed-budget adaptive time-stepping MDP specification."""

    a_initial: float = 0.05
    a_final: float = 1.0
    budget: int = 48
    reference_steps: int | None = None
    actions: tuple[float, ...] = (0.01, 0.02, 0.04, 0.08)
    density_scale: float = 1e-6
    power_scale: float = 1e-6
    reward_every: int | None = None

    def __post_init__(self) -> None:
        if not (isfinite(self.a_initial) and isfinite(self.a_final) and 0 < self.a_initial < self.a_final <= 1.0):
            raise ValueError("environment requires 0 < a_initial < a_final <= 1.")
        actions_tuple = tuple(float(step) for step in self.actions)
        object.__setattr__(self, "actions", actions_tuple)
        if not actions_tuple or any(not isfinite(step) or step <= 0 for step in actions_tuple):
            raise ValueError("environment.actions must be a non-empty sequence of positive numbers.")
        _require_positive_int(self.budget, "environment.budget")
        if self.minimum_steps(self.a_initial) > self.budget:
            raise ValueError("The PM budget cannot reach a_final even using the largest time step.")
        resolved_ref = 4 * self.budget if self.reference_steps is None else self.reference_steps
        _require_positive_int(resolved_ref, "environment.reference_steps")
        if resolved_ref <= self.budget:
            raise ValueError("environment.reference_steps must be strictly greater than environment.budget.")
        object.__setattr__(self, "reference_steps", resolved_ref)
        _require_positive_float(self.density_scale, "environment.density_scale")
        _require_positive_float(self.power_scale, "environment.power_scale")
        if self.reward_every is not None:
            _require_positive_int(self.reward_every, "environment.reward_every")

    def minimum_steps(self, a: float) -> int:
        """Minimum number of remaining KDK steps required to reach a_final."""
        return ceil(log(self.a_final / a) / max(self.actions) - 1e-10)

    def legal_targets(self, a: float, remaining: int) -> tuple[float | None, ...]:
        """Return legal next scale factors for each action under the remaining budget."""
        interval = log(self.a_final / a)
        legal: list[float | None] = []
        seen: set[float] = set()
        for step_size in self.actions:
            target = self.a_final if step_size >= interval - 1e-12 else a * exp(step_size)
            if target in seen or target <= a or self.minimum_steps(target) >= remaining:
                legal.append(None)
            else:
                legal.append(target)
                seen.add(target)
        return tuple(legal)


@dataclass(frozen=True)
class PolicyConfig:
    """Architecture hyperparameters for the O(3)-equivariant Actor-Critic."""

    channels: int = 8
    kernel_size: int = 3
    conv_layers: int = 2
    residual: bool = False
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        _require_positive_int(self.channels, "policy.channels")
        _require_positive_int(self.kernel_size, "policy.kernel_size")
        if self.kernel_size % 2 == 0:
            raise ValueError(f"policy.kernel_size must be an odd integer, got {self.kernel_size!r}.")
        _require_positive_int(self.conv_layers, "policy.conv_layers")
        if not isinstance(self.residual, bool):
            raise ValueError(f"policy.residual must be a bool, got {self.residual!r}.")
        _require_positive_int(self.hidden_dim, "policy.hidden_dim")


@dataclass(frozen=True)
class PPOConfig:
    """PPO optimization and GAE hyperparameters."""

    learning_rate: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 8
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        _require_positive_float(self.learning_rate, "ppo.learning_rate")
        _require_positive_int(self.epochs, "ppo.epochs")
        _require_positive_int(self.minibatch_size, "ppo.minibatch_size")
        _require_positive_float(self.clip_ratio, "ppo.clip_ratio")
        _require_non_negative_float(self.value_coef, "ppo.value_coef")
        _require_non_negative_float(self.entropy_coef, "ppo.entropy_coef")
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError(f"ppo.gae_lambda must be in [0, 1], got {self.gae_lambda!r}.")
        _require_positive_float(self.max_grad_norm, "ppo.max_grad_norm")


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop schedule and checkpoint destination."""

    seed: int = 0
    updates: int = 2
    episodes_per_update: int = 2
    output_path: str = "outputs/policy.pt"

    def __post_init__(self) -> None:
        if not _is_strict_int(self.seed):
            raise ValueError(f"training.seed must be an integer, got {self.seed!r}.")
        _require_positive_int(self.updates, "training.updates")
        _require_positive_int(self.episodes_per_update, "training.episodes_per_update")
        if not isinstance(self.output_path, str) or not self.output_path.strip():
            raise ValueError("training.output_path must be a non-empty path string.")


@dataclass(frozen=True)
class EvaluationConfig:
    """Policy checkpoint evaluation settings."""

    checkpoint_path: str = "outputs/policy.pt"
    seed: int | None = None
    episodes: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_path, str) or not self.checkpoint_path.strip():
            raise ValueError("evaluation.checkpoint_path must be a non-empty path string.")
        if self.seed is not None and not _is_strict_int(self.seed):
            raise ValueError(f"evaluation.seed must be an integer or omitted, got {self.seed!r}.")
        _require_positive_int(self.episodes, "evaluation.episodes")


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration unifying simulation, RL training, and evaluation."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        """Build and validate an ExperimentConfig from a nested dictionary."""
        if not isinstance(raw, dict):
            raise ValueError("Configuration root must be a table/mapping.")
        allowed_sections = {
            "runtime",
            "cosmology",
            "simulation",
            "environment",
            "policy",
            "ppo",
            "training",
            "evaluation",
        }
        _check_unknown_keys("root", raw, allowed_sections)

        def section(name: str, allowed: set[str]) -> dict[str, Any]:
            value = raw.get(name, {})
            if not isinstance(value, dict):
                raise ValueError(f"Section [{name}] must be a table/mapping.")
            _check_unknown_keys(name, value, allowed)
            return dict(value)

        runtime_data = section("runtime", {"device", "threads"})
        cosmology_data = section("cosmology", {"h", "omega_b", "omega_cdm", "scalar_amplitude", "scalar_index"})
        sim_data = section(
            "simulation",
            {"grid_size", "particle_size", "box_mpc", "a_initial", "a_final", "steps", "seed", "output_path"},
        )
        env_data = section(
            "environment",
            {
                "a_initial",
                "a_final",
                "budget",
                "reference_steps",
                "actions",
                "density_scale",
                "power_scale",
                "reward_every",
            },
        )
        policy_data = section("policy", {"channels", "kernel_size", "conv_layers", "residual", "hidden_dim"})
        ppo_data = section(
            "ppo",
            {
                "learning_rate",
                "epochs",
                "minibatch_size",
                "clip_ratio",
                "value_coef",
                "entropy_coef",
                "gae_lambda",
                "max_grad_norm",
            },
        )
        train_data = section("training", {"seed", "updates", "episodes_per_update", "output_path"})
        eval_data = section("evaluation", {"checkpoint_path", "seed", "episodes"})

        if env_data.get("reward_every") == "terminal":
            env_data["reward_every"] = None

        simulation = SimulationConfig(**sim_data)
        env_data.setdefault("a_initial", simulation.a_initial)
        env_data.setdefault("a_final", simulation.a_final)
        if env_data["a_initial"] != simulation.a_initial or env_data["a_final"] != simulation.a_final:
            raise ValueError(
                "simulation and environment must share the same a_initial and a_final bounds."
            )

        return cls(
            runtime=RuntimeConfig(**runtime_data),
            cosmology=CosmologyConfig(**cosmology_data),
            simulation=simulation,
            environment=EnvironmentConfig(**env_data),
            policy=PolicyConfig(**policy_data),
            ppo=PPOConfig(**ppo_data),
            training=TrainingConfig(**train_data),
            evaluation=EvaluationConfig(**eval_data),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        """Load an ExperimentConfig from a .toml or .json file."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        suffix = config_path.suffix.lower()
        if suffix == ".toml":
            with config_path.open("rb") as stream:
                data = tomllib.load(stream)
        elif suffix == ".json":
            with config_path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
        else:
            raise ValueError(f"Unsupported configuration format {suffix!r}; expected '.toml' or '.json'.")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain nested dictionary suitable for serialization."""
        data = asdict(self)
        data["environment"]["actions"] = list(self.environment.actions)
        return data

    def to_toml(self) -> str:
        """Render a readable, self-documented TOML representation."""
        reward_line = (
            'reward_every = "terminal"  # "terminal" for terminal-only reward, or positive integer N'
            if self.environment.reward_every is None
            else f"reward_every = {self.environment.reward_every}"
        )
        sim_output_line = (
            '# output_path = "outputs/snapshot.pt"'
            if self.simulation.output_path is None
            else f'output_path = "{self.simulation.output_path}"'
        )
        eval_seed_line = (
            "# seed = 0  # omit to continue from checkpoint next_seed"
            if self.evaluation.seed is None
            else f"seed = {self.evaluation.seed}"
        )
        actions_literal = ", ".join(f"{step:g}" for step in self.environment.actions)
        return "\n".join(
            [
                "# CosmoRL experiment configuration",
                "",
                "[runtime]",
                f'device = "{self.runtime.device}"  # "auto" | "cpu" | "cuda"',
                f"threads = {self.runtime.threads}",
                "",
                "[cosmology]",
                f"h = {self.cosmology.h:g}",
                f"omega_b = {self.cosmology.omega_b:g}",
                f"omega_cdm = {self.cosmology.omega_cdm:g}",
                f"scalar_amplitude = {self.cosmology.scalar_amplitude:g}",
                f"scalar_index = {self.cosmology.scalar_index:g}",
                "",
                "[simulation]",
                f"grid_size = {self.simulation.grid_size}",
                f"particle_size = {self.simulation.resolved_particle_size}",
                f"box_mpc = {self.simulation.box_mpc:g}",
                f"a_initial = {self.simulation.a_initial:g}",
                f"a_final = {self.simulation.a_final:g}",
                f"steps = {self.simulation.steps}",
                f"seed = {self.simulation.seed}",
                sim_output_line,
                "",
                "[environment]",
                f"budget = {self.environment.budget}",
                f"reference_steps = {self.environment.reference_steps}",
                f"actions = [{actions_literal}]",
                f"density_scale = {self.environment.density_scale:g}",
                f"power_scale = {self.environment.power_scale:g}",
                reward_line,
                "",
                "[policy]",
                f"channels = {self.policy.channels}",
                f"kernel_size = {self.policy.kernel_size}",
                f"conv_layers = {self.policy.conv_layers}",
                f"residual = {'true' if self.policy.residual else 'false'}",
                f"hidden_dim = {self.policy.hidden_dim}",
                "",
                "[ppo]",
                f"learning_rate = {self.ppo.learning_rate:g}",
                f"epochs = {self.ppo.epochs}",
                f"minibatch_size = {self.ppo.minibatch_size}",
                f"clip_ratio = {self.ppo.clip_ratio:g}",
                f"value_coef = {self.ppo.value_coef:g}",
                f"entropy_coef = {self.ppo.entropy_coef:g}",
                f"gae_lambda = {self.ppo.gae_lambda:g}",
                f"max_grad_norm = {self.ppo.max_grad_norm:g}",
                "",
                "[training]",
                f"seed = {self.training.seed}",
                f"updates = {self.training.updates}",
                f"episodes_per_update = {self.training.episodes_per_update}",
                f'output_path = "{self.training.output_path}"',
                "",
                "[evaluation]",
                f'checkpoint_path = "{self.evaluation.checkpoint_path}"',
                eval_seed_line,
                f"episodes = {self.evaluation.episodes}",
                "",
            ]
        )


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load an ExperimentConfig from disk, or return defaults when path is None."""
    if path is None:
        return ExperimentConfig()
    return ExperimentConfig.from_file(path)


def write_default_config(path: str | Path, config: ExperimentConfig | None = None) -> Path:
    """Write a commented TOML configuration file to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = (config or ExperimentConfig()).to_toml()
    target.write_text(rendered, encoding="utf-8")
    return target
