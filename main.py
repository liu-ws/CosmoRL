"""Run a Particle-Mesh simulation or train and evaluate a time-step policy."""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from cosmorl.environment import EpisodeConfig, FieldErrors, TimeStepEnvironment
from cosmorl.initial import Cosmology, InitialConditions
from cosmorl.pm import Integrator, PeriodicMesh
from cosmorl.ppo import ActorCritic, collect_episode, evaluate_episode, update_policy


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runtime = argparse.ArgumentParser(add_help=False)
    runtime.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    runtime.add_argument("--threads", type=positive_int, default=4, help="PyTorch CPU threads")
    simulation = argparse.ArgumentParser(add_help=False, parents=[runtime])
    simulation.add_argument("--grid", type=positive_int, default=16, help="force cells per axis")
    simulation.add_argument("--particles", type=positive_int, help="particles per axis (defaults to --grid)")
    simulation.add_argument("--box-mpc", type=float, default=100, help="comoving box side in Mpc")
    simulation.add_argument("--a-initial", type=float, default=0.05)
    simulation.add_argument("--a-final", type=float, default=1)
    simulation.add_argument("--seed", type=int, default=0)
    simulate = commands.add_parser("simulate", parents=[simulation], help="fixed steps in ln(a)")
    simulate.add_argument("--steps", type=positive_int, default=64)
    simulate.add_argument("--output", type=Path, help="optional final particle and density snapshot")
    train = commands.add_parser("train", parents=[simulation], help="train a PPO time-step policy")
    train.add_argument("--budget", type=positive_int, default=48)
    train.add_argument("--reference-steps", type=positive_int, help="defaults to four times the PM budget")
    train.add_argument("--density-scale", type=float, default=1e-6, help="density error scale in the terminal loss")
    train.add_argument("--power-scale", type=float, default=1e-6, help="power error scale in the terminal loss")
    train.add_argument("--updates", type=positive_int, default=2)
    train.add_argument("--episodes-per-update", type=positive_int, default=2)
    train.add_argument("--output", type=Path, default=Path("policy.pt"))
    evaluate = commands.add_parser("evaluate", parents=[runtime], help="compare a policy to fixed steps")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--seed", type=int, help="first evaluation seed (defaults to the next unused seed)")
    evaluate.add_argument("--episodes", type=positive_int, default=2)
    return parser.parse_args(sys.argv[1:] or ["simulate"])


def build_simulation(settings: dict, device: torch.device) -> tuple[Integrator, InitialConditions]:
    grid = settings["grid"]
    if grid < 2:
        raise ValueError("The force grid needs at least two cells per axis.")
    initial = InitialConditions(
        settings["particles"], settings["box_mpc"], settings["a_initial"], Cosmology(), device
    )
    mesh = PeriodicMesh(grid, initial.background.omega_m, device)
    return Integrator(mesh, initial.background), initial


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(data, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def simulate(args: argparse.Namespace, settings: dict, device: torch.device) -> None:
    integrator, initial = build_simulation(settings, device)
    state = integrator.initialize(args.a_initial, *initial.sample(args.seed))
    initial_rms = state.delta.square().mean().sqrt().item()
    state = integrator.advance_fixed(state, args.a_final, args.steps)
    final_rms = state.delta.square().mean().sqrt().item()
    print(
        f"device={device} a={state.a:g} particles={len(state.positions)} "
        f"pm_steps={args.steps} initial_delta_rms={initial_rms:.6g} final_delta_rms={final_rms:.6g} "
        f"mean_delta={state.delta.mean().item():.3g}"
    )
    if args.output is not None:
        save({
            **settings, "a": state.a, "seed": args.seed, "positions": state.positions.cpu(),
            "momentum": state.momentum.cpu(), "density_contrast": state.delta.cpu(),
        }, args.output)


def train(args: argparse.Namespace, settings: dict, device: torch.device) -> None:
    config = EpisodeConfig(
        a_initial=args.a_initial,
        a_final=args.a_final,
        budget=args.budget,
        reference_steps=args.reference_steps or 4 * args.budget,
        density_scale=args.density_scale,
        power_scale=args.power_scale,
    )
    environment = TimeStepEnvironment(*build_simulation(settings, device), config)
    torch.manual_seed(args.seed)
    policy = ActorCritic(len(config.actions)).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    next_seed = args.seed
    for update in range(1, args.updates + 1):
        rollouts = [
            collect_episode(environment, policy, seed)
            for seed in range(next_seed, next_seed + args.episodes_per_update)
        ]
        next_seed += len(rollouts)
        loss = update_policy(policy, optimizer, rollouts)
        mean_reward = sum(rollout.reward for rollout in rollouts) / len(rollouts)
        mean_steps = sum(len(rollout.observations) for rollout in rollouts) / len(rollouts)
        print(f"update={update} reward={mean_reward:.6g} mean_pm_steps={mean_steps:g} ppo_loss={loss:.6g}", flush=True)
    save({
        "policy": policy.state_dict(), "simulation": settings, "episode": asdict(config), "next_seed": next_seed,
    }, args.output)
    print(f"checkpoint={args.output}")


def describe(errors: FieldErrors | None, config: EpisodeConfig) -> str:
    if errors is None:
        return "failed"
    return (
        f"loss={errors.loss(config):.6g} e_delta={errors.density:.6g} "
        f"e_power={errors.power:.6g} cross_r={errors.correlation:.6g}"
    )


def evaluate(args: argparse.Namespace, device: torch.device) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = EpisodeConfig(**checkpoint["episode"])
    integrator, initial = build_simulation(checkpoint["simulation"], device)
    environment = TimeStepEnvironment(integrator, initial, config)
    policy = ActorCritic(len(config.actions)).to(device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    first_seed = checkpoint["next_seed"] if args.seed is None else args.seed
    for seed in range(first_seed, first_seed + args.episodes):
        result = evaluate_episode(environment, policy, seed)
        baseline = integrator.advance_fixed(environment.initial_state, config.a_final, config.budget)
        print(f"seed={seed} policy pm_steps={environment.steps_used} {describe(result.errors, config)}")
        print(f"seed={seed} fixed  pm_steps={config.budget} {describe(environment.errors(baseline), config)}")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    backend = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if backend == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    device = torch.device(backend)
    try:
        if args.command == "evaluate":
            evaluate(args, device)
            return
        settings = {
            "grid": args.grid, "particles": args.particles or args.grid, "box_mpc": args.box_mpc,
            "a_initial": args.a_initial,
        }
        if not 0 < args.a_initial < args.a_final <= 1:
            raise ValueError("Require 0 < a_initial < a_final <= 1.")
        if args.command == "train":
            train(args, settings, device)
        else:
            simulate(args, settings, device)
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
