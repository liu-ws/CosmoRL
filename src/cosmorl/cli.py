"""Config-driven command-line interface for CosmoRL."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .config import EnvironmentConfig, ExperimentConfig, load_config, write_default_config

if TYPE_CHECKING:
    from .rl.environment import FieldErrors


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CosmoRL CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="cosmorl",
        description="Config-driven Particle-Mesh cosmology simulation and PPO time-stepping.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_option(subparser: argparse.ArgumentParser, *, required: bool = False) -> None:
        subparser.add_argument(
            "-c",
            "--config",
            type=Path,
            required=required,
            help="Path to a TOML or JSON experiment configuration file (defaults to built-in values).",
        )

    sim_parser = subparsers.add_parser("simulate", help="Run a fixed-step Particle-Mesh simulation.")
    add_config_option(sim_parser)

    train_parser = subparsers.add_parser("train", help="Train an O(3)-equivariant PPO time-step policy.")
    add_config_option(train_parser)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a policy checkpoint against fixed steps.")
    add_config_option(eval_parser)

    init_parser = subparsers.add_parser("init-config", help="Generate a default commented TOML configuration file.")
    init_parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("configs/experiment.toml"),
        help="Destination path for the generated TOML file (default: configs/experiment.toml).",
    )
    init_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )

    validate_parser = subparsers.add_parser("validate-config", help="Validate a configuration file and print a summary.")
    add_config_option(validate_parser, required=True)

    return parser


def _format_errors(errors: FieldErrors | None, env_config: EnvironmentConfig) -> str:
    if errors is None:
        return "failed"
    return (
        f"loss={errors.loss(env_config):.6g} "
        f"e_delta={errors.density:.6g} "
        f"e_power={errors.power:.6g} "
        f"cross_r={errors.correlation:.6g}"
    )


def _print_config_summary(config: ExperimentConfig, source: Path) -> None:
    sim = config.simulation
    env = config.environment
    pol = config.policy
    print(f"Configuration valid: {source}")
    print(
        f"  runtime:    device={config.runtime.device}, threads={config.runtime.threads}\n"
        f"  simulation: grid={sim.grid_size}^3, particles={sim.resolved_particle_size}^3, "
        f"box={sim.box_mpc:g} Mpc, a=[{sim.a_initial:g}, {sim.a_final:g}], steps={sim.steps}\n"
        f"  mdp:        budget={env.budget}, reference_steps={env.reference_steps}, "
        f"actions={list(env.actions)}, reward_every={env.reward_every or 'terminal'}\n"
        f"  policy:     channels={pol.channels}, kernel_size={pol.kernel_size}, "
        f"conv_layers={pol.conv_layers}, residual={pol.residual}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `cosmorl` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init-config":
            if args.output.exists() and not args.force:
                raise SystemExit(
                    f"Refusing to overwrite existing file '{args.output}'. Pass --force to overwrite."
                )
            path = write_default_config(args.output)
            print(f"Wrote default configuration to {path}")
            return 0

        if args.command == "validate-config":
            config = load_config(args.config)
            _print_config_summary(config, args.config)
            return 0

        config = load_config(args.config)

        # Defer heavy torch/classy imports until an execution command is requested.
        from .workflows import run_evaluation, run_simulation, run_training

        if args.command == "simulate":
            result = run_simulation(config)
            print(
                f"device={result.device} a={result.a_final:g} particles={result.num_particles} "
                f"pm_steps={result.pm_steps} initial_delta_rms={result.initial_delta_rms:.6g} "
                f"final_delta_rms={result.final_delta_rms:.6g} mean_delta={result.mean_delta:.3g}"
            )
            if result.output_path is not None:
                print(f"snapshot={result.output_path}")
            return 0

        if args.command == "train":
            result = run_training(
                config,
                on_update=lambda m: print(
                    f"update={m.update} reward={m.mean_reward:.6g} "
                    f"mean_pm_steps={m.mean_pm_steps:g} ppo_loss={m.ppo_loss:.6g}",
                    flush=True,
                ),
            )
            print(f"checkpoint={result.checkpoint_path}")
            return 0

        if args.command == "evaluate":
            result = run_evaluation(config)
            for item in result.episodes:
                print(
                    f"seed={item.seed} policy pm_steps={item.policy_steps} "
                    f"{_format_errors(item.policy_errors, result.environment)}"
                )
                print(
                    f"seed={item.seed} fixed  pm_steps={item.fixed_steps} "
                    f"{_format_errors(item.fixed_errors, result.environment)}"
                )
            return 0

        parser.error(f"Unknown command: {args.command}")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error
    return 0
