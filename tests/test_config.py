"""Tests for configuration schemas, TOML/JSON loading, validation, and CLI."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from cosmorl.cli import build_parser, main
from cosmorl.config import (
    EnvironmentConfig,
    ExperimentConfig,
    PolicyConfig,
    PPOConfig,
    SimulationConfig,
    load_config,
    write_default_config,
)


class ConfigTests(unittest.TestCase):
    def test_default_config_round_trip_toml(self) -> None:
        default = ExperimentConfig()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.toml"
            write_default_config(path, default)
            loaded = load_config(path)
            self.assertEqual(loaded, default)

    def test_json_config_loading_and_reward_every_terminal(self) -> None:
        payload = {
            "simulation": {"grid_size": 8, "box_mpc": 50.0, "a_initial": 0.1, "a_final": 1.0},
            "environment": {"budget": 40, "reward_every": "terminal"},
            "policy": {"kernel_size": 5, "conv_layers": 3, "residual": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exp.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config(path)
            self.assertEqual(cfg.simulation.resolved_particle_size, 8)
            self.assertEqual(cfg.environment.a_initial, 0.1)
            self.assertEqual(cfg.environment.reference_steps, 160)
            self.assertIsNone(cfg.environment.reward_every)
            self.assertTrue(cfg.policy.residual)

    def test_unknown_key_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentConfig.from_dict({"simulation": {"grid_size": 8, "unknown_param": 1}})

    def test_mismatched_scale_factor_bounds_raise(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentConfig.from_dict(
                {
                    "simulation": {"a_initial": 0.05, "a_final": 1.0},
                    "environment": {"a_initial": 0.1, "a_final": 1.0},
                }
            )

    def test_invalid_policy_and_ppo_parameters_raise(self) -> None:
        with self.assertRaises(ValueError):
            PolicyConfig(kernel_size=4)
        with self.assertRaises(ValueError):
            PPOConfig(gae_lambda=1.5)
        with self.assertRaises(ValueError):
            SimulationConfig(grid_size=1)

    def test_action_mask_respects_budget_and_merges_clipped_actions(self) -> None:
        config = EnvironmentConfig(a_initial=0.9, budget=2, reference_steps=8)
        targets = config.legal_targets(0.9, remaining=2)
        self.assertEqual(tuple(target is not None for target in targets), (False, False, True, True))
        self.assertEqual(config.legal_targets(targets[2], remaining=1), (None, None, None, 1.0))
        self.assertEqual(config.legal_targets(0.995, remaining=1), (1.0, None, None, None))

    def test_preset_configs_in_repo_are_valid(self) -> None:
        repo_configs = Path(__file__).resolve().parents[1] / "configs"
        for filename in ("default.toml", "simulate.toml", "train.toml", "evaluate.toml"):
            with self.subTest(config=filename):
                cfg = load_config(repo_configs / filename)
                self.assertIsInstance(cfg, ExperimentConfig)


class CliTests(unittest.TestCase):
    def test_init_and_validate_config_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "my_exp.toml"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["init-config", str(cfg_path)])
            self.assertEqual(exit_code, 0)
            self.assertTrue(cfg_path.is_file())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["validate-config", "-c", str(cfg_path)])
            self.assertEqual(exit_code, 0)
            self.assertIn("Configuration valid", stdout.getvalue())

            with self.assertRaises(SystemExit):
                main(["init-config", str(cfg_path)])

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["init-config", str(cfg_path), "--force"]), 0)

    def test_cli_subcommands_accept_only_config_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["train", "-c", "configs/train.toml"])
        self.assertEqual(args.command, "train")
        self.assertEqual(args.config, Path("configs/train.toml"))


if __name__ == "__main__":
    unittest.main()
