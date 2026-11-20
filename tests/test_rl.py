"""Unit tests for O(3)-equivariant convolutions, RL environment, rewards, and GAE."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

from cosmorl.config import CosmologyConfig, EnvironmentConfig
from cosmorl.physics import InitialConditions, Integrator, PeriodicMesh
from cosmorl.rl import (
    ActorCritic,
    EquivariantConv3d,
    FieldComparison,
    TimeStepEnvironment,
    compute_gae,
)

CPU = torch.device("cpu")
DTYPE = torch.float64


class EquivariantNetworkTests(unittest.TestCase):
    def test_convolution_uses_periodic_diagonal_for_geometric_couplings(self) -> None:
        layer = EquivariantConv3d(1, 1)
        scalar = torch.zeros(1, 4, 4, 4, 1)
        vector = torch.zeros(1, 4, 4, 4, 3, 1)
        axial = torch.zeros_like(vector)
        scalar[0, 3, 3, 0, 0] = 2.0
        vector[0, 3, 3, 0, :, 0] = torch.tensor([1.0, 0.0, 1.0])
        axial[0, 3, 3, 0, :, 0] = torch.tensor([0.0, 0.0, 1.0])
        with torch.no_grad():
            for weight in layer.radial_weights.values():
                weight.zero_()
            for path in ("divergence", "gradient", "polar_projection", "polar_to_axial", "axial_to_polar"):
                layer.radial_weights[path][..., 1] = 12.0  # Gaussian centered at sqrt(2).
            layer.radial_weights["polar"][..., 2] = 4.0  # Restore isotropic part of n n^T.

        scalar_out, vector_out, axial_out = layer(scalar, vector, axial)
        direction = torch.tensor([-1.0, -1.0, 0.0]) / math.sqrt(2.0)
        polar_input = vector[0, 3, 3, 0, :, 0]
        axial_input = axial[0, 3, 3, 0, :, 0]
        dot = direction @ polar_input
        gate = dot.sigmoid()
        torch.testing.assert_close(scalar_out[0, 0, 0, 0, 0], gate * dot)
        torch.testing.assert_close(
            vector_out[0, 0, 0, 0, :, 0],
            gate * ((2.0 + dot) * direction + torch.linalg.cross(direction, axial_input)),
        )
        torch.testing.assert_close(
            axial_out[0, 0, 0, 0, :, 0],
            gate * torch.linalg.cross(direction, polar_input),
        )

    def test_convolution_shares_radial_filter_across_equally_distant_offsets(self) -> None:
        layer = EquivariantConv3d(1, 1, kernel_size=7)
        kernel = layer.build_kernel(input_axial=False, output_axial=True)
        axial_offset = kernel[..., 6, 3, 3]  # (3, 0, 0)
        diagonal_offset = kernel[..., 5, 5, 4]  # (2, 2, 1)
        torch.testing.assert_close(axial_offset[0, 0], diagonal_offset[0, 0])
        direction = torch.tensor([2.0, 2.0, 1.0]) / 3.0
        torch.testing.assert_close(diagonal_offset[0, 1:], axial_offset[0, 1] * direction)

    def test_wide_convolution_preserves_grid_shape_and_o3_symmetries(self) -> None:
        layer = EquivariantConv3d(2, 3, kernel_size=5)
        scalar = torch.randn(1, 6, 6, 6, 2)
        vector = torch.randn(1, 6, 6, 6, 3, 2)
        axial = torch.randn_like(vector)

        def reflect_x(field: torch.Tensor) -> torch.Tensor:
            return field.flip(1).roll(1, dims=2)

        signs = torch.tensor([-1.0, 1.0, 1.0])[:, None]
        outputs = layer(scalar, vector, axial)
        transformed = layer(reflect_x(scalar), reflect_x(vector) * signs, -reflect_x(axial) * signs)
        expected = (reflect_x(outputs[0]), reflect_x(outputs[1]) * signs, -reflect_x(outputs[2]) * signs)
        self.assertEqual(outputs[0].shape, (1, 6, 6, 6, 3))
        for actual, target in zip(transformed, expected, strict=True):
            torch.testing.assert_close(actual, target)

    def test_policy_respects_reflection_and_action_mask(self) -> None:
        fields = torch.randn(1, 4, 3, 3, 3)
        scalars = torch.tensor([[0.1, 0.9]])
        allowed = torch.tensor([[True, False, True, True]])
        reflected = fields.flip(2)
        reflected[:, 1] *= -1
        for depth, residual in ((2, False), (4, True), (1, True)):
            with self.subTest(depth=depth, residual=residual):
                policy = ActorCritic(4, conv_layers=depth, residual=residual)
                original, v_original = policy(fields, scalars, allowed)
                transformed, v_transformed = policy(reflected, scalars, allowed)
                torch.testing.assert_close(original.probs, transformed.probs)
                torch.testing.assert_close(v_original, v_transformed)
                self.assertEqual(original.probs[0, 1].item(), 0.0)

    def test_policy_residual_switch_controls_shortcuts(self) -> None:
        fields = torch.randn(1, 4, 3, 3, 3)
        scalar = fields[:, :1].movedim(1, -1)
        polar = fields[:, 1:].movedim(1, -1).unsqueeze(-1)
        for residual in (False, True):
            with self.subTest(residual=residual):
                policy = ActorCritic(4, channels=2, conv_layers=4, residual=residual)
                self.assertEqual(len(policy.convolutions), 4)
                with torch.no_grad():
                    for layer in policy.convolutions[1:]:
                        for parameter in layer.parameters():
                            parameter.zero_()
                expected = policy.convolutions[0](scalar, polar, return_axial=True)
                if not residual:
                    expected = tuple(torch.zeros_like(field) for field in expected)
                for actual, target in zip(policy.encode(fields), expected, strict=True):
                    torch.testing.assert_close(actual, target)


class EnvironmentAndPPOTests(unittest.TestCase):
    def test_reward_interval_changes_timing_with_identical_total_and_reference(self) -> None:
        initial = InitialConditions(8, 100.0, 0.5, CosmologyConfig(), CPU)
        solver = Integrator(PeriodicMesh(8, initial.background.omega_m, CPU), initial.background)
        terminal_errors = None
        for interval, scored_steps in ((None, (5,)), (1, (1, 2, 3, 4, 5)), (3, (3, 5)), (5, (5,))):
            with self.subTest(reward_every=interval):
                config = EnvironmentConfig(
                    a_initial=0.5,
                    a_final=0.6,
                    budget=5,
                    reference_steps=13,
                    actions=(0.04,),
                    reward_every=interval,
                )
                environment = TimeStepEnvironment(solver, initial, config)
                environment.reset(seed=13)
                total, last_loss = 0.0, 0.0
                for step in range(1, 6):
                    result = environment.step(0)
                    self.assertEqual(result.observation is None, step == 5)
                    if step in scored_steps:
                        self.assertIsNotNone(result.errors)
                        loss = result.errors.loss(config)
                        self.assertAlmostEqual(result.reward, last_loss - loss)
                        last_loss = loss
                    else:
                        self.assertIsNone(result.errors)
                        self.assertEqual(result.reward, 0.0)
                    total += result.reward
                assert result.errors is not None
                self.assertAlmostEqual(total, -result.errors.loss(config))
                if interval is None:
                    terminal_errors = result.errors
                self.assertEqual(result.errors, terminal_errors)
                assert environment.initial_state is not None
                reference = solver.advance_fixed(environment.initial_state, config.a_final, config.reference_steps)
                torch.testing.assert_close(
                    environment._reference_spectrum(config.a_final),
                    environment.comparison.spectrum(reference.delta),
                )

    def test_dense_rewards_settle_failure_and_reset_between_episodes(self) -> None:
        config = EnvironmentConfig(a_initial=0.5, a_final=0.6, budget=5, reference_steps=13, reward_every=1)
        initial = InitialConditions(8, 100.0, config.a_initial, CosmologyConfig(), CPU)
        solver = Integrator(PeriodicMesh(8, initial.background.omega_m, CPU), initial.background)
        environment = TimeStepEnvironment(solver, initial, config)
        environment.reset(seed=13)
        first = environment.step(0)
        with patch.object(solver, "step", side_effect=FloatingPointError):
            terminal = environment.step(0)
        self.assertIsNone(terminal.observation)
        self.assertIsNone(terminal.errors)
        self.assertAlmostEqual(first.reward + terminal.reward, -2.0)
        environment.reset(seed=13)
        self.assertEqual(environment.step(0).reward, first.reward)

    def test_comparison_recovers_amplitude_and_phase_error(self) -> None:
        mesh = PeriodicMesh(8, 0.3, CPU)
        comparison = FieldComparison(mesh.q_squared, 8, 8)
        phase = math.tau * torch.arange(8, dtype=DTYPE)[:, None, None] / 8
        reference = comparison.spectrum(phase.cos().expand(mesh.shape))
        errors = comparison.compare(-2.0 * reference, reference)
        self.assertAlmostEqual(errors.density, 9.0)
        self.assertAlmostEqual(errors.power, 9.0)
        self.assertAlmostEqual(errors.correlation, -1.0)

    def test_terminal_gae_matches_undiscounted_rewards_at_lambda_one(self) -> None:
        values = torch.tensor([0.2, 0.3, 0.4])
        rewards = torch.tensor([0.0, 0.0, -0.7])
        advantage = compute_gae(values, rewards, gae_lambda=1.0)
        torch.testing.assert_close(advantage, -0.7 - values)


if __name__ == "__main__":
    unittest.main()
