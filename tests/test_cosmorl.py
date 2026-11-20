import math
import unittest
from unittest.mock import patch

import torch

from cosmorl.environment import EpisodeConfig, FieldComparison, TimeStepEnvironment
from cosmorl.initial import Cosmology, InitialConditions
from cosmorl.pm import Background, Integrator, PeriodicMesh
from cosmorl.ppo import ActorCritic, VectorConvolution, generalized_advantage


CPU = torch.device("cpu")
DTYPE = torch.float64


class ParticleMeshTests(unittest.TestCase):
    def test_cic_conserves_mass_at_periodic_boundary(self):
        mesh = PeriodicMesh(8, 0.3, CPU)
        positions = torch.tensor([[15 / 16, 0, 0], [1, 0, 0]], dtype=DTYPE)
        stencil = mesh.stencil(positions)
        expected = torch.zeros(mesh.shape, dtype=DTYPE)
        expected[0, 0, 0], expected[7, 0, 0] = 1.5, 0.5
        torch.testing.assert_close(mesh.deposit(stencil), expected)
        torch.testing.assert_close(stencil.weights.sum(dim=1), torch.ones(2, dtype=DTYPE))

    def test_odd_grid_fourier_force_has_analytic_sign_and_normalization(self):
        mesh = PeriodicMesh(9, 0.3, CPU)
        phase = math.tau * torch.arange(9, dtype=DTYPE)[:, None, None] / 9
        delta = phase.cos().expand(mesh.shape)
        force = mesh.grid_force(delta)
        torch.testing.assert_close(force[0], (-0.45 / math.tau * phase.sin()).expand(mesh.shape))
        torch.testing.assert_close(force[1:], torch.zeros_like(force[1:]))

    def test_mixed_nyquist_mode_retains_transverse_force(self):
        mesh = PeriodicMesh(8, 0.3, CPU)
        x = torch.arange(8, dtype=DTYPE)[:, None, None]
        phase_y = math.tau * x.transpose(0, 1) / 8
        nyquist_mode = (math.pi * x).cos()
        delta = (nyquist_mode * phase_y.cos()).expand(mesh.shape)
        force = mesh.grid_force(delta)
        expected = (-0.45 / (17 * math.tau) * nyquist_mode * phase_y.sin()).expand(mesh.shape)
        torch.testing.assert_close(force[1], expected)
        torch.testing.assert_close(force[[0, 2]], torch.zeros_like(force[[0, 2]]))

    def test_kdk_uniform_translation_matches_eds_drift(self):
        mesh = PeriodicMesh(4, 1, CPU)
        axis = torch.arange(4, dtype=DTYPE) / 4
        positions = torch.cartesian_prod(axis, axis, axis)
        momentum = torch.full_like(positions, 0.03)
        solver = Integrator(mesh, Background(1, 0, 0))
        state = solver.initialize(0.2, positions, momentum)
        advanced = solver.step(state, 0.3)
        drift = 2 * (1 / math.sqrt(0.2) - 1 / math.sqrt(0.3))
        torch.testing.assert_close(advanced.positions, (positions + drift * momentum).remainder(1))
        torch.testing.assert_close(advanced.momentum, momentum)


class InitialConditionTests(unittest.TestCase):
    def test_class_realization_is_seeded_and_has_growing_mode_momentum(self):
        a = 0.05
        initial = InitialConditions(8, 100, a, Cosmology(), CPU)
        positions, momentum = initial.sample(13)
        repeated_positions, repeated_momentum = initial.sample(13)
        torch.testing.assert_close(positions, repeated_positions, atol=0, rtol=0)
        torch.testing.assert_close(momentum, repeated_momentum, atol=0, rtol=0)
        self.assertGreater(momentum.square().mean().item(), 0)
        self.assertTrue(((positions >= 0) & (positions < 1)).all().item())
        delta = initial.density_amplitude[1, 0, 0].item()
        u = initial.momentum_amplitude[1, 0, 0].item()
        background = initial.background
        expansion = math.sqrt(background.omega_m / a**3 + background.omega_r / a**4 + background.omega_lambda)
        self.assertAlmostEqual(u / (delta * a**2 * expansion), 1, delta=0.05)


class ReinforcementLearningTests(unittest.TestCase):
    def test_convolution_uses_periodic_diagonal_for_geometric_couplings(self):
        layer = VectorConvolution(1, 1)
        scalar = torch.zeros(1, 4, 4, 4, 1)
        vector = torch.zeros(1, 4, 4, 4, 3, 1)
        axial = torch.zeros_like(vector)
        scalar[0, 3, 3, 0, 0] = 2
        vector[0, 3, 3, 0, :, 0] = torch.tensor([1., 0., 1.])
        axial[0, 3, 3, 0, :, 0] = torch.tensor([0., 0., 1.])
        with torch.no_grad():
            for weight in layer.radial_weights.values():
                weight.zero_()
            for path in ("divergence", "gradient", "polar_projection", "polar_to_axial", "axial_to_polar"):
                layer.radial_weights[path][..., 1] = 12  # The Gaussian centered at sqrt(2).
            layer.radial_weights["polar"][..., 2] = 4  # Add back the isotropic part of the projection.

        scalar_out, vector_out, axial_out = layer(scalar, vector, axial)
        direction = torch.tensor([-1., -1., 0.]) / math.sqrt(2)
        polar_input = vector[0, 3, 3, 0, :, 0]
        axial_input = axial[0, 3, 3, 0, :, 0]
        dot = direction @ polar_input
        gate = dot.sigmoid()
        torch.testing.assert_close(scalar_out[0, 0, 0, 0, 0], gate * dot)
        torch.testing.assert_close(vector_out[0, 0, 0, 0, :, 0],
                                   gate * ((2 + dot) * direction + torch.linalg.cross(direction, axial_input)))
        torch.testing.assert_close(axial_out[0, 0, 0, 0, :, 0], gate * torch.linalg.cross(direction, polar_input))

    def test_convolution_shares_radial_filter_across_equally_distant_offsets(self):
        layer = VectorConvolution(1, 1, kernel_size=7)
        kernel = layer._kernel(input_axial=False, output_axial=True)
        axial_offset = kernel[..., 6, 3, 3]  # (3, 0, 0)
        diagonal_offset = kernel[..., 5, 5, 4]  # (2, 2, 1)
        torch.testing.assert_close(axial_offset[0, 0], diagonal_offset[0, 0])
        direction = torch.tensor([2., 2., 1.]) / 3
        torch.testing.assert_close(diagonal_offset[0, 1:], axial_offset[0, 1] * direction)

    def test_wide_convolution_preserves_grid_shape_and_symmetries(self):
        layer = VectorConvolution(2, 3, kernel_size=5)
        scalar = torch.randn(1, 6, 6, 6, 2)
        vector = torch.randn(1, 6, 6, 6, 3, 2)
        axial = torch.randn_like(vector)

        # Combine a reflection with a one-cell periodic translation.
        def reflect_x(field):
            return field.flip(1).roll(1, dims=2)

        signs = torch.tensor([-1., 1., 1.])[:, None]
        outputs = layer(scalar, vector, axial)
        transformed = layer(reflect_x(scalar), reflect_x(vector) * signs, -reflect_x(axial) * signs)
        expected = (reflect_x(outputs[0]), reflect_x(outputs[1]) * signs, -reflect_x(outputs[2]) * signs)
        self.assertEqual(outputs[0].shape, (1, 6, 6, 6, 3))
        for actual, target in zip(transformed, expected, strict=True):
            torch.testing.assert_close(actual, target)

    def test_action_mask_respects_budget_and_merges_clipped_actions(self):
        config = EpisodeConfig(a_initial=0.9, budget=2, reference_steps=8)
        targets = config.legal_targets(0.9, remaining=2)
        self.assertEqual(tuple(target is not None for target in targets), (False, False, True, True))
        self.assertEqual(config.legal_targets(targets[2], remaining=1), (None, None, None, 1.0))
        self.assertEqual(config.legal_targets(0.995, remaining=1), (1.0, None, None, None))

    def test_reward_interval_changes_timing_with_identical_total_and_reference(self):
        initial = InitialConditions(8, 100, 0.5, Cosmology(), CPU)
        solver = Integrator(PeriodicMesh(8, initial.background.omega_m, CPU), initial.background)
        for interval, scored_steps in ((None, (5,)), (1, (1, 2, 3, 4, 5)), (3, (3, 5)), (5, (5,))):
            with self.subTest(reward_every=interval):
                config = EpisodeConfig(a_initial=0.5, a_final=0.6, budget=5, reference_steps=13,
                                       actions=(0.04,), reward_every=interval)
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
                        self.assertEqual(result.reward, 0)
                    total += result.reward
                self.assertAlmostEqual(total, -result.errors.loss(config))
                if interval is None:
                    terminal_errors = result.errors
                self.assertEqual(result.errors, terminal_errors)
                reference = solver.advance_fixed(environment.initial_state, config.a_final, config.reference_steps)
                torch.testing.assert_close(
                    environment._reference_spectrum(config.a_final), environment.comparison.spectrum(reference.delta),
                )

    def test_dense_rewards_settle_failure_and_reset_between_episodes(self):
        config = EpisodeConfig(a_initial=0.5, a_final=0.6, budget=5, reference_steps=13, reward_every=1)
        initial = InitialConditions(8, 100, config.a_initial, Cosmology(), CPU)
        solver = Integrator(PeriodicMesh(8, initial.background.omega_m, CPU), initial.background)
        environment = TimeStepEnvironment(solver, initial, config)
        environment.reset(seed=13)
        first = environment.step(0)
        with patch.object(solver, "step", side_effect=FloatingPointError):
            terminal = environment.step(0)
        self.assertIsNone(terminal.observation)
        self.assertIsNone(terminal.errors)
        self.assertAlmostEqual(first.reward + terminal.reward, -2)
        environment.reset(seed=13)
        self.assertEqual(environment.step(0).reward, first.reward)

    def test_comparison_recovers_amplitude_and_phase_error(self):
        mesh = PeriodicMesh(8, 0.3, CPU)
        comparison = FieldComparison(mesh.q_squared, 8, 8)
        phase = math.tau * torch.arange(8, dtype=DTYPE)[:, None, None] / 8
        reference = comparison.spectrum(phase.cos().expand(mesh.shape))
        errors = comparison.compare(-2 * reference, reference)
        self.assertAlmostEqual(errors.density, 9)
        self.assertAlmostEqual(errors.power, 9)
        self.assertAlmostEqual(errors.correlation, -1)

    def test_policy_respects_reflection_and_action_mask(self):
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
                self.assertEqual(original.probs[0, 1].item(), 0)

    def test_policy_residual_switch_controls_shortcuts(self):
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
                for actual, target in zip(policy._encode(fields), expected, strict=True):
                    torch.testing.assert_close(actual, target)

    def test_terminal_gae_matches_undiscounted_rewards_at_lambda_one(self):
        values = torch.tensor([0.2, 0.3, 0.4])
        rewards = torch.tensor([0.0, 0.0, -0.7])
        advantage = generalized_advantage(values, rewards, gae_lambda=1)
        torch.testing.assert_close(advantage, -0.7 - values)


if __name__ == "__main__":
    unittest.main()
