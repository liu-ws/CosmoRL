import math
import unittest

import torch

from cosmorl.environment import EpisodeConfig, FieldComparison
from cosmorl.initial import Cosmology, InitialConditions
from cosmorl.pm import Background, Integrator, PeriodicMesh
from cosmorl.ppo import ActorCritic, generalized_advantage


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
    def test_action_mask_respects_budget_and_merges_clipped_actions(self):
        config = EpisodeConfig(a_initial=0.9, budget=2, reference_steps=8)
        targets = config.legal_targets(0.9, remaining=2)
        self.assertEqual(tuple(target is not None for target in targets), (False, False, True, True))
        self.assertEqual(config.legal_targets(targets[2], remaining=1), (None, None, None, 1.0))
        self.assertEqual(config.legal_targets(0.995, remaining=1), (1.0, None, None, None))

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
        policy = ActorCritic(4)
        fields = torch.randn(1, 4, 3, 3, 3)
        scalars = torch.tensor([[0.1, 0.9]])
        allowed = torch.tensor([[True, False, True, True]])
        reflected = fields.flip(2)
        reflected[:, 1] *= -1
        original, v_original = policy(fields, scalars, allowed)
        transformed, v_transformed = policy(reflected, scalars, allowed)
        torch.testing.assert_close(original.probs, transformed.probs)
        torch.testing.assert_close(v_original, v_transformed)
        self.assertEqual(original.probs[0, 1].item(), 0)

    def test_terminal_gae_matches_undiscounted_rewards_at_lambda_one(self):
        values = torch.tensor([0.2, 0.3, 0.4])
        rewards = torch.tensor([0.0, 0.0, -0.7])
        advantage = generalized_advantage(values, rewards, gae_lambda=1)
        torch.testing.assert_close(advantage, -0.7 - values)


if __name__ == "__main__":
    unittest.main()
