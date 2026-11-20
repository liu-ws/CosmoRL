"""Unit tests for Particle-Mesh gravity, KDK integration, and CLASS initial conditions."""

from __future__ import annotations

import math
import unittest

import torch

from cosmorl.config import CosmologyConfig
from cosmorl.physics import Background, InitialConditions, Integrator, PeriodicMesh

CPU = torch.device("cpu")
DTYPE = torch.float64


class ParticleMeshTests(unittest.TestCase):
    def test_cic_conserves_mass_at_periodic_boundary(self) -> None:
        mesh = PeriodicMesh(8, 0.3, CPU)
        positions = torch.tensor([[15 / 16, 0, 0], [1, 0, 0]], dtype=DTYPE)
        stencil = mesh.stencil(positions)
        expected = torch.zeros(mesh.shape, dtype=DTYPE)
        expected[0, 0, 0], expected[7, 0, 0] = 1.5, 0.5
        torch.testing.assert_close(mesh.deposit(stencil), expected)
        torch.testing.assert_close(stencil.weights.sum(dim=1), torch.ones(2, dtype=DTYPE))

    def test_odd_grid_fourier_force_has_analytic_sign_and_normalization(self) -> None:
        mesh = PeriodicMesh(9, 0.3, CPU)
        phase = math.tau * torch.arange(9, dtype=DTYPE)[:, None, None] / 9
        delta = phase.cos().expand(mesh.shape)
        force = mesh.grid_force(delta)
        torch.testing.assert_close(force[0], (-0.45 / math.tau * phase.sin()).expand(mesh.shape))
        torch.testing.assert_close(force[1:], torch.zeros_like(force[1:]))

    def test_mixed_nyquist_mode_retains_transverse_force(self) -> None:
        mesh = PeriodicMesh(8, 0.3, CPU)
        x = torch.arange(8, dtype=DTYPE)[:, None, None]
        phase_y = math.tau * x.transpose(0, 1) / 8
        nyquist_mode = (math.pi * x).cos()
        delta = (nyquist_mode * phase_y.cos()).expand(mesh.shape)
        force = mesh.grid_force(delta)
        expected = (-0.45 / (17 * math.tau) * nyquist_mode * phase_y.sin()).expand(mesh.shape)
        torch.testing.assert_close(force[1], expected)
        torch.testing.assert_close(force[[0, 2]], torch.zeros_like(force[[0, 2]]))

    def test_kdk_uniform_translation_matches_eds_drift(self) -> None:
        mesh = PeriodicMesh(4, 1.0, CPU)
        axis = torch.arange(4, dtype=DTYPE) / 4
        positions = torch.cartesian_prod(axis, axis, axis)
        momentum = torch.full_like(positions, 0.03)
        solver = Integrator(mesh, Background(1.0, 0.0, 0.0))
        state = solver.initialize(0.2, positions, momentum)
        advanced = solver.step(state, 0.3)
        drift = 2.0 * (1.0 / math.sqrt(0.2) - 1.0 / math.sqrt(0.3))
        torch.testing.assert_close(advanced.positions, (positions + drift * momentum).remainder(1.0))
        torch.testing.assert_close(advanced.momentum, momentum)


class InitialConditionTests(unittest.TestCase):
    def test_class_realization_is_seeded_and_has_growing_mode_momentum(self) -> None:
        a = 0.05
        initial = InitialConditions(8, 100.0, a, CosmologyConfig(), CPU)
        positions, momentum = initial.sample(13)
        repeated_positions, repeated_momentum = initial.sample(13)
        torch.testing.assert_close(positions, repeated_positions, atol=0, rtol=0)
        torch.testing.assert_close(momentum, repeated_momentum, atol=0, rtol=0)
        self.assertGreater(momentum.square().mean().item(), 0)
        self.assertTrue(((positions >= 0) & (positions < 1)).all().item())
        delta = initial.density_amplitude[1, 0, 0].item()
        u = initial.momentum_amplitude[1, 0, 0].item()
        expansion = float(initial.background.expansion(a))
        self.assertAlmostEqual(u / (delta * a**2 * expansion), 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
