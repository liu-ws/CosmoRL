"""Fixed-budget adaptive KDK time-stepping MDP environment."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import tau

import torch
from torch import Tensor

from ..config import EnvironmentConfig
from ..physics.cosmology import InitialConditions
from ..physics.pm import Integrator, ParticleState, fixed_targets


@dataclass(frozen=True)
class FieldErrors:
    """Band-limited field, power-spectrum, and cross-correlation diagnostics."""

    density: float
    power: float
    correlation: float

    def loss(self, config: EnvironmentConfig) -> float:
        """Compute bounded composite loss in [0, 1]."""
        density_term = self.density / (self.density + config.density_scale)
        power_term = self.power / (self.power + config.power_scale)
        return 0.5 * (density_term + power_term)


class FieldComparison:
    """Band-limited Fourier-space comparison between a candidate field and reference."""

    def __init__(self, q_squared: Tensor, mesh_size: int, particle_size: int) -> None:
        mode = q_squared.sqrt() / tau
        cutoff = 0.3 * min(mesh_size, particle_size) / 2.0
        self.indices = ((mode > 0) & (mode <= cutoff)).flatten().nonzero().flatten()
        if self.indices.numel() == 0:
            raise ValueError("The comparison band is empty. Increase grid_size and particle_size.")

        selected_modes = mode.flatten()[self.indices]
        self.bins = (selected_modes + 1e-12).floor().long() - 1
        # Only kz = 0 is self-conjugate within this sub-Nyquist band.
        kz = self.indices.remainder(mesh_size // 2 + 1)
        self.weights = 2.0 - (kz == 0).to(mode.dtype)
        self.counts = torch.bincount(self.bins, weights=self.weights, minlength=int(cutoff))
        self.valid_bins = (self.counts > 0).nonzero().flatten()

    def spectrum(self, delta: Tensor) -> Tensor:
        """Extract forward-normalized R2C Fourier coefficients inside the comparison band."""
        return torch.fft.rfftn(delta, norm="forward").flatten()[self.indices]

    def shell_average(self, values: Tensor) -> Tensor:
        """Compute conjugate-weighted spherical shell averages over valid wave-number bins."""
        summed = torch.zeros_like(self.counts).scatter_add_(0, self.bins, self.weights * values)
        return summed[self.valid_bins] / self.counts[self.valid_bins]

    def compare(self, spectrum: Tensor, reference: Tensor) -> FieldErrors:
        """Evaluate relative density error, shell power-spectrum error, and mean cross-correlation."""
        reference_energy = reference.abs().square()
        energy = spectrum.abs().square()
        eps = torch.finfo(spectrum.real.dtype).eps

        e_delta = torch.dot(self.weights, (spectrum - reference).abs().square())
        e_delta = e_delta / (torch.dot(self.weights, reference_energy) + eps)

        p_ref = self.shell_average(reference_energy)
        p = self.shell_average(energy)
        e_power = ((p - p_ref) / (p_ref + eps)).square().mean()

        cross = self.shell_average((spectrum * reference.conj()).real)
        correlation = (cross / (p * p_ref).sqrt().clamp_min(eps)).clamp(-1.0, 1.0).mean()
        return FieldErrors(*torch.stack((e_delta, e_power, correlation)).tolist())


@dataclass(frozen=True)
class Observation:
    """Grid-deposited physical fields, global scalar context, and legal action mask."""

    fields: Tensor
    scalars: Tensor
    action_mask: Tensor


@dataclass(frozen=True)
class Transition:
    """Result of one environment step."""

    observation: Observation | None
    reward: float
    errors: FieldErrors | None = None


class TimeStepEnvironment:
    """RL environment for choosing adaptive KDK log-scale-factor steps under a fixed PM budget."""

    def __init__(
        self,
        integrator: Integrator,
        initial: InitialConditions,
        config: EnvironmentConfig,
    ) -> None:
        self.integrator = integrator
        self.initial = initial
        self.config = config
        mesh = integrator.mesh
        self.comparison = FieldComparison(mesh.q_squared, mesh.size, initial.shape[0])
        assert config.reference_steps is not None
        self._reference_targets = tuple(fixed_targets(config.a_initial, config.a_final, config.reference_steps))

        self.initial_state: ParticleState | None = None
        self.state: ParticleState | None = None
        self.steps_used = 0
        self.targets: tuple[float | None, ...] = ()
        self._reference_anchor: ParticleState | None = None
        self._reference_index = 0
        self._final_reference: Tensor | None = None
        self._scored_loss = 0.0

    @torch.no_grad()
    def reset(self, seed: int) -> Observation:
        """Sample a new initial realization and return the opening observation."""
        self.initial_state = self.integrator.initialize(self.config.a_initial, *self.initial.sample(seed))
        self._reference_anchor = self.initial_state
        self._reference_index = 0
        self._final_reference = None
        self._scored_loss = 0.0
        self.state = self.initial_state
        self.steps_used = 0
        self.targets = self.config.legal_targets(self.state.a, self.config.budget)
        return self.observe(self.state)

    def observe(self, state: ParticleState) -> Observation:
        """Build the 4-channel (density + momentum) grid observation and action mask."""
        mesh = self.integrator.mesh
        momentum = mesh.deposit(state.stencil, state.momentum) * (mesh.size**3 / len(state.momentum))
        fields = torch.cat((state.delta.unsqueeze(0), momentum)).float()
        remaining_fraction = (self.config.budget - self.steps_used) / self.config.budget
        scalars = fields.new_tensor((state.a, remaining_fraction))
        action_mask = torch.tensor([target is not None for target in self.targets], device=fields.device)
        return Observation(fields=fields, scalars=scalars, action_mask=action_mask)

    @torch.no_grad()
    def step(self, action: int) -> Transition:
        """Advance the PM integrator by one selected discrete step size."""
        if self.state is None:
            raise RuntimeError("Reset the environment before stepping.")
        if not (0 <= action < len(self.targets)) or (target := self.targets[action]) is None:
            raise ValueError("Action is masked by the episode budget.")

        self.steps_used += 1
        try:
            state = self.integrator.step(self.state, target)
        except FloatingPointError:
            return self._finish()

        if state.a == self.config.a_final:
            return self._finish(self.errors(state))

        self.targets = self.config.legal_targets(state.a, self.config.budget - self.steps_used)
        if all(candidate is None for candidate in self.targets):
            return self._finish()

        self.state = state
        interval = self.config.reward_every
        if interval is not None and self.steps_used % interval == 0:
            errors = self.errors(state)
            return Transition(self.observe(state), self._reward(errors.loss(self.config)), errors)
        return Transition(self.observe(state), 0.0)

    def errors(self, state: ParticleState) -> FieldErrors:
        """Compare a particle state against the high-resolution reference trajectory at state.a."""
        return self.comparison.compare(self.comparison.spectrum(state.delta), self._reference_spectrum(state.a))

    def _reward(self, loss: float) -> float:
        reward = self._scored_loss - loss
        self._scored_loss = loss
        return reward

    def _finish(self, errors: FieldErrors | None = None) -> Transition:
        self.state = None
        loss = 2.0 if errors is None else errors.loss(self.config)
        return Transition(None, self._reward(loss), errors)

    def _reference_spectrum(self, a: float) -> Tensor:
        if a == self.config.a_final and self._final_reference is not None:
            return self._final_reference
        assert self.initial_state is not None and self._reference_anchor is not None

        end = bisect_right(self._reference_targets, a)
        if end < self._reference_index:
            anchor, start = self.initial_state, 0
        else:
            anchor, start = self._reference_anchor, self._reference_index

        for target in self._reference_targets[start:end]:
            anchor = self.integrator.step(anchor, target)
        if end >= self._reference_index:
            self._reference_anchor, self._reference_index = anchor, end

        # Probe the exact requested scale factor without mutating the fixed reference trajectory.
        reference = anchor if a == anchor.a else self.integrator.step(anchor, a)
        spectrum = self.comparison.spectrum(reference.delta)
        if a == self.config.a_final:
            self._final_reference = spectrum
        return spectrum
