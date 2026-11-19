"""A fixed-budget environment for adaptive KDK time stepping."""

from dataclasses import dataclass
from math import ceil, exp, isfinite, log, tau

import torch
from torch import Tensor

from .initial import InitialConditions
from .pm import Integrator, ParticleState


@dataclass(frozen=True)
class EpisodeConfig:
    a_initial: float = 0.05
    a_final: float = 1.0
    budget: int = 48
    reference_steps: int = 192
    actions: tuple[float, ...] = (0.01, 0.02, 0.04, 0.08)
    density_scale: float = 1e-6
    power_scale: float = 1e-6

    def __post_init__(self) -> None:
        if not 0 < self.a_initial < self.a_final <= 1:
            raise ValueError("Require 0 < a_initial < a_final <= 1.")
        if not self.actions or any(not isfinite(h) or h <= 0 for h in self.actions):
            raise ValueError("Time-step choices must be finite and positive.")
        if self.minimum_steps(self.a_initial) > self.budget:
            raise ValueError("The PM budget cannot reach a_final even using the largest time step.")
        if self.reference_steps <= self.budget:
            raise ValueError("The reference must use more PM steps than the policy budget.")
        if not all(isfinite(scale) and scale > 0 for scale in (self.density_scale, self.power_scale)):
            raise ValueError("Error scales must be finite and positive.")

    def minimum_steps(self, a: float) -> int:
        return ceil(log(self.a_final / a) / max(self.actions) - 1e-10)

    def legal_targets(self, a: float, remaining: int) -> tuple[float | None, ...]:
        interval = log(self.a_final / a)
        legal = []
        seen = set()
        for h in self.actions:
            target = self.a_final if h >= interval - 1e-12 else a * exp(h)
            if target in seen or target <= a or self.minimum_steps(target) >= remaining:
                legal.append(None)
            else:
                legal.append(target)
                seen.add(target)
        return tuple(legal)


@dataclass(frozen=True)
class FieldErrors:
    density: float
    power: float
    correlation: float

    def loss(self, config: EpisodeConfig) -> float:
        density_term = self.density / (self.density + config.density_scale)
        power_term = self.power / (self.power + config.power_scale)
        return 0.5 * (density_term + power_term)


class FieldComparison:
    def __init__(self, q_squared: Tensor, mesh_size: int, particle_size: int):
        mode = q_squared.sqrt() / tau
        cutoff = 0.3 * min(mesh_size, particle_size) / 2
        self.indices = ((mode > 0) & (mode <= cutoff)).flatten().nonzero().flatten()
        if self.indices.numel() == 0:
            raise ValueError("The comparison band is empty. Increase the grid and particle resolutions.")
        selected_modes = mode.flatten()[self.indices]
        self.bins = (selected_modes + 1e-12).floor().long() - 1
        # Only kz=0 is self-conjugate in this band, which is below Nyquist.
        kz = self.indices.remainder(mesh_size // 2 + 1)
        self.weights = 2 - (kz == 0).to(mode.dtype)
        self.counts = torch.bincount(self.bins, weights=self.weights, minlength=int(cutoff))
        self.valid_bins = (self.counts > 0).nonzero().flatten()

    def spectrum(self, delta: Tensor) -> Tensor:
        return torch.fft.rfftn(delta, norm="forward").flatten()[self.indices]

    def shell_average(self, values: Tensor) -> Tensor:
        summed = torch.zeros_like(self.counts).scatter_add_(0, self.bins, self.weights * values)
        return summed[self.valid_bins] / self.counts[self.valid_bins]

    def compare(self, spectrum: Tensor, reference: Tensor) -> FieldErrors:
        reference_energy = reference.abs().square()
        energy = spectrum.abs().square()
        eps = torch.finfo(spectrum.real.dtype).eps
        e_delta = torch.dot(self.weights, (spectrum - reference).abs().square())
        e_delta = e_delta / (torch.dot(self.weights, reference_energy) + eps)
        p_ref = self.shell_average(reference_energy)
        p = self.shell_average(energy)
        e_power = ((p - p_ref) / (p_ref + eps)).square().mean()
        cross = self.shell_average((spectrum * reference.conj()).real)
        correlation = (cross / (p * p_ref).sqrt().clamp_min(eps)).clamp(-1, 1).mean()
        return FieldErrors(*torch.stack((e_delta, e_power, correlation)).tolist())


@dataclass(frozen=True)
class Observation:
    fields: Tensor
    scalars: Tensor
    actions: Tensor


@dataclass(frozen=True)
class Transition:
    observation: Observation | None
    reward: float
    errors: FieldErrors | None = None


class TimeStepEnvironment:
    def __init__(self, integrator: Integrator, initial: InitialConditions, config: EpisodeConfig):
        self.integrator = integrator
        self.initial = initial
        self.config = config
        mesh = integrator.mesh
        self.comparison = FieldComparison(mesh.q_squared, mesh.size, initial.shape[0])
        self.state: ParticleState | None = None
        self.steps_used = 0
        self.targets: tuple[float | None, ...] = ()

    @torch.no_grad()
    def reset(self, seed: int) -> Observation:
        self.initial_state = self.integrator.initialize(self.config.a_initial, *self.initial.sample(seed))
        reference = self.integrator.advance_fixed(self.initial_state, self.config.a_final, self.config.reference_steps)
        self.reference = self.comparison.spectrum(reference.delta)
        self.state = self.initial_state
        self.steps_used = 0
        self.targets = self.config.legal_targets(self.state.a, self.config.budget)
        return self.observe(self.state)

    def observe(self, state: ParticleState) -> Observation:
        mesh = self.integrator.mesh
        momentum = mesh.deposit(state.stencil, state.momentum) * (mesh.size**3 / len(state.momentum))
        fields = torch.cat((state.delta.unsqueeze(0), momentum)).float()
        scalars = fields.new_tensor((state.a, (self.config.budget - self.steps_used) / self.config.budget))
        actions = torch.tensor([target is not None for target in self.targets], device=fields.device)
        return Observation(fields, scalars, actions)

    @torch.no_grad()
    def step(self, action: int) -> Transition:
        if self.state is None:
            raise RuntimeError("Reset the environment before stepping.")
        if not 0 <= action < len(self.targets) or (target := self.targets[action]) is None:
            raise ValueError("Action is masked by the episode budget.")
        self.steps_used += 1
        try:
            state = self.integrator.step(self.state, target)
        except FloatingPointError:
            self.state = None
            return Transition(None, -2)
        if state.a == self.config.a_final:
            self.state = None
            errors = self.errors(state)
            return Transition(None, -errors.loss(self.config), errors)
        self.targets = self.config.legal_targets(state.a, self.config.budget - self.steps_used)
        if all(target is None for target in self.targets):
            self.state = None
            return Transition(None, -2)
        self.state = state
        return Transition(self.observe(state), 0)

    def errors(self, state: ParticleState) -> FieldErrors:
        return self.comparison.compare(self.comparison.spectrum(state.delta), self.reference)
