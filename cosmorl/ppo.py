"""Periodic scalar/vector encoder and masked PPO for complete episodes."""

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from .environment import Observation, TimeStepEnvironment, Transition


class VectorConvolution(nn.Module):
    """Periodic convolution with radial filters and Cartesian angular factors."""

    _ISOTROPIC_PATHS = ("scalar", "polar", "axial")

    def __init__(
        self, input_channels: int, output_channels: int, *,
        kernel_size: int = 3, input_axial: bool = True,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.kernel_size = kernel_size
        self.input_axial = input_axial
        radius = kernel_size // 2
        self.padding = radius

        axis = torch.arange(-radius, radius + 1)
        offsets = torch.cartesian_prod(axis, axis, axis)
        squared_radii, radius_index, counts = offsets.square().sum(dim=-1).unique(
            return_inverse=True, return_counts=True,
        )
        radii = squared_radii.to(torch.float64).sqrt()
        # Gaussian basis on the distinct neighbor distances, with a separate center term.
        width = radii.diff(prepend=radii.new_tensor([-1])).amin()
        distance = radii[1:, None] - radii[None, 1:]
        radial_basis = F.pad((-(distance / width).square()).exp(), (1, 0, 1, 0))
        radial_basis[0, 0] = 1
        # Compute normalized direction vectors
        directions = F.normalize(offsets.to(torch.get_default_dtype()), dim=-1)
        identity = torch.eye(3, dtype=directions.dtype)
        self.register_buffer("radius_index", radius_index, persistent=False)
        self.register_buffer("radial_basis", radial_basis.to(directions), persistent=False)
        self.register_buffer("normalization", counts[radius_index].to(directions).reciprocal(), persistent=False)
        self.register_buffer("direction", directions.T, persistent=False)
        quadrupole = torch.einsum("ki,kj->ijk", directions, directions)
        quadrupole = quadrupole - identity[..., None] * (radius_index != 0) / 3
        self.register_buffer("quadrupole", quadrupole, persistent=False)
        cross = torch.linalg.cross(directions[:, None, :], identity[None, :, :], dim=-1)
        self.register_buffer("cross", cross.permute(2, 1, 0), persistent=False)
        self.register_buffer("identity", identity.unsqueeze(-1), persistent=False)

        self.radial_weights = nn.ParameterDict()
        bound = 1 / sqrt(3 * input_channels * len(radii))
        paths = ("scalar", "polar", "divergence", "gradient", "polar_projection", "polar_to_axial")
        if input_axial:
            paths += ("axial", "axial_projection", "axial_to_polar")
        for name in paths:
            # Directional terms have no contribution at the kernel center.
            basis_count = len(radii) - (name not in self._ISOTROPIC_PATHS)
            self.radial_weights[name] = nn.Parameter(
                torch.empty(output_channels, input_channels, basis_count).uniform_(-bound, bound)
            )
        self.scalar_bias = nn.Parameter(torch.zeros(output_channels))

    def _kernel(self, input_axial: bool, output_axial: bool) -> Tensor:
        def term(name: str, geometry: Tensor | float = 1.0) -> Tensor:
            weights = self.radial_weights[name]
            if name not in self._ISOTROPIC_PATHS:
                weights = F.pad(weights, (1, 0))
            radial = (weights @ self.radial_basis).index_select(-1, self.radius_index) * self.normalization
            return radial[..., None, None, :] * geometry

        scalar = term("scalar")
        scalar_row = [scalar, term("divergence", self.direction[None, :, :])]
        polar_row = [term("gradient", self.direction[:, None, :]),
                     term("polar", self.identity) + term("polar_projection", self.quadrupole)]
        if input_axial:
            scalar_row.append(torch.zeros_like(scalar_row[1]))
            polar_row.append(term("axial_to_polar", self.cross))
        rows = [scalar_row, polar_row]
        if output_axial:
            axial_row = [torch.zeros_like(polar_row[0]), term("polar_to_axial", self.cross)]
            if input_axial:
                axial_row.append(term("axial", self.identity) + term("axial_projection", self.quadrupole))
            rows.append(axial_row)

        kernel = torch.cat([torch.cat(row, dim=3) for row in rows], dim=2)
        out_channels, in_channels, out_components, in_components, _ = kernel.shape
        return kernel.permute(2, 0, 3, 1, 4).reshape(
            out_components * out_channels, in_components * in_channels, *((self.kernel_size,) * 3),
        )

    def forward(
        self, scalar: Tensor, vector: Tensor, axial: Tensor | None = None, *, return_axial: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Return scalar and polar fields, plus the axial field when supplied or requested."""
        output_axial = return_axial or axial is not None
        parts = [scalar.unsqueeze(-2), vector]
        if axial is not None:
            if not self.input_axial:
                raise ValueError("This convolution does not accept an axial input.")
            parts.append(axial)
        fields = torch.cat(parts, dim=-2).flatten(-2).movedim(-1, 1)

        if self.padding:
            for dim, size in enumerate(scalar.shape[1:4], start=2):
                indices = torch.arange(-self.padding, size + self.padding, device=fields.device).remainder(size)
                fields = fields.index_select(dim, indices)

        output = F.conv3d(fields, self._kernel(axial is not None, output_axial))
        fields = output.movedim(1, -1).unflatten(-1, (-1, self.scalar_bias.numel()))
        scalar = fields[..., 0, :] + self.scalar_bias
        gate = scalar.sigmoid()
        vectors = fields[..., 1:, :] * gate.unsqueeze(-2)
        if output_axial:
            return scalar * gate, vectors[..., :3, :], vectors[..., 3:, :]
        return scalar * gate, vectors


class ActorCritic(nn.Module):
    def __init__(
        self, actions: int, channels: int = 8, *,
        kernel_size: int = 3, conv_layers: int = 2, residual: bool = False,
    ):
        super().__init__()
        if type(conv_layers) is not int or conv_layers < 1:
            raise ValueError("conv_layers must be a positive integer.")
        if not isinstance(residual, bool):
            raise ValueError(f"residual must be a bool, got {residual!r}.")
        self.residual = residual
        self.convolutions = nn.ModuleList(
            VectorConvolution(1 if index == 0 else channels, channels, kernel_size=kernel_size, input_axial=index > 0)
            for index in range(conv_layers)
        )
        self.shared = nn.Sequential(nn.Linear(3 * channels + 2, 32), nn.Tanh())
        self.actor = nn.Linear(32, actions)
        self.critic = nn.Linear(32, 1)

    def _encode(self, fields: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        scalar = fields[:, :1].movedim(1, -1)
        polar = fields[:, 1:].movedim(1, -1).unsqueeze(-1)
        layers = iter(self.convolutions)
        scalar, polar, axial = next(layers)(scalar, polar, return_axial=True)
        for layer in layers:
            next_scalar, next_polar, next_axial = layer(scalar, polar, axial)
            if self.residual:
                next_scalar = scalar + next_scalar
                next_polar = polar + next_polar
                next_axial = axial + next_axial
            scalar, polar, axial = next_scalar, next_polar, next_axial
        return scalar, polar, axial

    def forward(self, fields: Tensor, scalars: Tensor, allowed: Tensor) -> tuple[Categorical, Tensor]:
        scalar, polar, axial = self._encode(fields)
        mean_scalar = scalar.mean(dim=(1, 2, 3))
        mean_vector_power = polar.square().sum(dim=-2).mean(dim=(1, 2, 3))
        mean_axial_power = axial.square().sum(dim=-2).mean(dim=(1, 2, 3))
        features = self.shared(torch.cat((mean_scalar, mean_vector_power, scalars, mean_axial_power), dim=-1))
        logits = self.actor(features).masked_fill(~allowed, -torch.inf)
        return Categorical(logits=logits), self.critic(features).squeeze(-1)

    def single(self, observation: Observation) -> tuple[Categorical, Tensor]:
        return self(observation.fields[None], observation.scalars[None], observation.actions[None])


def generalized_advantage(values: Tensor, rewards: Tensor, gae_lambda: float = 0.95) -> Tensor:
    """GAE with gamma=1 and zero bootstrap after the true terminal transition."""
    next_value = values.new_zeros(())
    next_advantage = values.new_zeros(())
    advantages = []
    for value, reward in zip(reversed(values), reversed(rewards), strict=True):
        next_advantage = reward + next_value - value + gae_lambda * next_advantage
        advantages.append(next_advantage)
        next_value = value
    return torch.stack(advantages[::-1])


@dataclass
class Rollout:
    observations: list[Observation]
    actions: Tensor
    log_probabilities: Tensor
    advantages: Tensor
    returns: Tensor
    reward: float


@torch.no_grad()
def collect_episode(environment: TimeStepEnvironment, policy: ActorCritic, seed: int) -> Rollout:
    observations, actions, log_probabilities, values, rewards = [], [], [], [], []
    observation = environment.reset(seed)
    while observation is not None:
        distribution, value = policy.single(observation)
        action = distribution.sample()
        transition = environment.step(int(action.item()))
        observations.append(observation)
        actions.append(action)
        log_probabilities.append(distribution.log_prob(action))
        values.append(value)
        rewards.append(transition.reward)
        observation = transition.observation
    values = torch.cat(values)
    advantages = generalized_advantage(values, values.new_tensor(rewards))
    return Rollout(
        observations, torch.cat(actions), torch.cat(log_probabilities),
        advantages, advantages + values, sum(rewards),
    )


def update_policy(
    policy: ActorCritic, optimizer: torch.optim.Optimizer, rollouts: list[Rollout], epochs: int = 4,
    minibatch_size: int = 8,
) -> float:
    observations = [observation for rollout in rollouts for observation in rollout.observations]
    actions = torch.cat([rollout.actions for rollout in rollouts])
    old_log_prob = torch.cat([rollout.log_probabilities for rollout in rollouts])
    advantages = torch.cat([rollout.advantages for rollout in rollouts])
    returns = torch.cat([rollout.returns for rollout in rollouts])
    advantages = (advantages - advantages.mean()) / advantages.std(correction=0).clamp_min(1e-8)
    losses = []
    for _ in range(epochs):
        for indices in torch.randperm(len(observations)).split(minibatch_size):
            fields, scalars, masks = [], [], []
            for index in indices.tolist():
                saved = observations[index]
                fields.append(saved.fields)
                scalars.append(saved.scalars)
                masks.append(saved.actions)
            distribution, values = policy(torch.stack(fields), torch.stack(scalars), torch.stack(masks))
            weights = (distribution.log_prob(actions[indices]) - old_log_prob[indices]).exp()
            advantage = advantages[indices]
            actor_loss = -torch.minimum(weights * advantage, weights.clamp(0.8, 1.2) * advantage).mean()
            critic_loss = (values - returns[indices]).square().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * distribution.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
    return torch.stack(losses).mean().item()


@torch.no_grad()
def evaluate_episode(environment: TimeStepEnvironment, policy: ActorCritic, seed: int) -> Transition:
    observation = environment.reset(seed)
    while True:
        distribution, _ = policy.single(observation)
        transition = environment.step(int(distribution.logits.argmax(dim=-1).item()))
        if transition.observation is None:
            return transition
        observation = transition.observation
