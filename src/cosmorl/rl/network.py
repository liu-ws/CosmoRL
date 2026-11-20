"""O(3)-equivariant periodic 3D convolution and invariant Actor-Critic network."""

from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from ..config import PolicyConfig
from .environment import Observation


class EquivariantConv3d(nn.Module):
    """Periodic 3D O(3)-equivariant convolution over scalar, polar, and axial vector fields.

    Decomposes grid offsets into Euclidean radius r and unit direction n, expanding radial
    filters on Gaussian shells and angular couplings on Cartesian irreps: 1, n, Q(n) = n n^T - I/3,
    and the cross-product operator [n]_x.
    """

    _ISOTROPIC_PATHS = ("scalar", "polar", "axial")

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        input_axial: bool = True,
    ) -> None:
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
            return_inverse=True,
            return_counts=True,
        )
        radii = squared_radii.to(torch.float64).sqrt()

        # Gaussian radial basis on distinct neighbor shells, plus an isolated center term at r = 0.
        width = radii.diff(prepend=radii.new_tensor([-1.0])).amin()
        distance = radii[1:, None] - radii[None, 1:]
        radial_basis = F.pad((-(distance / width).square()).exp(), (1, 0, 1, 0))
        radial_basis[0, 0] = 1.0

        directions = F.normalize(offsets.to(torch.get_default_dtype()), dim=-1)
        identity = torch.eye(3, dtype=directions.dtype)
        quadrupole = torch.einsum("ki,kj->ijk", directions, directions)
        quadrupole = quadrupole - identity[..., None] * (radius_index != 0) / 3.0
        cross = torch.linalg.cross(directions[:, None, :], identity[None, :, :], dim=-1)

        self.register_buffer("radius_index", radius_index, persistent=False)
        self.register_buffer("radial_basis", radial_basis.to(directions), persistent=False)
        self.register_buffer("normalization", counts[radius_index].to(directions).reciprocal(), persistent=False)
        self.register_buffer("direction", directions.T, persistent=False)
        self.register_buffer("quadrupole", quadrupole, persistent=False)
        self.register_buffer("cross", cross.permute(2, 1, 0), persistent=False)
        self.register_buffer("identity", identity.unsqueeze(-1), persistent=False)

        self.radial_weights = nn.ParameterDict()
        bound = 1.0 / sqrt(3 * in_channels * len(radii))
        paths = ("scalar", "polar", "divergence", "gradient", "polar_projection", "polar_to_axial")
        if input_axial:
            paths += ("axial", "axial_projection", "axial_to_polar")
        for name in paths:
            # Directional couplings vanish at r = 0 and omit the center radial basis function.
            basis_count = len(radii) - (name not in self._ISOTROPIC_PATHS)
            self.radial_weights[name] = nn.Parameter(
                torch.empty(out_channels, in_channels, basis_count).uniform_(-bound, bound)
            )
        self.scalar_bias = nn.Parameter(torch.zeros(out_channels))

    def build_kernel(self, *, input_axial: bool, output_axial: bool) -> Tensor:
        """Assemble the 3D convolution kernel tensor from radial weights and angular factors."""

        def term(name: str, geometry: Tensor | float = 1.0) -> Tensor:
            weights = self.radial_weights[name]
            if name not in self._ISOTROPIC_PATHS:
                weights = F.pad(weights, (1, 0))
            radial = (weights @ self.radial_basis).index_select(-1, self.radius_index) * self.normalization
            return radial[..., None, None, :] * geometry

        scalar = term("scalar")
        scalar_row = [scalar, term("divergence", self.direction[None, :, :])]
        polar_row = [
            term("gradient", self.direction[:, None, :]),
            term("polar", self.identity) + term("polar_projection", self.quadrupole),
        ]
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
            out_components * out_channels,
            in_components * in_channels,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
        )

    def forward(
        self,
        scalar: Tensor,
        vector: Tensor,
        axial: Tensor | None = None,
        *,
        return_axial: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Apply periodic O(3)-equivariant convolution followed by scalar-gated nonlinearity."""
        output_axial = return_axial or axial is not None
        parts = [scalar.unsqueeze(-2), vector]
        if axial is not None:
            if not self.input_axial:
                raise ValueError("This convolution layer does not accept an axial input.")
            parts.append(axial)
        fields = torch.cat(parts, dim=-2).flatten(-2).movedim(-1, 1)

        if self.padding:
            for dim, size in enumerate(scalar.shape[1:4], start=2):
                indices = torch.arange(-self.padding, size + self.padding, device=fields.device).remainder(size)
                fields = fields.index_select(dim, indices)

        kernel = self.build_kernel(input_axial=axial is not None, output_axial=output_axial)
        output = F.conv3d(fields, kernel)
        fields = output.movedim(1, -1).unflatten(-1, (-1, self.scalar_bias.numel()))
        scalar_pre = fields[..., 0, :] + self.scalar_bias
        gate = scalar_pre.sigmoid()
        vectors = fields[..., 1:, :] * gate.unsqueeze(-2)
        if output_axial:
            return scalar_pre * gate, vectors[..., :3, :], vectors[..., 3:, :]
        return scalar_pre * gate, vectors


class ActorCritic(nn.Module):
    """Translation- and O(3)-invariant Actor-Critic with masked discrete action head."""

    def __init__(
        self,
        num_actions: int,
        channels: int = 8,
        *,
        kernel_size: int = 3,
        conv_layers: int = 2,
        residual: bool = False,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if type(conv_layers) is not int or conv_layers < 1:
            raise ValueError("conv_layers must be a positive integer.")
        if not isinstance(residual, bool):
            raise ValueError(f"residual must be a bool, got {residual!r}.")
        self.residual = residual
        self.convolutions = nn.ModuleList(
            EquivariantConv3d(
                1 if index == 0 else channels,
                channels,
                kernel_size=kernel_size,
                input_axial=index > 0,
            )
            for index in range(conv_layers)
        )
        self.shared = nn.Sequential(nn.Linear(3 * channels + 2, hidden_dim), nn.Tanh())
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    @classmethod
    def from_config(cls, num_actions: int, config: PolicyConfig) -> ActorCritic:
        """Instantiate an ActorCritic from a validated PolicyConfig."""
        return cls(
            num_actions=num_actions,
            channels=config.channels,
            kernel_size=config.kernel_size,
            conv_layers=config.conv_layers,
            residual=config.residual,
            hidden_dim=config.hidden_dim,
        )

    def encode(self, fields: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode 4-channel (delta, momentum) grid into scalar, polar, and axial feature fields."""
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

    def forward(
        self,
        fields: Tensor,
        scalars: Tensor,
        action_mask: Tensor,
    ) -> tuple[Categorical, Tensor]:
        """Compute masked action distribution and state-value estimate for a batch of observations."""
        scalar, polar, axial = self.encode(fields)
        mean_scalar = scalar.mean(dim=(1, 2, 3))
        mean_polar_power = polar.square().sum(dim=-2).mean(dim=(1, 2, 3))
        mean_axial_power = axial.square().sum(dim=-2).mean(dim=(1, 2, 3))
        invariant = torch.cat((mean_scalar, mean_polar_power, scalars, mean_axial_power), dim=-1)
        features = self.shared(invariant)
        logits = self.actor(features).masked_fill(~action_mask, -torch.inf)
        return Categorical(logits=logits), self.critic(features).squeeze(-1)

    def act(self, observation: Observation) -> tuple[Categorical, Tensor]:
        """Evaluate the policy and critic on a single unbatched Observation."""
        return self(observation.fields[None], observation.scalars[None], observation.action_mask[None])
