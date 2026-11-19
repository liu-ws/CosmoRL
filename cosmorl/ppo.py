"""Periodic scalar/vector encoder and masked PPO for complete episodes."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .environment import Observation, TimeStepEnvironment, Transition


class VectorConvolution(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.scalar_mix = nn.Linear(2 * input_channels, output_channels)
        self.vector_mix = nn.Linear(4 * input_channels, output_channels, bias=False)

    def forward(self, scalar: Tensor, vector: Tensor) -> tuple[Tensor, Tensor]:
        scalar_neighbors = torch.zeros_like(scalar)
        vector_neighbors = torch.zeros_like(vector)
        longitudinal = []
        directed_density = []
        for axis in range(3):
            spatial_dim = axis + 1
            forward = scalar.roll(-1, dims=spatial_dim)
            backward = scalar.roll(1, dims=spatial_dim)
            vector_pair = vector.roll(-1, dims=spatial_dim) + vector.roll(1, dims=spatial_dim)
            scalar_neighbors = scalar_neighbors + forward + backward
            vector_neighbors = vector_neighbors + vector_pair
            longitudinal.append(vector_pair.select(-2, axis) / 2)
            directed_density.append((forward - backward) / 2)
        scalar = self.scalar_mix(torch.cat((scalar, scalar_neighbors / 6), dim=-1))
        vector = self.vector_mix(torch.cat((
            vector,
            vector_neighbors / 6,
            torch.stack(longitudinal, dim=-2),
            torch.stack(directed_density, dim=-2),
        ), dim=-1))
        gate = scalar.sigmoid()
        return scalar * gate, vector * gate.unsqueeze(-2)


class ActorCritic(nn.Module):
    def __init__(self, actions: int, channels: int = 8):
        super().__init__()
        self.first = VectorConvolution(1, channels)
        self.second = VectorConvolution(channels, channels)
        self.shared = nn.Sequential(nn.Linear(2 * channels + 2, 32), nn.Tanh())
        self.actor = nn.Linear(32, actions)
        self.critic = nn.Linear(32, 1)

    def forward(self, fields: Tensor, scalars: Tensor, allowed: Tensor) -> tuple[Categorical, Tensor]:
        scalar = fields[:, :1].movedim(1, -1)
        vector = fields[:, 1:].movedim(1, -1).unsqueeze(-1)
        scalar, vector = self.second(*self.first(scalar, vector))
        mean_scalar = scalar.mean(dim=(1, 2, 3))
        mean_vector_power = vector.square().sum(dim=-2).mean(dim=(1, 2, 3))
        features = self.shared(torch.cat((mean_scalar, mean_vector_power, scalars), dim=-1))
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
