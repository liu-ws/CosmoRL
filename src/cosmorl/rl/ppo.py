"""Generalized Advantage Estimation (GAE) and masked PPO rollout/update routines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import PPOConfig
from .environment import Observation, TimeStepEnvironment, Transition
from .network import ActorCritic


def compute_gae(values: Tensor, rewards: Tensor, gae_lambda: float = 0.95) -> Tensor:
    """Compute finite-horizon GAE with gamma=1 and zero terminal bootstrap."""
    next_value = values.new_zeros(())
    next_advantage = values.new_zeros(())
    advantages: list[Tensor] = []
    for value, reward in zip(reversed(values), reversed(rewards), strict=True):
        next_advantage = reward + next_value - value + gae_lambda * next_advantage
        advantages.append(next_advantage)
        next_value = value
    return torch.stack(advantages[::-1])


@dataclass
class Rollout:
    """Complete episode trajectory collected under a frozen behavior policy."""

    observations: list[Observation]
    actions: Tensor
    log_probabilities: Tensor
    advantages: Tensor
    returns: Tensor
    reward: float


@torch.no_grad()
def collect_episode(
    environment: TimeStepEnvironment,
    policy: ActorCritic,
    seed: int,
    *,
    gae_lambda: float = 0.95,
) -> Rollout:
    """Roll out one full episode and compute GAE advantages and value targets."""
    observations: list[Observation] = []
    actions: list[Tensor] = []
    log_probabilities: list[Tensor] = []
    values: list[Tensor] = []
    rewards: list[float] = []

    observation = environment.reset(seed)
    while observation is not None:
        distribution, value = policy.act(observation)
        action = distribution.sample()
        transition = environment.step(int(action.item()))
        observations.append(observation)
        actions.append(action)
        log_probabilities.append(distribution.log_prob(action))
        values.append(value)
        rewards.append(transition.reward)
        observation = transition.observation

    value_tensor = torch.cat(values)
    advantages = compute_gae(value_tensor, value_tensor.new_tensor(rewards), gae_lambda=gae_lambda)
    return Rollout(
        observations=observations,
        actions=torch.cat(actions),
        log_probabilities=torch.cat(log_probabilities),
        advantages=advantages,
        returns=advantages + value_tensor,
        reward=sum(rewards),
    )


def update_policy(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollouts: list[Rollout],
    config: PPOConfig | None = None,
) -> float:
    """Run clipped PPO minibatch updates across collected episode rollouts."""
    ppo = config or PPOConfig()
    observations = [obs for rollout in rollouts for obs in rollout.observations]
    actions = torch.cat([rollout.actions for rollout in rollouts])
    old_log_prob = torch.cat([rollout.log_probabilities for rollout in rollouts])
    advantages = torch.cat([rollout.advantages for rollout in rollouts])
    returns = torch.cat([rollout.returns for rollout in rollouts])
    advantages = (advantages - advantages.mean()) / advantages.std(correction=0).clamp_min(1e-8)

    clip_low = 1.0 - ppo.clip_ratio
    clip_high = 1.0 + ppo.clip_ratio
    losses: list[Tensor] = []

    for _ in range(ppo.epochs):
        for indices in torch.randperm(len(observations)).split(ppo.minibatch_size):
            fields, scalars, masks = [], [], []
            for index in indices.tolist():
                saved = observations[index]
                fields.append(saved.fields)
                scalars.append(saved.scalars)
                masks.append(saved.action_mask)

            distribution, values = policy(
                torch.stack(fields),
                torch.stack(scalars),
                torch.stack(masks),
            )
            ratios = (distribution.log_prob(actions[indices]) - old_log_prob[indices]).exp()
            batch_adv = advantages[indices]
            actor_loss = -torch.minimum(
                ratios * batch_adv,
                ratios.clamp(clip_low, clip_high) * batch_adv,
            ).mean()
            critic_loss = (values - returns[indices]).square().mean()
            loss = actor_loss + ppo.value_coef * critic_loss - ppo.entropy_coef * distribution.entropy().mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), ppo.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())

    return torch.stack(losses).mean().item()


@torch.no_grad()
def evaluate_episode(
    environment: TimeStepEnvironment,
    policy: ActorCritic,
    seed: int,
) -> Transition:
    """Evaluate a trained policy greedily (argmax over legal logits) on a given seed."""
    observation = environment.reset(seed)
    while True:
        distribution, _ = policy.act(observation)
        action = int(distribution.logits.argmax(dim=-1).item())
        transition = environment.step(action)
        if transition.observation is None:
            return transition
        observation = transition.observation
