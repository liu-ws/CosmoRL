"""CosmoRL: Particle-Mesh cosmology and O(3)-equivariant reinforcement learning."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .config import (
    CosmologyConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    PolicyConfig,
    PPOConfig,
    RuntimeConfig,
    SimulationConfig,
    TrainingConfig,
    load_config,
    write_default_config,
)

__version__ = "0.2.0"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Background": (".physics.cosmology", "Background"),
    "InitialConditions": (".physics.cosmology", "InitialConditions"),
    "CICStencil": (".physics.pm", "CICStencil"),
    "Integrator": (".physics.pm", "Integrator"),
    "ParticleState": (".physics.pm", "ParticleState"),
    "PeriodicMesh": (".physics.pm", "PeriodicMesh"),
    "fixed_targets": (".physics.pm", "fixed_targets"),
    "spectral_geometry": (".physics.pm", "spectral_geometry"),
    "FieldComparison": (".rl.environment", "FieldComparison"),
    "FieldErrors": (".rl.environment", "FieldErrors"),
    "Observation": (".rl.environment", "Observation"),
    "TimeStepEnvironment": (".rl.environment", "TimeStepEnvironment"),
    "Transition": (".rl.environment", "Transition"),
    "ActorCritic": (".rl.network", "ActorCritic"),
    "EquivariantConv3d": (".rl.network", "EquivariantConv3d"),
    "Rollout": (".rl.ppo", "Rollout"),
    "collect_episode": (".rl.ppo", "collect_episode"),
    "compute_gae": (".rl.ppo", "compute_gae"),
    "evaluate_episode": (".rl.ppo", "evaluate_episode"),
    "update_policy": (".rl.ppo", "update_policy"),
    "EvaluationEpisodeResult": (".workflows", "EvaluationEpisodeResult"),
    "EvaluationResult": (".workflows", "EvaluationResult"),
    "SimulationResult": (".workflows", "SimulationResult"),
    "TrainingResult": (".workflows", "TrainingResult"),
    "UpdateMetrics": (".workflows", "UpdateMetrics"),
    "build_simulation": (".workflows", "build_simulation"),
    "load_checkpoint": (".workflows", "load_checkpoint"),
    "run_evaluation": (".workflows", "run_evaluation"),
    "run_simulation": (".workflows", "run_simulation"),
    "run_training": (".workflows", "run_training"),
    "save_checkpoint": (".workflows", "save_checkpoint"),
}

__all__ = [
    "__version__",
    "CosmologyConfig",
    "EnvironmentConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "PPOConfig",
    "PolicyConfig",
    "RuntimeConfig",
    "SimulationConfig",
    "TrainingConfig",
    "load_config",
    "write_default_config",
    *_LAZY_EXPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
