"""
Model adapter registry.

Each adapter encapsulates how a specific model type builds its CLI command.
Register new adapters here so the orchestrator discovers them automatically.
"""
from experiment_orchestrator.adapters.base import ModelAdapter
from experiment_orchestrator.adapters.medgemma import MedGemmaAdapter
from experiment_orchestrator.adapters.med3dvlm import Med3DVLMAdapter
from experiment_orchestrator.adapters.qwen import QwenAdapter
from experiment_orchestrator.adapters.medfusion import MedFusionAdapter

ADAPTER_REGISTRY: dict[str, ModelAdapter] = {
    "medgemma": MedGemmaAdapter(),
    "med3dvlm": Med3DVLMAdapter(),
    "qwen": QwenAdapter(),
    "medfusion": MedFusionAdapter(),
}


def get_adapter(name: str) -> ModelAdapter:
    """Look up an adapter by name. Raises KeyError if not found."""
    if name not in ADAPTER_REGISTRY:
        raise KeyError(
            f"Unknown adapter '{name}'. Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[name]
