"""Built-in adapters, loaded lazily with their optional dependencies."""

__all__ = ["SafetyBenchCheck", "TransformersAdapter"]


def __getattr__(name: str):
    if name == "SafetyBenchCheck":
        from .safetybench import SafetyBenchCheck

        return SafetyBenchCheck
    if name == "TransformersAdapter":
        from .transformers import TransformersAdapter

        return TransformersAdapter
    raise AttributeError(name)
