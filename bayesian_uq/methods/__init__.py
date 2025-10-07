from importlib import import_module
from typing import Dict, Type

from bayesian_uq.methods.base import BaseUQMethod, UQMethodConfig, UQPrediction

METHOD_REGISTRY: Dict[str, Type[BaseUQMethod]] = {}
DISCOVERY_MODULES = ("mc_dropout", "deep_ensemble", "bayesian_vi", "vanilla_softmax")
DISCOVERED = False


def register_method(name: str):
    def decorator(cls: Type[BaseUQMethod]):
        if name in METHOD_REGISTRY:
            raise ValueError(f"Method '{name}' is already registered")
        if not issubclass(cls, BaseUQMethod):
            raise TypeError(f"Class {cls.__name__} must inherit from BaseUQMethod")
        METHOD_REGISTRY[name] = cls
        return cls

    return decorator


def _ensure_discovered() -> None:
    global DISCOVERED
    if DISCOVERED:
        return
    for module_name in DISCOVERY_MODULES:
        import_module(f"{__name__}.{module_name}")
    DISCOVERED = True


def get_method(name: str, config=None):
    _ensure_discovered()
    if name not in METHOD_REGISTRY:
        available = list_methods()
        raise ValueError(
            f"Method '{name}' not found in registry. Available methods={available}"
        )
    method_class = METHOD_REGISTRY[name]
    if config is None:
        if hasattr(method_class, "default_config"):
            config = method_class.default_config()
        else:
            config = UQMethodConfig(name=name)
    elif isinstance(config, dict):
        if hasattr(method_class, "config_class"):
            config = method_class.config_class(**config)
        else:
            config = UQMethodConfig(**config)
    return method_class(config)


def list_methods():
    _ensure_discovered()
    return sorted(METHOD_REGISTRY.keys())


def get_method_class(name: str):
    _ensure_discovered()
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Method '{name}' not found in registry")
    return METHOD_REGISTRY[name]


def method_info(name: str):
    _ensure_discovered()
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Method '{name}' not found in registry")
    cls = METHOD_REGISTRY[name]
    return {
        "name": name,
        "class": cls.__name__,
        "module": cls.__module__,
        "docstring": cls.__doc__,
        "has_default_config": hasattr(cls, "default_config"),
        "config_class": cls.config_class.__name__
        if hasattr(cls, "config_class")
        else "UQMethodConfig",
    }


__all__ = [
    "BaseUQMethod",
    "UQMethodConfig",
    "UQPrediction",
    "register_method",
    "get_method",
    "list_methods",
    "get_method_class",
    "method_info",
]
