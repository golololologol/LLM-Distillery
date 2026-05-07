import importlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Callable

from classes.data_classes import ConvoProcessed

log = logging.getLogger(__name__)

_registry: dict[str, type["InferenceBackend"]] = {}
_discovered = False


def _discover_backends():
    global _discovered
    if _discovered:
        return
    _discovered = True
    inference_dir = Path(__file__).parent
    for child in sorted(inference_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if not (child / "backend.py").exists():
            continue
        module_name = f"classes.inference.{child.name}.backend"
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            log.debug(f"Backend '{child.name}' skipped (missing deps): {e}")


def register_backend(name: str, cls: type["InferenceBackend"]):
    _registry[name] = cls


def get_backend(name: str) -> type["InferenceBackend"]:
    _discover_backends()
    if name in _registry:
        return _registry[name]
    available = sorted(_registry.keys())
    raise ValueError(f"Unknown backend: {name!r}. Available: {available}")


def get_available_backends() -> list[str]:
    _discover_backends()
    return sorted(_registry.keys())


class InferenceBackend(ABC):
    @abstractmethod
    def start(self, distribution_writer) -> None:
        """Load model onto GPU, prepare for inference."""

    @abstractmethod
    def process_chunk(
        self,
        samples: list[ConvoProcessed],
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Process all samples and write results to data_manager. Blocks until done."""

    @abstractmethod
    def stop(self) -> None:
        """Unload model, free GPU memory, terminate worker processes."""
