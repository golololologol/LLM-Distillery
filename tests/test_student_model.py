import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import classes.student.model as student_model_module
from classes.student.model import StudentModel, _patch_device_hooks


class FakeModel:
    def __init__(self):
        self.train_called = False
        self.grad_checkpointing_called = False
        self.device = None
        self._modules_list = [self]

    def train(self):
        self.train_called = True

    def to(self, device):
        self.device = device
        return self

    def gradient_checkpointing_enable(self):
        self.grad_checkpointing_called = True

    def named_parameters(self):
        return []

    def modules(self):
        return self._modules_list


def _make_loader(name):
    class Loader:
        calls = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            return FakeModel()

    Loader.__name__ = name
    return Loader


def _install_fake_liger(monkeypatch, liger_loader):
    liger_module = ModuleType("liger_kernel")
    liger_transformers = ModuleType("liger_kernel.transformers")
    setattr(liger_transformers, "AutoLigerKernelForCausalLM", liger_loader)
    setattr(liger_module, "transformers", liger_transformers)
    monkeypatch.setitem(sys.modules, "liger_kernel", liger_module)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", liger_transformers)


def _make_student(**config_overrides):
    config = SimpleNamespace(
        training_precision="bf16",
        training_strategy="naive_layer_split",
        liger_kernel=True,
        multi_gpu=True,
        device_map="auto",
        num_gpu0_layers=None,
        max_memory={},
        max_memory_hf={},
        grad_checkpointing=False,
    )
    for key, value in config_overrides.items():
        setattr(config, key, value)

    student_config = SimpleNamespace(
        model_path="fake/model",
        context_len=128,
        attn_implementation="sdpa",
        freeze_layers=[],
    )

    return StudentModel(cast(Any, config), cast(Any, student_config), cast(Any, SimpleNamespace()))


def test_load_model_uses_liger_with_device_hooks_for_naive_layer_split_multi_gpu(monkeypatch, capsys):
    standard_loader = _make_loader("StandardLoader")
    liger_loader = _make_loader("LigerLoader")
    monkeypatch.setattr(student_model_module, "AutoModelForCausalLM", standard_loader)
    _install_fake_liger(monkeypatch, liger_loader)

    patch_calls = []
    monkeypatch.setattr(student_model_module, "_patch_device_hooks", lambda model: patch_calls.append(model))

    student = _make_student(multi_gpu=True)
    student._load_model(rank=0)

    assert len(liger_loader.calls) == 1
    assert len(standard_loader.calls) == 0
    kwargs = liger_loader.calls[0][1]
    assert kwargs["device_map"] == "auto"
    assert kwargs["cross_entropy"] is False
    assert kwargs["fused_linear_cross_entropy"] is False
    assert student.model.train_called is True
    assert len(patch_calls) == 1

    out = capsys.readouterr().out
    assert "Using Liger Kernel" in out


def test_load_model_keeps_liger_for_single_gpu_naive_layer_split(monkeypatch, capsys):
    standard_loader = _make_loader("StandardLoader")
    liger_loader = _make_loader("LigerLoader")
    monkeypatch.setattr(student_model_module, "AutoModelForCausalLM", standard_loader)
    _install_fake_liger(monkeypatch, liger_loader)

    student = _make_student(multi_gpu=False)
    student._load_model(rank=0)

    assert len(liger_loader.calls) == 1
    assert len(standard_loader.calls) == 0
    kwargs = liger_loader.calls[0][1]
    assert kwargs["device_map"] == {"": "cuda:0"}
    assert kwargs["cross_entropy"] is False
    assert kwargs["fused_linear_cross_entropy"] is False
    assert student.model.train_called is True

    out = capsys.readouterr().out
    assert "Using Liger Kernel" in out


def test_patch_device_hooks_wraps_pre_forward():
    import torch

    hook = SimpleNamespace(execution_device=torch.device("cuda:1"))
    original_calls = []
    hook.pre_forward = lambda module, *args, **kwargs: original_calls.append((args, kwargs)) or (args, kwargs)

    child = SimpleNamespace(_hf_hook=hook)
    model = SimpleNamespace()
    model.modules = lambda: [model, child]

    _patch_device_hooks(model)

    # pre_forward should now be wrapped
    assert hook.pre_forward is not None
    assert len(original_calls) == 0  # not called yet


def test_load_model_keeps_liger_for_multi_gpu_ddp(monkeypatch, capsys):
    standard_loader = _make_loader("StandardLoader")
    liger_loader = _make_loader("LigerLoader")
    monkeypatch.setattr(student_model_module, "AutoModelForCausalLM", standard_loader)
    _install_fake_liger(monkeypatch, liger_loader)

    student = _make_student(training_strategy="ddp", multi_gpu=True)
    student._load_model(rank=0)

    assert len(liger_loader.calls) == 1
    assert len(standard_loader.calls) == 0
    kwargs = liger_loader.calls[0][1]
    assert "device_map" not in kwargs
    assert kwargs["cross_entropy"] is False
    assert kwargs["fused_linear_cross_entropy"] is False
    assert student.model.device == "cuda:0"
    assert student.model.train_called is True

    out = capsys.readouterr().out
    assert "Using Liger Kernel" in out
    assert "Disabling Liger Kernel" not in out