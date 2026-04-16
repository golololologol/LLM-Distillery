import pytest
from pydantic import ValidationError
from classes.args import PipelineConfig, TeacherConfig, StudentConfig


def _base_config(**overrides):
    base = {
        "cache_folder": "test_cache",
        "dataset_path": "test_data/test_50.jsonl",
        "validation_dataset_path": "test_data/ultrachat_20_val.jsonl",
        "teacher_configs_path": "teacher_configs",
        "student_config_path": "student_configs/tinyllama_1.1b.toml",
        "num_epochs": 1,
        "batch_size": 2,
        "lr": 1e-4,
        "optimizer": "adamw",
    }
    base.update(overrides)
    return base


def test_valid_config_parses():
    cfg = PipelineConfig(**_base_config())
    assert cfg.batch_size == 2


def test_missing_required_field():
    d = _base_config()
    del d["lr"]
    with pytest.raises(ValidationError):
        PipelineConfig(**d)


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        PipelineConfig(**_base_config(bogus_field="abc"))


def test_negative_context_len():
    with pytest.raises(ValidationError):
        StudentConfig(model_path="x", freeze_layers=[], save_final_training_state=False, context_len=-1)


def test_zero_temperature():
    with pytest.raises(ValidationError):
        TeacherConfig(model_path="some/model", temperature=0)


def test_invalid_loss_type():
    with pytest.raises(ValidationError):
        PipelineConfig(**_base_config(loss_type="mse"))


def test_invalid_optimizer():
    with pytest.raises(ValidationError):
        PipelineConfig(**_base_config(optimizer="invalid"))


def test_adam_betas_list_to_tuple():
    cfg = PipelineConfig(**_base_config(adam_betas=[0.9, 0.999]))
    assert isinstance(cfg.adam_betas, tuple)
    assert cfg.adam_betas == (0.9, 0.999)


def test_incompatible_compile_liger():
    with pytest.raises(ValueError, match="torch_compile and liger_kernel"):
        PipelineConfig(**_base_config(torch_compile=True, liger_kernel=True))


def test_custom_device_map_needs_gpu0_layers():
    with pytest.raises(ValueError, match="num_gpu0_layers"):
        PipelineConfig(**_base_config(device_map="custom"))


def test_teacher_config_valid():
    tc = TeacherConfig(model_path="some/model", backend_type="vllm", context_len=2048)
    assert tc.model_path == "some/model"
    assert tc.backend_type == "vllm"


def test_teacher_config_missing_model_path():
    with pytest.raises(ValidationError):
        TeacherConfig(backend_type="vllm")


def test_teacher_collectable_requires_context_len():
    with pytest.raises(ValidationError):
        TeacherConfig(model_path="some/model", backend_type="vllm")


def test_student_config_valid():
    sc = StudentConfig(
        model_path="some/student",
        freeze_layers=["embed"],
        save_final_training_state=False,
        context_len=2048,
    )
    assert sc.model_path == "some/student"
    assert sc.freeze_layers == ["embed"]
