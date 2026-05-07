import utils.merging_utils as _mu
from typing import Any
if not hasattr(_mu, "resolve_train_target"):
    _mu.resolve_train_target = lambda *a, **kw: None

from collect_and_finetune import _collection_params_hash


_BASE: dict[str, Any] = dict(
    context_len=2048,
    temperature=1.0,
    save_roles=["assistant"],
)


def test_identical_inputs_same_hash():
    assert _collection_params_hash(**_BASE) == _collection_params_hash(**_BASE)


def test_context_len_changes_hash():
    a = _collection_params_hash(**_BASE)
    b = _collection_params_hash(**{**_BASE, "context_len": 4096})
    assert a != b


def test_temperature_changes_hash():
    a = _collection_params_hash(**_BASE)
    b = _collection_params_hash(**{**_BASE, "temperature": 0.5})
    assert a != b


def test_save_roles_changes_hash():
    a = _collection_params_hash(**_BASE)
    b = _collection_params_hash(**{**_BASE, "save_roles": ["user", "assistant"]})
    assert a != b


def test_save_roles_order_invariant():
    a = _collection_params_hash(**{**_BASE, "save_roles": ["assistant", "user"]})
    b = _collection_params_hash(**{**_BASE, "save_roles": ["user", "assistant"]})
    assert a == b


def test_render_options_changes_hash():
    a = _collection_params_hash(**_BASE, render_options={"enable_thinking": True})
    b = _collection_params_hash(**_BASE, render_options={"enable_thinking": False})
    assert a != b


def test_render_options_none_vs_empty_dict_same():
    a = _collection_params_hash(**_BASE, render_options=None)
    b = _collection_params_hash(**_BASE, render_options={})
    assert a == b


def test_chat_template_changes_hash():
    a = _collection_params_hash(**_BASE, chat_template=None)
    b = _collection_params_hash(**_BASE, chat_template="{% for m in messages %}X{% endfor %}")
    assert a != b


def test_chat_template_different_overrides_different_hash():
    a = _collection_params_hash(**_BASE, chat_template="{% if 1 %}A{% endif %}")
    b = _collection_params_hash(**_BASE, chat_template="{% if 1 %}B{% endif %}")
    assert a != b
