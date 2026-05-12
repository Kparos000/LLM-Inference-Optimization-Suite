import types

import pytest

from inference_bench import env
from inference_bench.env import load_local_env


def test_load_local_env_does_not_raise_when_python_dotenv_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_import_error(module_name: str) -> types.ModuleType:
        if module_name == "dotenv":
            raise ImportError
        return types.ModuleType(module_name)

    monkeypatch.setattr(env, "import_module", raise_import_error)

    load_local_env()


def test_load_local_env_is_importable_and_safe() -> None:
    assert callable(load_local_env)
    load_local_env()
