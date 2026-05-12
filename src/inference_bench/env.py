"""Optional local environment loading."""

from __future__ import annotations

from importlib import import_module


def load_local_env() -> None:
    """Load local environment variables from .env when python-dotenv is available."""

    try:
        dotenv_module = import_module("dotenv")
    except ImportError:
        return

    load_dotenv = getattr(dotenv_module, "load_dotenv", None)
    if callable(load_dotenv):
        load_dotenv()
