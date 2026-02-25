# src/athena_regime/config/utils.py
from __future__ import annotations

import os
import re
from typing import Any


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def expand_env(obj: Any, *, strict: bool = True) -> Any:
    """
    Recursively expand environment variables in a nested structure.

    Replaces occurrences of ${VAR_NAME} inside string values with os.environ["VAR_NAME"].

    Args:
        obj: Any nested structure composed of dict/list/str/primitives.
        strict: If True, raise KeyError when a referenced env var is missing.
                If False, leave the placeholder unchanged.

    Returns:
        A new structure with substitutions applied (non-mutating).
    """
    if isinstance(obj, dict):
        return {k: expand_env(v, strict=strict) for k, v in obj.items()}

    if isinstance(obj, list):
        return [expand_env(v, strict=strict) for v in obj]

    if isinstance(obj, str):
        def _repl(match: re.Match[str]) -> str:
            var = match.group(1)
            if var in os.environ:
                return os.environ[var]
            if strict:
                raise KeyError(f"Environment variable '{var}' is not set")
            return match.group(0)  # keep ${VAR} as-is

        return _ENV_PATTERN.sub(_repl, obj)

    return obj


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base.

    - If both values are dicts: merge recursively.
    - Otherwise: override replaces base.

    Args:
        base: Base dictionary.
        override: Override dictionary.

    Returns:
        The merged dictionary (same object as base, mutated in place).
    """
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base