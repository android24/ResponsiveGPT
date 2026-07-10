import json
import os
from pathlib import Path
from typing import Mapping


SECRET_KEYS = (
    "API_KEY",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "PASSWORD",
    "SECRET",
    "PRIVATE_KEY",
)


def _parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]
            env[key.strip()] = value
    return env


def _load_account_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Account config must be a JSON object: {path}")

    env: dict[str, str] = {}
    local = payload.get("local", {})
    if isinstance(local, dict):
        env.update({str(k): str(v) for k, v in local.items() if v is not None})

    accounts = payload.get("accounts", {})
    active = str(payload.get("active", "") or "")
    if isinstance(accounts, dict) and active:
        account = accounts.get(active, {})
        if not isinstance(account, dict):
            raise ValueError(f"Active account is not an object: {active}")
        env.update({str(k): str(v) for k, v in account.items() if v is not None})
    return env


def load_private_env(
    env_path: str | os.PathLike[str] = ".env",
    account_path: str | os.PathLike[str] = "config/accounts.local.json",
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """
    Load local-only credentials without requiring them to be committed.

    Precedence, from low to high:
    1. .env
    2. config/accounts.local.json
    3. current process environment
    """
    merged = _parse_env_file(Path(env_path))
    merged.update(_load_account_file(Path(account_path)))
    source_env = os.environ if environ is None else environ
    for key, value in source_env.items():
        if value is not None:
            merged[str(key)] = str(value)
    return merged


def redact_env(env: Mapping[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in env.items():
        if any(token in key.upper() for token in SECRET_KEYS):
            redacted[str(key)] = "***REDACTED***" if value else ""
        else:
            redacted[str(key)] = str(value)
    return redacted
