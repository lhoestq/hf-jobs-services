"""Parsers for the CLI's repeatable options (env, secrets, labels, volumes)."""

import os
import re
import sys

from huggingface_hub import Volume
from huggingface_hub.utils import parse_hf_mount

from .errors import ServicesError


def parse_env_map(values: list[str], file: str | None = None, *, token: str | None = None) -> dict[str, str]:
    """Parse `--env`/`--secrets` values (and an optional dotenv file) into a dict.

    A `KEY=VALUE` item sets the value inline. A bare `KEY` pulls the value from the local environment,
    which is how `--secrets HF_TOKEN` forwards the token the CLI is already authenticated with.
    """
    env_map: dict[str, str] = {}
    if file:
        env_map.update(parse_dotenv(read_text(file)))
    for value in values:
        env_map.update(parse_dotenv(value, environ=_extended_environ(token)))
    return env_map


def parse_dotenv(value: str, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Parse one or more `KEY=VALUE` lines (or a bare `KEY`) into a dict."""
    environ = environ if environ is not None else dict(os.environ)
    parsed: dict[str, str] = {}
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, raw_value = _EXPORT_RE.sub("", line).partition("=")
        key = key.strip()
        if not _KEY_RE.fullmatch(key):
            raise ServicesError(f"Invalid environment variable name in '{line}'. Expected KEY=VALUE.")
        if not separator:
            if key not in environ:
                parsed[key] = ""
            else:
                parsed[key] = environ[key]
            continue
        parsed[key] = raw_value.strip().strip("\"'")
    return parsed


def parse_labels(values: list[str], name: str | None = None) -> dict[str, str] | None:
    """Parse `--label key=value` items, merging `--name` in as the `name` label."""
    labels: dict[str, str] = {}
    for value in values:
        key, separator, label_value = value.partition("=")
        if not separator or not key.strip():
            raise ServicesError(f"Invalid label '{value}'. Expected key=value.")
        labels[key.strip()] = label_value.strip()
    if name is not None:
        labels["name"] = name
    return labels or None


def parse_volumes(specs: list[str]) -> list[Volume]:
    """Parse `hf://[TYPE/]SOURCE[/PATH]:/MOUNT_PATH[:ro|:rw]` volume specs.

    Same syntax as `hf jobs uv run -v`: type defaults to `models`, buckets are read-write by default,
    repos are always read-only.
    """
    volumes: list[Volume] = []
    for spec in specs:
        try:
            mount = parse_hf_mount(spec)
        except Exception as error:
            raise ServicesError(f"Invalid volume '{spec}': {error}") from error
        volumes.append(
            Volume(
                type=mount.source.type,
                source=mount.source.id,
                mount_path=mount.mount_path,
                read_only=mount.read_only,
                path=mount.source.path_in_repo or None,
                revision=mount.source.revision or None,
            )
        )
    return volumes


def read_text(path: str) -> str:
    """Read a file, or standard input when the path is `-` (as in `hf jobs uv run --env-file -`)."""
    if path == "-":
        return sys.stdin.read()
    try:
        with open(path) as f:
            return f.read()
    except OSError as error:
        raise ServicesError(f"Could not read '{path}': {error.strerror}") from error


def _extended_environ(token: str | None) -> dict[str, str]:
    """Local environment with the resolved HF token injected, so bare `HF_TOKEN` resolves."""
    environ = dict(os.environ)
    if token and "HF_TOKEN" not in environ:
        environ["HF_TOKEN"] = token
    return environ


_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_EXPORT_RE = re.compile(r"^export\s+")
