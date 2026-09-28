"""Turn the `--with-services` value into the list of service Jobs to start.

The value is either a template call (`dask(num_workers=4)`) resolved through
[`SERVICES_TEMPLATES`](./templates.md), or the path of a services file. When it is not passed, the
file is looked up next to the script as `<script-name>-services.yml`, then as `jobs-services.yml`.
A services file lists the long-running Jobs that accompany the main one:

```yaml
services:
  server:
    image: python:3.12
    command: ["python", "-m", "http.server", "8000"]
    replicas: 2
```

Every service becomes one HF Job per replica, all of them joining the same network group and claiming
their name (plus a replica index) as a network alias.
"""

import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import Volume

from .errors import ServicesError
from .parsing import parse_volumes, read_text
from .templates import SERVICES_TEMPLATES

# Keys coming from a container-composition tool, with no meaning for HF Jobs (services start in
# parallel, and reach each other on every port through the network group). Rejected with a targeted
# message instead of the generic "unknown key" one, since reusing an existing file is common.
_UNSUPPORTED_KEYS: dict[str, str] = {
    "depends_on": (
        "services start in parallel: connect with retries instead, members are resolvable before they are ready."
    ),
    "networks": "all services already share one auto-generated network group.",
    "ports": "services reach each other on every port inside the network group.",
    "build": "only pre-built images can be used.",
    "restart": "a failing service makes the whole run fail.",
}

_SUPPORTED_KEYS = {"image", "command", "env", "secrets", "flavor", "replicas", "expose", "volumes", "ssh", "labels"}

# Fallback name of a services file, when it is not named after the script it serves.
DEFAULT_SERVICES_FILENAMES = ("jobs-services.yml", "jobs-services.yaml")


@dataclass
class ServiceSpec:
    """One service Job: what to run, and the network alias it claims in the network group."""

    service: str
    alias: str
    image: str
    command: list[str]
    env: dict[str, str] | None = None
    secrets: dict[str, str] | None = None
    flavor: str | None = None
    expose: list[int] | None = None
    volumes: list[Volume] | None = None
    ssh: bool = False
    labels: dict[str, str] = field(default_factory=dict)


def resolve_services(value: str) -> dict[str, Any]:
    """Resolve a `--with-services` value (template call or services file path) to a services dict."""
    template = _parse_template_call(value)
    if template is not None:
        return template
    if not Path(value).is_file():
        raise ServicesError(f"Services file not found: {value}")
    return _parse_services_file(value)


def services_candidates(script: str) -> list[str]:
    """Services files looked up for `script`: `<script-name>-services.yml` then `jobs-services.yml`."""
    candidates = []
    if not script.startswith(("http://", "https://")):
        stem = Path(script).stem or "jobs"
        candidates += [str(Path(script).parent / f"{stem}-services{suffix}") for suffix in (".yml", ".yaml")]
    return [*candidates, *DEFAULT_SERVICES_FILENAMES]


def discover_services_file(script: str) -> str | None:
    """Find the services file of `script` without an explicit `--with-services`."""
    return next((candidate for candidate in services_candidates(script) if Path(candidate).is_file()), None)


def build_service_specs(services: dict[str, Any]) -> list[ServiceSpec]:
    """Validate a services dict and expand `replicas` into one spec per replica."""
    if not isinstance(services, dict) or not services.get("services"):
        raise ServicesError("Services configuration must have a non-empty 'services' section")
    specs: list[ServiceSpec] = []
    for name, config in services["services"].items():
        specs.extend(_build_service_specs(str(name), config))
    return specs


def _build_service_specs(name: str, config: Any) -> list[ServiceSpec]:
    if not isinstance(config, dict):
        raise ServicesError(f"Service '{name}' must be a mapping of options, got {type(config).__name__}")

    unknown = set(config) - _SUPPORTED_KEYS
    if unknown:
        details = "; ".join(
            f"'{key}' is not supported ({reason})" for key, reason in _UNSUPPORTED_KEYS.items() if key in unknown
        )
        supported = ", ".join(sorted(_SUPPORTED_KEYS))
        raise ServicesError(
            f"Unknown key(s) for service '{name}': {', '.join(sorted(unknown))}."
            + (f" {details}" if details else "")
            + f" Supported keys: {supported}."
        )
    if "image" not in config or "command" not in config:
        raise ServicesError(f"Service '{name}' must have 'image' and 'command' specified")

    base = _sanitize(name, 34)
    replicas = _positive_int(config.get("replicas", 1), "replicas", name)
    aliases = [base] if replicas == 1 else [f"{base[:32]}-{index}" for index in range(replicas)]

    return [
        ServiceSpec(
            service=name,
            alias=alias,
            image=str(config["image"]),
            command=_command(config["command"]),
            env=_str_map(config.get("env")),
            secrets=_str_map(config.get("secrets")),
            flavor=config.get("flavor"),
            expose=_ports(config.get("expose"), name),
            volumes=parse_volumes([str(volume) for volume in config["volumes"]]) if config.get("volumes") else None,
            ssh=bool(config.get("ssh", False)),
            labels={str(key): str(value) for key, value in (config.get("labels") or {}).items()},
        )
        for alias in aliases
    ]


def _parse_template_call(value: str) -> dict[str, Any] | None:
    """Return the services dict of a `name(key=value, ...)` template call, or None if not a call.

    A bare name that is also an existing file is left to the file branch: `--with-services services.yml`
    should not fail because it happens to look like a template with no arguments.
    """
    match = _TEMPLATE_CALL_RE.fullmatch(value.strip())
    if match is None:
        return None
    name, raw_args = match.group(1), match.group(2) or ""
    if name not in SERVICES_TEMPLATES and Path(value).is_file():
        return None
    if name not in SERVICES_TEMPLATES:
        available = ", ".join(f"`{template}(param=value)`" for template in SERVICES_TEMPLATES)
        raise ServicesError(
            f"Unknown services template '{name}'. Available templates: {available}. "
            f"Or pass the path of a services file."
        )

    kwargs: dict[str, Any] = {}
    for arg in _KWARG_RE.finditer(raw_args):
        key, raw_value = arg.group(1), arg.group(2).strip()
        kwargs[key] = _coerce(raw_value)
    try:
        return SERVICES_TEMPLATES[name](**kwargs)
    except TypeError as error:
        raise ServicesError(
            f"Invalid arguments for template '{name}': {error}. Usage: {name}({template_params(name)})"
        ) from error


def _parse_services_file(path: str) -> dict[str, Any]:
    try:
        services = yaml.safe_load(read_text(path))
    except yaml.YAMLError as error:
        raise ServicesError(f"Invalid YAML in '{path}': {error}") from error
    if not isinstance(services, dict):
        raise ServicesError(f"'{path}' must contain a mapping with a 'services' section")
    return services


def template_params(name: str) -> str:
    """Human-readable parameter list of a template, for error messages and `hf jobs-services templates`."""
    parameters = []
    for parameter in inspect.signature(SERVICES_TEMPLATES[name]).parameters.values():
        text = parameter.name
        if parameter.default is not inspect.Parameter.empty:
            default = parameter.default
            text = (
                f"{text}={default}"
                if isinstance(default, (int, float, bool)) or default is None
                else f'{text}="{default}"'
            )
        parameters.append(text)
    return ", ".join(parameters)


def _coerce(raw_value: str) -> Any:
    """Coerce a template argument: quoted strings stay strings, then bool, int, float, string."""
    if len(raw_value) > 1 and raw_value[0] == raw_value[-1] and raw_value[0] in "\"'":
        return raw_value[1:-1]
    match raw_value.lower():
        case "true":
            return True
        case "false":
            return False
        case _:
            pass
    for cast in (int, float):
        try:
            return cast(raw_value)
        except ValueError:
            continue
    return raw_value


def _command(command: Any) -> list[str]:
    if isinstance(command, str):
        # A bare string is a shell line: `sh -c` it so `|`, `&&` and `${VAR}` work as in a shell.
        return ["sh", "-c", command]
    if isinstance(command, list) and command:
        return [str(part) for part in command]
    raise ServicesError("Service 'command' must be a non-empty list or a string")


def _ports(expose: Any, name: str) -> list[int] | None:
    if expose is None:
        return None
    if not isinstance(expose, list):
        expose = [expose]
    try:
        return [int(port) for port in expose]
    except (TypeError, ValueError) as error:
        raise ServicesError(f"Service '{name}' has an invalid 'expose' value: {expose}") from error


def _str_map(value: Any) -> dict[str, str] | None:
    """Normalize a YAML env/secrets mapping: YAML scalars can be ints or booleans, the API wants strings."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ServicesError(f"'{value}' must be a mapping of KEY: value")
    return {str(key): str(val).lower() if isinstance(val, bool) else str(val) for key, val in value.items()}


def _positive_int(value: Any, key: str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ServicesError(f"Service '{name}' has an invalid '{key}': {value}") from error
    if parsed < 1:
        raise ServicesError(f"Service '{name}' must have '{key}' >= 1, got {parsed}")
    return parsed


def _sanitize(name: str, max_length: int) -> str:
    """Make `name` usable as a network group alias: lowercase alphanumerics and dashes."""
    sanitized = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")
    if not sanitized:
        raise ServicesError(f"Service name '{name}' cannot be turned into a valid network alias")
    return sanitized[:max_length]


_TEMPLATE_CALL_RE = re.compile(r"([A-Za-z_]\w*)(?:\((.*)\))?", re.DOTALL)
_KWARG_RE = re.compile(r"(\w+)\s*=\s*(\"[^\"]*\"|'[^']*'|[^,]+)")
