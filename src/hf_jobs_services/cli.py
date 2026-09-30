"""`hf jobs-services`: run an HF Job alongside long-running services in one network group.

Installed as an `hf` CLI extension, the binary is named `hf-jobs-services` and `hf jobs-services ...` forwards
to it, so this module's `main()` only ever sees the arguments after `jobs-services`.
"""

import contextlib
import functools
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import click
from huggingface_hub import HfApi, JobInfo, JobStage, get_token
from huggingface_hub.errors import HfHubHTTPError

from .errors import ServicesError
from .output import hint, log, result, table, warn
from .parsing import parse_env_map, parse_labels, parse_volumes
from .runner import (
    GROUP_LABEL,
    MAIN_SERVICE,
    SERVICE_LABEL,
    ServiceJob,
    cancel_group_jobs,
    cancel_services,
    hint_connect,
    list_group_jobs,
    new_group_name,
    stage_name,
    start_services,
    stream_logs,
    wait_for_services,
)
from .services import (
    DEFAULT_SERVICES_FILENAMES,
    ServiceSpec,
    build_service_specs,
    discover_services_file,
    find_default_services_file,
    resolve_services,
    services_candidates,
    template_params,
)
from .templates import SERVICES_TEMPLATES

# Fallback for the service Jobs when the user did not ask for a longer run: services must not stay up
# forever if the terminal running `hf jobs-services` dies (a Ctrl+C'd laptop, a dropped SSH session...).
DEFAULT_SERVICES_TIMEOUT = "2h"

# Trailing arguments belong to the script: `hf jobs-services uv run -v ... s.py --epochs 3`. Known options are
# still parsed anywhere (same as `hf jobs uv run`), so pass script flags after a `--` separator to be safe.
RUN_CONTEXT = {"ignore_unknown_options": True}


class AliasedGroup(click.Group):
    """Accept pipe-separated command names (`@cli.command("ls | list")`), like the built-in `hf` CLI."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.displays: dict[str, str] = {}

    def add_command(self, cmd: click.Command, name: str | None = None) -> None:
        names = [part.strip() for part in (name or cmd.name or "").split("|") if part.strip()]
        cmd.name = names[0]
        for alias in names:
            self.commands[alias] = cmd
            self.displays[alias] = " | ".join(names)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Show an aliased command once (`ls | list`) instead of one line per alias."""
        rows = []
        for name in super().list_commands(ctx):
            display = self.displays.get(name, name)
            if display not in {row[0] for row in rows}:
                rows.append((display, self.commands[name].get_short_help_str(limit=70)))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


def handle_errors(func: Callable[..., None]) -> Callable[..., None]:
    """Render API and file failures as a one-line error instead of a traceback."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            return func(*args, **kwargs)
        except HfHubHTTPError as error:
            raise ServicesError(_api_message(error)) from None
        except FileNotFoundError as error:
            raise ServicesError(f"File not found: {error}") from error

    return wrapper


@click.group(
    cls=AliasedGroup, name="jobs-services", help="Run Jobs alongside background services (Ray, Dask, Spark, custom)."
)
def cli() -> None:
    pass


@click.group(cls=AliasedGroup, name="uv", help="Run UV scripts as Jobs, alongside services.")
def uv() -> None:
    pass


cli.add_command(uv)

# The options of `hf jobs run`, shared verbatim by `hf jobs-services run` and `hf jobs-services uv run`.
COMMON_OPTIONS: list[Callable[..., Any]] = [
    click.option("--flavor", default=None, help="Hardware flavor of the Job, e.g. 'cpu-upgrade' or 'a10g-small'."),
    click.option("--timeout", default=None, help="Max duration of the Job, e.g. 300, '30m', '2h', '1d'."),
    click.option(
        "--services-timeout",
        default=None,
        help=f"Max duration of each service Job. Defaults to --timeout, or to {DEFAULT_SERVICES_TIMEOUT}.",
    ),
    click.option("--name", default=None, help="Name of the main Job."),
    click.option("-l", "--label", "labels", multiple=True, help="Label of the main Job as key=value (repeatable)."),
    click.option(
        "-e", "--env", multiple=True, help="Environment variable as KEY=VALUE, or KEY to forward the local one."
    ),
    click.option(
        "--env-file", default=None, help="Path to a dotenv file with the environment variables, or '-' for stdin."
    ),
    click.option("-s", "--secrets", multiple=True, help="Secret as KEY=VALUE, or KEY to forward the local one."),
    click.option("--secrets-file", default=None, help="Path to a dotenv file with the secrets, or '-' for stdin."),
    click.option(
        "-v", "--volume", "volumes", multiple=True, help="Volume to mount: hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro]."
    ),
    click.option("--expose", multiple=True, type=int, help="Port of the Job to expose through the Jobs proxy."),
    click.option("--ssh", is_flag=True, default=False, help="Enable SSH access to the Job."),
    click.option("--resource-group-id", default=None, help="Resource group shared by the Job and its services."),
    click.option(
        "--namespace", default=None, help="Namespace of the Job and its services. Defaults to the current user."
    ),
    click.option("--token", default=None, help="User access token. Defaults to the locally saved token."),
    click.option("-d", "--detach", is_flag=True, default=False, help="Return as soon as the Job is started."),
    click.option("--dry-run", is_flag=True, default=False, help="Show what would run, without starting anything."),
]


def common_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Add the shared run options to a command (decorators apply bottom-up, so iterate in reverse)."""
    for option in reversed(COMMON_OPTIONS):
        func = option(func)
    return func


SERVICES_OPTION = click.option(
    "--with-services",
    default=None,
    metavar="SPEC",
    help="Services to run alongside the Job: a template call (e.g. 'dask(num_workers=4)', see "
    "`hf jobs-services templates`) or the path of a services file.",
)


def launch(
    api: HfApi,
    namespace: str,
    *,
    token: str | None,
    specs: list[ServiceSpec],
    source: str,
    group: str,
    services_timeout: str,
    resource_group_id: str | None,
    detach: bool,
    start_main_job: Callable[[], JobInfo],
) -> None:
    """Start the services, wait for them to run, start the main Job, and leave nothing behind."""
    log(f"Services loaded from: {source}")
    started: list[ServiceJob] = []
    main_job: JobInfo | None = None
    try:
        started = start_services(
            api,
            specs,
            group=group,
            namespace=namespace,
            token=token,
            timeout=services_timeout,
            resource_group_id=resource_group_id,
        )
        wait_for_services(api, started, namespace=namespace, token=token, timeout=_seconds(services_timeout))
        hint_connect(group)
        main_job = start_main_job()
        result("Job started", id=main_job.id, url=main_job.url)

        if detach:
            hint(f"Use `hf jobs logs -f {namespace}/{main_job.id}` to stream the logs.")
            hint(f"Services stay up: stop them with `hf jobs-services stop {group}`, or let their timeout expire.")
            return

        final = stream_logs(api, main_job, namespace=namespace)
        if final.status.stage != JobStage.COMPLETED:
            message = f": {final.status.message}" if final.status.message else ""
            raise ServicesError(f"Job {final.id} finished with stage '{stage_name(final.status.stage)}'{message}")
    except KeyboardInterrupt:
        log("Interrupted: cancelling the Job and its services...")
        if main_job is not None:
            api.cancel_job(job_id=main_job.id, namespace=namespace)
        raise click.Abort() from None
    finally:
        # In detached mode the services are left running on purpose (the Job keeps using them).
        if main_job is None or not detach:
            cancel_services(api, started, namespace=namespace)


def main_labels(labels: tuple[str, ...], name: str | None, group: str) -> dict[str, str]:
    """Labels of the main Job: the ones the user asked for, plus the ones `ls` and `stop` filter on."""
    return {**(parse_labels(list(labels), name) or {}), GROUP_LABEL: group, SERVICE_LABEL: MAIN_SERVICE}


def main_aliases(specs: list[ServiceSpec]) -> list[str] | None:
    """The `main` alias is only claimed when no service already answers to it."""
    return None if MAIN_SERVICE in {spec.alias for spec in specs} else [MAIN_SERVICE]


@cli.command("run", context_settings=RUN_CONTEXT)
@SERVICES_OPTION
@click.argument("image", metavar="IMAGE")
@click.argument("command", nargs=-1, type=click.UNPROCESSED, metavar="COMMAND...")
@common_options
@handle_errors
def run(
    with_services: str,
    image: str,
    command: tuple[str, ...],
    flavor: str | None,
    timeout: str | None,
    services_timeout: str | None,
    name: str | None,
    labels: tuple[str, ...],
    env: tuple[str, ...],
    env_file: str | None,
    secrets: tuple[str, ...],
    secrets_file: str | None,
    volumes: tuple[str, ...],
    expose: tuple[int, ...],
    ssh: bool,
    resource_group_id: str | None,
    namespace: str | None,
    token: str | None,
    detach: bool,
    dry_run: bool,
) -> None:
    """Run a container Job, alongside services in a shared network group.

    Like `hf jobs run` (and `docker run`): the image and the command to run in it come first. The services
    are started first and waited for, the Job starts once they are all RUNNING, and the services are
    canceled when the Job ends (or when this command is interrupted).

    \b
    Examples:
      hf jobs-services run --with-services "dask(num_workers=4)" python:3.12 python my_script.py --epochs 3
      hf jobs-services run --with-services "ray" python:3.12 sh -c 'python -c "import ray; ray.init()"'
    """
    if not command:
        raise ServicesError(
            f"Missing COMMAND. Example: hf jobs-services run --with-services ray {image} python -c 'print(1)'"
        )
    source = with_services or find_default_services_file()
    if source is None:
        raise ServicesError(
            f"No services file found. Pass --with-services, or write one of: {', '.join(DEFAULT_SERVICES_FILENAMES)}."
        )
    specs = build_service_specs(resolve_services(source))
    group = new_group_name()
    services_timeout = services_timeout or timeout or DEFAULT_SERVICES_TIMEOUT

    if dry_run:
        _print_plan(
            source,
            specs,
            group,
            {"image": image, "command": " ".join(command)},
            flavor,
            timeout,
            services_timeout,
            detach,
        )
        return

    api = HfApi(token=token)
    namespace = namespace or api.whoami(token=token)["name"]
    resolved_token = token if isinstance(token, str) else get_token()

    def start_main_job() -> JobInfo:
        return api.run_job(
            image=image,
            command=list(command),
            env=parse_env_map(list(env), env_file, token=resolved_token) or None,
            secrets=parse_env_map(list(secrets), secrets_file, token=resolved_token) or None,
            flavor=flavor,
            timeout=timeout,
            name=name,
            labels=main_labels(labels, name, group),
            volumes=parse_volumes(list(volumes)) or None,
            expose=list(expose) or None,
            ssh=ssh,
            network_group=group,
            network_aliases=main_aliases(specs),
            resource_group_id=resource_group_id,
            namespace=namespace,
            token=token,
        )

    launch(
        api,
        namespace,
        token=token,
        specs=specs,
        source=source,
        group=group,
        services_timeout=services_timeout,
        resource_group_id=resource_group_id,
        detach=detach,
        start_main_job=start_main_job,
    )


@uv.command("run", context_settings=RUN_CONTEXT)
@SERVICES_OPTION
@click.argument("script", metavar="SCRIPT")
@click.argument("script_args", nargs=-1, type=click.UNPROCESSED, metavar="[ARGS]...")
@click.option("--with", "dependencies", multiple=True, help="Python dependency for the script (repeatable).")
@click.option("-p", "--python", "python_version", default=None, help="Python version for the script, e.g. '3.11'.")
@click.option("--image", default=None, help="Base image of the Job. Defaults to the default UV image.")
@common_options
@handle_errors
def run_uv(
    with_services: str,
    script: str,
    script_args: tuple[str, ...],
    dependencies: tuple[str, ...],
    python_version: str | None,
    image: str | None,
    flavor: str | None,
    timeout: str | None,
    services_timeout: str | None,
    name: str | None,
    labels: tuple[str, ...],
    env: tuple[str, ...],
    env_file: str | None,
    secrets: tuple[str, ...],
    secrets_file: str | None,
    volumes: tuple[str, ...],
    expose: tuple[int, ...],
    ssh: bool,
    resource_group_id: str | None,
    namespace: str | None,
    token: str | None,
    detach: bool,
    dry_run: bool,
) -> None:
    """Run a UV script as a Job, alongside services in a shared network group.

    Like `hf jobs uv run`, with services: the services are started first and waited for, the script starts
    once they are all RUNNING, and the services are canceled when the script's Job ends (or when this
    command is interrupted).

    \b
    Examples:
      hf jobs-services uv run --with-services "dask(num_workers=4)" my_dask_script.py
      hf jobs-services uv run my_script.py --epochs 3    # reads my_script-services.yml or jobs-services.yml
    """
    source = with_services or discover_services_file(script)
    if source is None:
        raise ServicesError(
            f"No services file found for '{script}'. Pass --with-services, or write one of: "
            f"{', '.join(services_candidates(script))}."
        )
    specs = build_service_specs(resolve_services(source))
    check_local_files(script, list(script_args))
    group = new_group_name()
    services_timeout = services_timeout or timeout or DEFAULT_SERVICES_TIMEOUT

    if dry_run:
        _print_plan(
            source,
            specs,
            group,
            {"script": script, "args": " ".join(script_args) or None},
            flavor,
            timeout,
            services_timeout,
            detach,
        )
        return

    api = HfApi(token=token)
    namespace = namespace or api.whoami(token=token)["name"]
    resolved_token = token if isinstance(token, str) else get_token()

    def start_main_job() -> JobInfo:
        return api.run_uv_job(
            script,
            script_args=list(script_args),
            dependencies=list(dependencies) or None,
            python=python_version,
            image=image,
            env=parse_env_map(list(env), env_file, token=resolved_token) or None,
            secrets=parse_env_map(list(secrets), secrets_file, token=resolved_token) or None,
            flavor=flavor,
            timeout=timeout,
            name=name,
            labels=main_labels(labels, name, group),
            volumes=parse_volumes(list(volumes)) or None,
            expose=list(expose) or None,
            ssh=ssh,
            network_group=group,
            network_aliases=main_aliases(specs),
            resource_group_id=resource_group_id,
            namespace=namespace,
            token=token,
        )

    launch(
        api,
        namespace,
        token=token,
        specs=specs,
        source=source,
        group=group,
        services_timeout=services_timeout,
        resource_group_id=resource_group_id,
        detach=detach,
        start_main_job=start_main_job,
    )


@cli.command("ls | list")
@click.option("--group", default=None, help="Only show the Jobs of this network group.")
@click.option("--namespace", default=None, help="Namespace to look into. Defaults to the current user.")
@click.option("--token", default=None, help="User access token. Defaults to the locally saved token.")
@handle_errors
def ls(group: str | None, namespace: str | None, token: str | None) -> None:
    """List the running Jobs started by hf jobs-services."""
    api = HfApi(token=token)
    namespace = namespace or api.whoami(token=token)["name"]
    jobs = list_group_jobs(api, namespace=namespace, token=token, group=group)
    if not jobs:
        warn("No services Job is running.")
        hint('Start one with: hf jobs-services uv run --with-services "dask(num_workers=4)" my_script.py')
        return
    groups = {(job.labels or {})[GROUP_LABEL] for job in jobs}
    table(
        [
            {
                "GROUP": (job.labels or {})[GROUP_LABEL],
                "SERVICE": (job.labels or {}).get("service", MAIN_SERVICE),
                "STAGE": stage_name(job.status.stage),
                "ID": f"{namespace}/{job.id}",
            }
            for job in jobs
        ],
        ["GROUP", "SERVICE", "STAGE", "ID"],
    )
    if len(groups) == 1:
        hint(f"Stop them with: hf jobs-services stop {groups.pop()}")


@cli.command("stop")
@click.argument("group")
@click.option("-y", "--yes", is_flag=True, default=False, help="Answer yes to the confirmation prompt.")
@click.option("--namespace", default=None, help="Namespace of the group. Defaults to the current user.")
@click.option("--token", default=None, help="User access token. Defaults to the locally saved token.")
@handle_errors
def stop(group: str, yes: bool, namespace: str | None, token: str | None) -> None:
    """Cancel every running Job of a services group."""
    api = HfApi(token=token)
    namespace = namespace or api.whoami(token=token)["name"]
    jobs = list_group_jobs(api, namespace=namespace, token=token, group=group)
    if not jobs:
        raise ServicesError(f"No running Job found for services group '{group}' in namespace '{namespace}'.")
    if not yes:
        click.confirm(f"Cancel {len(jobs)} Job(s) of services group '{group}'?", abort=True)
    canceled = cancel_group_jobs(api, jobs, namespace=namespace)
    result("Services group stopped", group=group, canceled=canceled)


@cli.command("templates")
def templates() -> None:
    """List the available services templates."""
    for name, template in SERVICES_TEMPLATES.items():
        summary = (template.__doc__ or "").strip().splitlines()[0]
        click.echo(f"{name}({template_params(name)})")
        click.echo(f"    {summary}")
    hint('Use one with: hf jobs-services uv run --with-services "ray(num_workers=4)" my_ray_script.py')


def _print_plan(
    source: str,
    specs: list[ServiceSpec],
    group: str,
    main: dict[str, Any],
    flavor: str | None,
    timeout: str | None,
    services_timeout: str,
    detach: bool,
) -> None:
    """What `--dry-run` would start: nothing is created."""
    result(
        "Dry run: nothing started",
        services_file=source,
        services_group=group,
        services=len(specs),
        **main,
        flavor=flavor,
        job_timeout=timeout,
        services_timeout=services_timeout,
        detach=detach or None,
    )
    table(
        [{"ALIAS": spec.alias, "IMAGE": spec.image, "FLAVOR": spec.flavor or "cpu-basic"} for spec in specs],
        ["ALIAS", "IMAGE", "FLAVOR"],
    )
    hint(f"The Job reaches the services at ${{HF_NETWORK_GROUP_PREFIX}}<ALIAS>:<PORT>, e.g. {specs[0].alias}")


def check_local_files(script: str, script_args: Sequence[str]) -> None:
    """Fail before starting anything when a local script referenced by the Job does not exist."""
    missing = [
        candidate
        for candidate in [script, *script_args]
        if candidate.endswith((".py", ".sh", ".yaml", ".yml", ".toml"))
        and not candidate.startswith(("http://", "https://"))
        and not Path(candidate).is_file()
    ]
    if missing:
        raise ServicesError(f"Script file not found: {', '.join(missing)}")


def _seconds(duration: str | float) -> float:
    """Convert a Job duration ('30m', '2h', 3600) to seconds, for `wait_for_job` timeouts."""
    if isinstance(duration, str) and duration and duration[-1] in "smhd":
        units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        return float(duration[:-1]) * units[duration[-1]]
    return float(duration)


def _api_message(error: HfHubHTTPError) -> str:
    """Prefer the message the Hub sent back: `str(error)` mostly repeats the local request context."""
    if error.server_message:
        return error.server_message
    with contextlib.suppress(ValueError, AttributeError):
        # `json()` raises on an HTML/text body, and `response` is duck-typed in tests.
        message = error.response.json().get("error")
        if message:
            return str(message)
    return str(error)


def main() -> None:
    # `run_uv_job` is flagged experimental in huggingface_hub: the warning is noise on every single run.
    warnings.filterwarnings("ignore", message=r".*is experimental.*", category=UserWarning)
    cli(prog_name="hf jobs-services")


if __name__ == "__main__":
    main()
