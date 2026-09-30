"""Job orchestration: start the services, run the main Job alongside them, then clean the services up."""

import uuid
from dataclasses import dataclass

from huggingface_hub import HfApi, JobInfo, JobStage
from huggingface_hub.errors import HfHubHTTPError

from .errors import ServicesError
from .output import hint, log, result
from .services import ServiceSpec

# Labels every Job started by the extension carries, so `hf jobs-services ls` and `hf jobs-services stop` can
# find them back (the network group name itself is not queryable through the Jobs API).
GROUP_LABEL = "services-group"
SERVICE_LABEL = "service"
MAIN_SERVICE = "main"

ACTIVE_STAGES: list[str] = [JobStage.SCHEDULING.value, JobStage.RUNNING.value]
SERVICE_READY_STAGES: list[JobStage] = [
    JobStage.RUNNING,
    JobStage.COMPLETED,
    JobStage.ERROR,
    JobStage.CANCELED,
    JobStage.DELETED,
]


@dataclass
class ServiceJob:
    """A started service, with the network alias it claims in the services group."""

    alias: str
    service: str
    job: JobInfo


def new_group_name() -> str:
    """Name of the Jobs network group backing one services group: 46 chars, lowercase and dashes."""
    return f"services-{uuid.uuid4().hex[:12]}"


def stage_name(stage: JobStage | str) -> str:
    """`JobStage` is a str-enum but str() gives 'JobStage.RUNNING': always display the bare value."""
    return getattr(stage, "value", None) or str(stage)


def start_services(
    api: HfApi,
    specs: list[ServiceSpec],
    *,
    group: str,
    namespace: str,
    token: str | None,
    timeout: str,
    resource_group_id: str | None = None,
) -> list[ServiceJob]:
    """Start one Job per service spec, all joining `group`. Services already started get canceled on failure."""
    started: list[ServiceJob] = []
    complete = False
    try:
        for spec in specs:
            job = api.run_job(
                image=spec.image,
                command=spec.command,
                env=spec.env,
                secrets=spec.secrets,
                flavor=spec.flavor,
                timeout=timeout,
                name=f"svc-{spec.alias}",
                labels={SERVICE_LABEL: spec.service, GROUP_LABEL: group, **spec.labels},
                volumes=spec.volumes,
                expose=spec.expose,
                ssh=spec.ssh,
                network_group=group,
                network_aliases=[spec.alias],
                resource_group_id=resource_group_id,
                namespace=namespace,
                token=token,
            )
            started.append(ServiceJob(alias=spec.alias, service=spec.service, job=job))
            result(f"Service {spec.alias} started", job=job.id, url=job.url)
        complete = True
    finally:
        if not complete:
            cancel_services(api, started, namespace=namespace)
    return started


def wait_for_services(
    api: HfApi, services: list[ServiceJob], *, namespace: str, token: str | None, timeout: float
) -> None:
    """Block until every service is RUNNING.

    A service that dies or fails while starting cancels the others and fails the run: the main Job is
    not started against a half-built cluster.
    """
    log(f"Waiting for {len(services)} service(s) to be running...")
    try:
        infos = api.wait_for_job(
            [service.job.id for service in services],
            stages=SERVICE_READY_STAGES,
            poll_interval=2,
            timeout=timeout,
            namespace=namespace,
            token=token,
        )
    except TimeoutError as error:
        cancel_services(api, services, namespace=namespace)
        raise ServicesError(
            f"Services did not start in time: {error} Use --services-timeout to allow more time."
        ) from error

    failed = [
        (service, info) for service, info in zip(services, infos, strict=True) if info.status.stage != JobStage.RUNNING
    ]
    if failed:
        cancel_services(api, services, namespace=namespace)
        details = ", ".join(f"{service.alias} ({stage_name(info.status.stage)})" for service, info in failed)
        logs = "\n".join(f"  hf jobs logs {namespace}/{service.job.id}" for service, _ in failed)
        raise ServicesError(f"Service(s) did not start: {details}.\nCheck their logs:\n{logs}")
    log("All services are running.")


def cancel_services(api: HfApi, services: list[ServiceJob], *, namespace: str) -> None:
    """Cancel the given services, best effort: a terminal Job cannot be canceled and that is fine."""
    for service in services:
        try:
            api.cancel_job(job_id=service.job.id, namespace=namespace)
            log(f"Cancelled service {service.alias} ({service.job.id})")
        except HfHubHTTPError as error:
            log(f"Could not cancel service {service.alias} ({service.job.id}): {error}")


def cancel_group_jobs(api: HfApi, jobs: list[JobInfo], *, namespace: str) -> int:
    """Cancel active Jobs of a services group. Returns how many were canceled."""
    canceled = 0
    for job in jobs:
        service = (job.labels or {}).get(SERVICE_LABEL, MAIN_SERVICE)
        try:
            api.cancel_job(job_id=job.id, namespace=namespace)
            result("Job canceled", service=service, job=job.id)
            canceled += 1
        except HfHubHTTPError as error:
            log(f"Could not cancel {job.id} ({service}): {error}")
    return canceled


def stream_logs(api: HfApi, job: JobInfo, *, namespace: str) -> JobInfo:
    """Stream the Job logs until it ends, then return its final info."""
    for line in api.fetch_job_logs(job_id=job.id, namespace=namespace, follow=True):
        print(line, end="", flush=True)
    # The log stream can end while the Job is still scheduling or shutting down: settle the final stage.
    return api.wait_for_job(job_id=job.id, namespace=namespace)


def list_group_jobs(api: HfApi, *, namespace: str, token: str | None, group: str | None = None) -> list[JobInfo]:
    """List the active Jobs started by `hf jobs-services`, optionally filtered to one group."""
    jobs = [
        job
        for job in api.list_jobs(status=ACTIVE_STAGES, namespace=namespace, token=token)
        if (job.labels or {}).get(GROUP_LABEL)
    ]
    if group is not None:
        jobs = [job for job in jobs if (job.labels or {})[GROUP_LABEL] == group]
    return jobs


def hint_connect(group: str) -> None:
    """How the main Job reaches the services."""
    hint(
        f"Services group '{group}': the Job reaches a service at ${{HF_NETWORK_GROUP_PREFIX}}<ALIAS>:<PORT>, "
        "e.g. http://${HF_NETWORK_GROUP_PREFIX}server:8000. Members are resolvable before they are ready: "
        "connect with retries."
    )
