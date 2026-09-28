"""Services templates for `hf jobs-services`.

Each template returns a services dict describing the services to run alongside the main job:

```python
>>> from hf_jobs_services.templates import dask
>>> dask(num_workers=2)["services"].keys()
dict_keys(['dask-scheduler', 'dask-worker'])
```

Services live in the same Jobs network group as the main job, so they are reachable from it at
`${HF_NETWORK_GROUP_PREFIX}<service-name>:<port>` (see each template docstring for the connection string).

Ported from `huggingface_hub.jobs_services` (PR #4931) so it ships with the extension rather than with the client.
"""

from typing import Any


def ray(
    *,
    num_workers: int = 2,
    image: str | None = None,
    ray_version: str | None = None,
    disable_usage_stats: bool = True,
    dashboard_host: str = "0.0.0.0",
    flavor: str = "cpu-upgrade",
    head_flavor: str = "cpu-upgrade",
) -> dict[str, Any]:
    """A Ray cluster: one `ray-head` + `num_workers` `ray-worker` services.

    The head exposes the Ray dashboard on port 8265. Workers join the head over the network group.

    Connect from the main job with:

    ```python
    from ray.job_submission import JobSubmissionClient
    client = JobSubmissionClient(f"http://{os.environ['HF_NETWORK_GROUP_PREFIX']}ray-head:8265")
    ```

    Direct `ray.init(address="auto")` is not available on HF Compute (GCS IP advertisement), submit work
    with `JobSubmissionClient` instead.

    Args:
        num_workers: Number of worker nodes. Defaults to 2.
        image: Docker image to use. Defaults to `"python:3.12"`.
        ray_version: Ray version to install. Defaults to the latest.
        disable_usage_stats: Whether to disable Ray usage stats reporting. Defaults to True.
        dashboard_host: Host to bind the Ray dashboard to. Defaults to `"0.0.0.0"`.
        flavor: Hardware flavor of the workers. Defaults to `"cpu-upgrade"`.
        head_flavor: Hardware flavor of the head. Defaults to `"cpu-upgrade"`.
    """
    image = image or "python:3.12"
    ray_flag = "" if ray_version is None else f"=={ray_version}"
    ray_pkg = f"ray{ray_flag}[default]" if ray_flag else "ray[default]"
    usage_stats_flag = "--disable-usage-stats" if disable_usage_stats else ""

    head_cmd = (
        f"pip install -q {ray_pkg} && "
        f"ray start --head --port=6379 --dashboard-host={dashboard_host} "
        f"{usage_stats_flag} && "
        "sleep infinity"
    )
    worker_cmd = (
        f"pip install -q {ray_pkg} && "
        f"ray start --address=${{HF_NETWORK_GROUP_PREFIX}}ray-head:6379 "
        f"{usage_stats_flag} && "
        "sleep infinity"
    )

    return {
        "services": {
            "ray-head": {
                "image": image,
                "command": ["sh", "-c", head_cmd],
                "env": {"RAY_DEDUP_LOGS": "false"},
                "flavor": head_flavor,
            },
            "ray-worker": {
                "image": image,
                "command": ["sh", "-c", worker_cmd],
                "flavor": flavor,
                "replicas": num_workers,
            },
        },
    }


def dask(
    *,
    num_workers: int = 2,
    image: str | None = None,
    dask_version: str | None = None,
    flavor: str = "cpu-upgrade",
    scheduler_flavor: str = "cpu-upgrade",
) -> dict[str, Any]:
    """A Dask cluster: one `dask-scheduler` + `num_workers` `dask-worker` services.

    Connect from the main job with:

    ```python
    from dask.distributed import Client
    client = Client(f"http://{os.environ['HF_NETWORK_GROUP_PREFIX']}dask-scheduler:8786")
    ```

    Args:
        num_workers: Number of worker nodes. Defaults to 2.
        image: Docker image to use. Defaults to `"python:3.12"`.
        dask_version: `dask`/`distributed` version to install. Defaults to the latest.
        flavor: Hardware flavor of the workers. Defaults to `"cpu-upgrade"`.
        scheduler_flavor: Hardware flavor of the scheduler. Defaults to `"cpu-upgrade"`.
    """
    image = image or "python:3.12"
    version_flag = f"=={dask_version}" if dask_version else ""
    dask_pkg = f"dask{version_flag} distributed{version_flag} pandas pyarrow"

    scheduler_cmd = f"pip install -q {dask_pkg} && sleep 3 && dask scheduler --host 0.0.0.0"
    worker_cmd = (
        f"pip install -q {dask_pkg} && sleep 5 && dask worker tcp://${{HF_NETWORK_GROUP_PREFIX}}dask-scheduler:8786"
    )

    return {
        "services": {
            "dask-scheduler": {
                "image": image,
                "command": ["sh", "-c", scheduler_cmd],
                "flavor": scheduler_flavor,
            },
            "dask-worker": {
                "image": image,
                "command": ["sh", "-c", worker_cmd],
                "flavor": flavor,
                "replicas": num_workers,
            },
        },
    }


def spark(
    *,
    num_workers: int = 2,
    spark_version: str = "3.5.1",
    flavor: str = "cpu-upgrade",
    master_flavor: str = "cpu-upgrade",
) -> dict[str, Any]:
    """A Spark standalone cluster: one `spark-master` + `num_workers` `spark-worker` services.

    Reach the master from the main job with
    `spark://${HF_NETWORK_GROUP_PREFIX}spark-master:7077` (thrift server on 9083).

    Note: the main job needs a JVM image to talk to this cluster, so `hf jobs-services run` with a plain UV
    script works better with [`spark_connect`] than with [`spark`].

    Args:
        num_workers: Number of worker nodes. Defaults to 2.
        spark_version: Spark image tag. Defaults to `"3.5.1"`.
        flavor: Hardware flavor of the workers. Defaults to `"cpu-upgrade"`.
        master_flavor: Hardware flavor of the master. Defaults to `"cpu-upgrade"`.
    """
    image = f"apache/spark:{spark_version}"
    spark_master_cmd = (
        "/opt/spark/sbin/start-master.sh && "
        "sleep 5 && "
        "/opt/spark/sbin/start-thriftserver.sh --master spark://$(hostname -i):7077 "
        "--driver-memory 2g --executor-memory 1g "
        "--conf spark.driver.host=$(hostname -i) > /dev/null 2>&1 & "
        "tail -f /dev/null"
    )
    spark_worker_cmd = (
        "/opt/spark/sbin/start-worker.sh spark://${HF_NETWORK_GROUP_PREFIX}spark-master:7077 && tail -f /dev/null"
    )

    return {
        "services": {
            "spark-master": {
                "image": image,
                "command": ["sh", "-c", spark_master_cmd],
                "flavor": master_flavor,
            },
            "spark-worker": {
                "image": image,
                "command": ["sh", "-c", spark_worker_cmd],
                "flavor": flavor,
                "replicas": num_workers,
            },
        },
    }


def spark_connect(
    *,
    num_workers: int = 2,
    spark_version: str = "3.5.1",
    flavor: str = "cpu-upgrade",
    master_flavor: str = "cpu-upgrade",
    connect_flavor: str = "cpu-upgrade",
) -> dict[str, Any]:
    """A Spark cluster with a Connect server: `spark-master`, `spark-connect` + `num_workers` `spark-worker`.

    Unlike [`spark`], the main job can be a plain UV script since the Connect client is pure Python.
    Connect from the main job with:

    ```python
    from pyspark.sql import SparkSession
    prefix = os.environ.get("HF_NETWORK_GROUP_PREFIX", "")
    spark = (
        SparkSession.builder.remote(f"sc://{prefix}spark-connect:15002")
        .appName("hf-jobs")
        .getOrCreate()
    )
    ```

    Args:
        num_workers: Number of worker nodes. Defaults to 2.
        spark_version: Spark image tag. Defaults to `"3.5.1"`.
        flavor: Hardware flavor of the workers. Defaults to `"cpu-upgrade"`.
        master_flavor: Hardware flavor of the master. Defaults to `"cpu-upgrade"`.
        connect_flavor: Hardware flavor of the Connect server. Defaults to `"cpu-upgrade"`.
    """
    image = f"apache/spark:{spark_version}-python3"
    spark_master_cmd = "/opt/spark/sbin/start-master.sh && tail -f /dev/null"
    spark_worker_cmd = (
        "/opt/spark/sbin/start-worker.sh spark://${HF_NETWORK_GROUP_PREFIX}spark-master:7077 && tail -f /dev/null"
    )
    spark_connect_cmd = (
        "SPARK_MASTER_URL=spark://${HF_NETWORK_GROUP_PREFIX}spark-master:7077 "
        "/opt/spark/sbin/start-connect-server.sh && tail -f /dev/null"
    )

    return {
        "services": {
            "spark-master": {
                "image": image,
                "command": ["sh", "-c", spark_master_cmd],
                "flavor": master_flavor,
            },
            "spark-worker": {
                "image": image,
                "command": ["sh", "-c", spark_worker_cmd],
                "flavor": flavor,
                "replicas": num_workers,
            },
            "spark-connect": {
                "image": image,
                "command": ["sh", "-c", spark_connect_cmd],
                "flavor": connect_flavor,
            },
        },
    }


# CLI lookup: `hf jobs-services run "dask(num_workers=4)" ...` calls `SERVICES_TEMPLATES["dask"](num_workers=4)`.
SERVICES_TEMPLATES: dict[str, Any] = {
    "ray": ray,
    "dask": dask,
    "spark": spark,
    "spark_connect": spark_connect,
}

__all__ = ["SERVICES_TEMPLATES", "dask", "ray", "spark", "spark_connect"]
