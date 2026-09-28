from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner
from huggingface_hub import JobStage
from huggingface_hub.errors import HfHubHTTPError

from hf_jobs_services import cli as cli_module
from hf_jobs_services.cli import cli


class FakeApi:
    """Records the calls the CLI makes instead of talking to the Hub."""

    instances: list["FakeApi"] = []

    def __init__(self, final_stage="COMPLETED", *args, **kwargs):
        self.final_stage = final_stage
        self.started = []  # (labels, kwargs) of every started Job
        self.canceled = []
        FakeApi.instances.append(self)

    def whoami(self, token=None):
        return {"name": "me"}

    def run_job(self, command, **kwargs):
        self.started.append((kwargs.get("labels") or {}, kwargs | {"command": command}))
        return self._job(f"svc{len(self.started)}")

    def run_uv_job(self, script, **kwargs):
        self.started.append((kwargs.get("labels") or {}, kwargs | {"script": script}))
        return self._job("main")

    def wait_for_job(self, job_id, stages=None, **kwargs):
        ids = job_id if isinstance(job_id, list) else [job_id]
        infos = [self._info(id) for id in ids]
        return infos if stages is not None else infos[0]

    def fetch_job_logs(self, job_id=None, follow=True, **kwargs):
        # the real method is a generator of raw log lines
        yield "hello from the job\n"

    def inspect_job(self, job_id=None, **kwargs):
        return self._info(job_id)

    def _info(self, id):
        stage = "RUNNING" if id.startswith("svc") else self.final_stage
        return SimpleNamespace(id=id, status=SimpleNamespace(stage=JobStage(stage), message=None), labels={})

    def cancel_job(self, job_id=None, namespace=None, **kwargs):
        self.canceled.append(job_id)

    def list_jobs(self, **kwargs):
        return [
            SimpleNamespace(
                id="svc1",
                labels={"services-group": "services-x", "service": "server"},
                status=SimpleNamespace(stage=JobStage.RUNNING, message=None),
            )
        ]

    def _job(self, id):
        return SimpleNamespace(id=id, url=f"https://huggingface.co/jobs/me/{id}", labels={})

    @property
    def services(self):
        return [labels.get("service") for labels, _ in self.started if "service" in labels]


def last() -> FakeApi:
    return FakeApi.instances[-1]


@pytest.fixture(autouse=True)
def _reset_instances():
    FakeApi.instances.clear()


@pytest.fixture
def patch_api(monkeypatch):
    monkeypatch.setattr(cli_module, "HfApi", FakeApi)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    (tmp_path / "my_script.py").write_text("print('hi')")
    monkeypatch.chdir(tmp_path)
    return CliRunner()


def test_run_starts_services_then_job_then_cancels(patch_api, workdir):
    result = workdir.invoke(cli, ["run", "--with-services", "dask(num_workers=1)", "my_script.py", "--epochs", "3"])
    assert result.exit_code == 0, result.output

    api = last()
    assert api.services == ["dask-scheduler", "dask-worker", "main"]
    group = api.started[0][1]["network_group"]
    assert group.startswith("services-")
    assert {labels["services-group"] for labels, _ in api.started} == {group}

    main = api.started[-1][1]
    assert main["script"] == "my_script.py"
    assert main["script_args"] == ["--epochs", "3"]
    assert main["network_aliases"] == ["main"]  # the Job is reachable as `main` too
    assert "hello from the job" in result.output
    assert api.canceled == ["svc1", "svc2"]  # services never left behind
    assert f"Services group '{group}'" in result.output


def test_run_dry_run_starts_nothing(workdir):
    result = workdir.invoke(cli, ["run", "--with-services", "ray", "my_script.py", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output and "ray-head" in result.output
    assert FakeApi.instances == []


def test_run_fails_before_spending_anything_when_script_is_missing(patch_api, workdir):
    result = workdir.invoke(cli, ["run", "--with-services", "ray", "nope.py"])
    assert result.exit_code == 1
    assert "Script file not found: nope.py" in result.output
    assert FakeApi.instances == [] or last().started == []


def test_run_reports_a_failed_job_and_cancels_services(monkeypatch, workdir):
    monkeypatch.setattr(cli_module, "HfApi", lambda *a, **k: FakeApi(final_stage="ERROR"))
    result = workdir.invoke(cli, ["run", "--with-services", "ray", "my_script.py"])
    assert result.exit_code == 1
    assert "finished with stage" in result.output
    assert last().canceled


def test_run_detached_leaves_services_running(patch_api, workdir):
    result = workdir.invoke(cli, ["run", "-d", "--with-services", "ray", "my_script.py"])
    assert result.exit_code == 0, result.output
    assert last().canceled == []
    assert "hf jobs-services stop services-" in result.output


def test_run_forwards_script_arguments_after_a_separator(patch_api, workdir):
    result = workdir.invoke(cli, ["run", "--with-services", "ray", "my_script.py", "--", "--flavor", "x"])
    assert result.exit_code == 0, result.output
    assert last().started[-1][1]["script_args"] == ["--flavor", "x"]


def test_run_passes_env_secrets_and_volumes(patch_api, workdir, monkeypatch):
    monkeypatch.setenv("MY_LOCAL", "local-value")
    result = workdir.invoke(
        cli,
        [
            "run",
            "--with-services",
            "ray",
            "my_script.py",
            "-e",
            "A=1",
            "-e",
            "MY_LOCAL",
            "-s",
            "KEY=secret",
            "-v",
            "hf://datasets/org/ds:/data",
            "--flavor",
            "cpu-upgrade",
            "--label",
            "team=ml",
        ],
    )
    assert result.exit_code == 0, result.output
    _, main = last().started[-1]
    assert main["env"] == {"A": "1", "MY_LOCAL": "local-value"}
    assert main["secrets"] == {"KEY": "secret"}
    assert [(v.type, v.source, v.mount_path) for v in main["volumes"]] == [("dataset", "org/ds", "/data")]
    assert main["flavor"] == "cpu-upgrade"
    assert main["labels"] == {"team": "ml", "services-group": last().started[0][1]["network_group"], "service": "main"}


def test_main_job_gives_up_its_alias_when_taken(patch_api, workdir, tmp_path):
    (tmp_path / "my_script-services.yml").write_text(
        yaml.safe_dump({"services": {"main": {"image": "python:3.12", "command": "sleep infinity"}}})
    )
    result = workdir.invoke(cli, ["run", "--with-services", "my_script-services.yml", "my_script.py"])
    assert result.exit_code == 0, result.output
    assert last().started[-1][1]["network_aliases"] is None


def test_ls_and_its_alias(patch_api, workdir):
    for command in ("ls", "list"):
        result = workdir.invoke(cli, [command])
        assert result.exit_code == 0, result.output
        assert "services-x" in result.output and "server" in result.output
        assert "hf jobs-services stop services-x" in result.output


def test_stop_cancels_the_group(patch_api, workdir):
    result = workdir.invoke(cli, ["stop", "services-x", "-y"])
    assert result.exit_code == 0, result.output
    assert last().canceled == ["svc1"]


def test_stop_on_unknown_group(patch_api, workdir):
    result = workdir.invoke(cli, ["stop", "services-nope", "-y"])
    assert result.exit_code == 1
    assert "No running Job found" in result.output


def test_service_yaml_is_loaded_from_file(patch_api, workdir, tmp_path):
    (tmp_path / "my_script-services.yml").write_text(
        yaml.safe_dump({"services": {"server": {"image": "python:3.12", "command": "python -m http.server"}}})
    )
    result = workdir.invoke(cli, ["run", "--with-services", "my_script-services.yml", "my_script.py"])
    assert result.exit_code == 0, result.output
    labels, kwargs = last().started[0]
    assert labels["service"] == "server"
    assert kwargs["command"] == ["sh", "-c", "python -m http.server"]


def test_run_reads_the_services_file_of_the_script_by_default(patch_api, workdir, tmp_path):
    (tmp_path / "my_script-services.yml").write_text(
        yaml.safe_dump({"services": {"server": {"image": "python:3.12", "command": "sleep infinity"}}})
    )
    result = workdir.invoke(cli, ["run", "my_script.py"])
    assert result.exit_code == 0, result.output
    assert "Services loaded from: my_script-services.yml" in result.output
    assert last().services == ["server", "main"]


def test_run_without_any_services_file_lists_what_it_looked_for(patch_api, workdir):
    result = workdir.invoke(cli, ["run", "my_script.py"])
    assert result.exit_code == 1
    assert "No services file found for 'my_script.py'" in result.output
    assert "my_script-services.yml" in result.output and "jobs-services.yml" in result.output


def test_api_errors_are_rendered_short(workdir, monkeypatch):
    response = SimpleNamespace(headers={}, request=None, json=lambda: {"error": "Flavor not found"})

    def boom(*args, **kwargs):
        raise HfHubHTTPError("Job creation failed", response=response)

    monkeypatch.setattr(
        cli_module,
        "HfApi",
        lambda *a, **k: SimpleNamespace(whoami=lambda token=None: {"name": "me"}, run_job=boom),
    )
    result = workdir.invoke(cli, ["run", "--with-services", "ray", "my_script.py"])
    assert result.exit_code == 1
    assert "Flavor not found" in result.output
    assert "Traceback" not in result.output


def test_templates_lists_the_available_ones(workdir):
    result = workdir.invoke(cli, ["templates"])
    assert result.exit_code == 0
    for name in ("ray", "dask", "spark", "spark_connect"):
        assert name in result.output


def test_group_help_is_prefixed_with_hf(workdir):
    result = workdir.invoke(cli, ["--help"], prog_name="hf jobs-services")
    assert result.exit_code == 0
    assert "Usage: hf jobs-services" in result.output


def test_unknown_command_is_reported(workdir):
    result = workdir.invoke(cli, ["nope"])
    assert result.exit_code == 2
    assert "No such command" in result.output
