from __future__ import annotations

import pytest
import yaml

from hf_jobs_services.errors import ServicesError
from hf_jobs_services.parsing import parse_dotenv, parse_env_map, parse_labels, parse_volumes
from hf_jobs_services.services import (
    build_service_specs,
    discover_services_file,
    resolve_services,
    services_candidates,
    template_params,
)


def test_template_call_expands_replicas():
    specs = build_service_specs(resolve_services("dask(num_workers=3)"))
    assert [spec.alias for spec in specs] == ["dask-scheduler", "dask-worker-0", "dask-worker-1", "dask-worker-2"]
    assert specs[0].flavor == "cpu-upgrade"
    # the worker connects to the scheduler through the network group prefix, resolved at runtime
    assert "${HF_NETWORK_GROUP_PREFIX}dask-scheduler:8786" in specs[1].command[-1]


def test_template_call_casts_arguments():
    specs = build_service_specs(resolve_services('ray(num_workers=2, image="python:3.11", disable_usage_stats=false)'))
    assert specs[0].image == "python:3.11"
    assert specs[0].env == {"RAY_DEDUP_LOGS": "false"}
    assert "ray start --head" in specs[0].command[-1]


def test_unknown_template_lists_available_ones():
    with pytest.raises(ServicesError, match="Unknown services template 'redis'"):
        resolve_services("redis(num_workers=1)")


def test_bad_template_argument_shows_usage():
    with pytest.raises(ServicesError, match="Usage: dask\\("):
        resolve_services("dask(nope=1)")


def test_services_file_is_discovered_next_to_the_script(tmp_path, monkeypatch):
    (tmp_path / "my_script.py").write_text("""print("hi")""")
    (tmp_path / "my_script-services.yml").write_text("services: {}")
    (tmp_path / "jobs-services.yml").write_text("services: {}")
    monkeypatch.chdir(tmp_path)
    assert discover_services_file("my_script.py") == "my_script-services.yml"


def test_services_file_falls_back_to_the_default_name(tmp_path, monkeypatch):
    (tmp_path / "my_script.py").write_text("""print("hi")""")
    (tmp_path / "jobs-services.yaml").write_text("services: {}")
    monkeypatch.chdir(tmp_path)
    assert discover_services_file("my_script.py") == "jobs-services.yaml"


def test_no_services_file_is_discovered_for_a_url_script():
    assert discover_services_file("https://huggingface.co/foo/bar.py") is None
    assert services_candidates("https://huggingface.co/foo/bar.py")[-2:] == [
        "jobs-services.yml",
        "jobs-services.yaml",
    ]


def test_yaml_file(tmp_path):
    path = tmp_path / "my_script-services.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "server": {
                        "image": "python:3.12",
                        "command": "python -m http.server 8000",
                        "expose": [8000],
                        "env": {"PORT": 8000, "DEBUG": True},
                    },
                    "worker": {"image": "busybox", "command": ["sleep", "infinity"], "replicas": 2},
                }
            }
        )
    )
    specs = build_service_specs(resolve_services(str(path)))
    assert [spec.alias for spec in specs] == ["server", "worker-0", "worker-1"]
    assert specs[0].command == ["sh", "-c", "python -m http.server 8000"]
    assert specs[0].expose == [8000]
    assert specs[0].env == {"PORT": "8000", "DEBUG": "true"}  # YAML scalars are normalized to strings


def test_yaml_without_services_section(tmp_path):
    path = tmp_path / "nope.yml"
    path.write_text("hello: world")
    with pytest.raises(ServicesError, match="non-empty 'services' section"):
        build_service_specs(resolve_services(str(path)))


def test_service_requires_image_and_command(tmp_path):
    path = tmp_path / "bad.yml"
    path.write_text(yaml.safe_dump({"services": {"server": {"image": "python:3.12"}}}))
    with pytest.raises(ServicesError, match="must have 'image' and 'command'"):
        build_service_specs(resolve_services(str(path)))


def test_unsupported_key_is_explained(tmp_path):
    path = tmp_path / "services.yml"
    path.write_text(yaml.safe_dump({"services": {"server": {"image": "x", "command": "x", "depends_on": ["db"]}}}))
    with pytest.raises(ServicesError, match="start in parallel"):
        build_service_specs(resolve_services(str(path)))


def test_service_names_are_sanitized(tmp_path):
    path = tmp_path / "services.yml"
    path.write_text(yaml.safe_dump({"services": {"My Server_1": {"image": "x", "command": "x"}}}))
    assert build_service_specs(resolve_services(str(path)))[0].alias == "my-server-1"


def test_missing_services_file():
    with pytest.raises(ServicesError, match="Services file not found"):
        resolve_services("/does/not/exist.yml")


def test_parse_dotenv_pulls_local_and_token_values(monkeypatch):
    monkeypatch.setenv("LOCAL_ONLY", "from-env")
    assert parse_dotenv("A=1\nB='two words'\nLOCAL_ONLY") == {"A": "1", "B": "two words", "LOCAL_ONLY": "from-env"}
    assert parse_dotenv("export C=3") == {"C": "3"}
    assert parse_env_map(["HF_TOKEN"], token="secret") == {"HF_TOKEN": "secret"}
    assert parse_env_map(["NOT_SET"]) == {"NOT_SET": ""}


def test_parse_dotenv_rejects_invalid_key():
    with pytest.raises(ServicesError, match="Invalid environment variable name"):
        parse_dotenv("not a key value")


def test_parse_labels_merges_name():
    assert parse_labels(["env=prod"], "run-1") == {"env": "prod", "name": "run-1"}
    assert parse_labels([]) is None


def test_parse_volumes():
    volumes = parse_volumes(["hf://datasets/org/ds:/data", "hf://buckets/org/b/sub:/mnt:rw"])
    # read_only stays None when no :ro/:rw flag was given: repos are mounted read-only by the API anyway.
    assert [(v.type, v.source, v.mount_path, v.read_only, v.path) for v in volumes] == [
        ("dataset", "org/ds", "/data", None, None),
        ("bucket", "org/b", "/mnt", False, "sub"),
    ]


def test_parse_volumes_rejects_garbage():
    with pytest.raises(ServicesError, match="Invalid volume"):
        parse_volumes(["not-a-volume-spec"])


def test_template_params_is_human_readable():
    assert template_params("dask").startswith("num_workers=2")
