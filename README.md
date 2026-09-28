# hf-jobs-services

An [`hf` CLI extension](https://huggingface.co/docs/huggingface_hub/guides/cli-extensions) to run an HF Job
alongside the long-running Jobs it needs - a Ray head, a Dask scheduler and its workers, a Spark master,
an HTTP server, a mock API - in one [Jobs network
group](https://huggingface.co/docs/huggingface_hub/guides/jobs#network-groups), with the lifecycle handled
for you: services start, then the Job starts, then the services are canceled.

```console
$ hf jobs-services uv run use_server.py
Services loaded from: use_server-services.yml
✓ Service server started
  job: 6aba86d76b030d633f69d0ad
  url: https://huggingface.co/jobs/lhoestq/6aba86d76b030d633f69d0ad
Waiting for 1 service(s) to be running...
All services are running.
Hint: Services group 'services-47c1b9a1fbc6': the Job reaches a service at
      $HF_NETWORK_GROUP_PREFIX<ALIAS>:<PORT>, e.g. http://$HF_NETWORK_GROUP_PREFIXserver:8000.
✓ Job started
  id: 6aba86dd6b030d633f69d0af
  url: https://huggingface.co/jobs/lhoestq/6aba86dd6b030d633f69d0af
SMOKE OK http://nwa-6aba86d765d08b3141560010-server:8000/ -> b'<!DOCTYPE HTML>\n<html lang="en">\n<head>\n'
Cancelled service server (6aba86d76b030d633f69d0ad)
```

(The script is a plain `urllib` GET against `http://$HF_NETWORK_GROUP_PREFIXserver:8000/`, retried until
the server answers. The upload progress bars of `hf jobs uv run` are left out.)

Both files are in [`examples/`](examples/) and run as-is: `cd examples && hf jobs-services uv run
use_server.py`.

Before spending any compute, `--dry-run` prints the plan without starting anything:

```console
$ hf jobs-services uv run --with-services "dask(num_workers=2)" --dry-run my_dask_script.py
✓ Dry run: nothing started
  services_file: dask(num_workers=2)
  services_group: services-8ca62872aa24
  services: 3
  script: my_dask_script.py
  services_timeout: 2h
ALIAS          IMAGE        FLAVOR
dask-scheduler python:3.12   cpu-upgrade
dask-worker-0  python:3.12   cpu-upgrade
dask-worker-1  python:3.12   cpu-upgrade
Hint: The Job reaches the services at $HF_NETWORK_GROUP_PREFIX<ALIAS>:<PORT>, e.g. dask-scheduler
```

## Install

```bash
hf extensions install lhoestq/hf-jobs-services
hf jobs-services --help
```

`hf jobs-services ...` then forwards to the extension, like any built-in `hf` command. The installer
builds an isolated venv with `uv`, so `uv` needs to be on your `PATH`, and re-installing over an existing
extension takes `hf extensions install --force`.

## Usage

```console
$ hf jobs-services --help
Usage: hf jobs-services [OPTIONS] COMMAND [ARGS]...

  Run Jobs alongside background services (Ray, Dask, Spark, custom).

Options:
  --help  Show this message and exit.

Commands:
  ls | list  List the running Jobs started by hf jobs-services.
  stop       Cancel every running Job of a services group.
  templates  List the available services templates.
  uv         Run UV scripts as Jobs, alongside services.
```

`hf jobs-services uv run` mirrors `hf jobs uv run` option for option (`--with`, `-p/--python`, `--image`,
`--flavor`, `--timeout`, `-e/--env`, `-s/--secrets`, `--env-file` / `--secrets-file` with `-` for stdin,
`-v/--volume`, `--expose`, `--ssh`, `-d/--detach`, `--dry-run`, ...) and adds `--with-services` and
`--services-timeout`. The old `hf jobs-services run` still works, but prints a pointer to
`hf jobs-services uv run`.

### One template, one script

```bash
hf jobs-services uv run --with-services "ray(num_workers=4)" my_ray_script.py
hf jobs-services uv run --with-services "dask(num_workers=4)" my_dask_script.py -- --num-rows 1000
hf jobs-services uv run --with-services "spark(num_workers=2)" my_spark_script.py
hf jobs-services uv run --with-services "spark_connect(num_workers=2)" my_connect_script.py
```

`hf jobs-services templates` lists the templates and their parameters. Arguments are typed
(`num_workers=4`, `flavor="cpu-upgrade"`, `disable_usage_stats=false`).

Anything after the script name goes to the script. Pass a `--` separator to make sure a flag of yours is
not read as an option of `hf jobs-services`.

### A services file

Without `--with-services`, the services file is looked up by name: first `<script-name>-services.yml`
next to the script, then `jobs-services.yml` in the current folder.

```yaml
# my_train_script-services.yml
services:
  # reachable from the Job at http://$HF_NETWORK_GROUP_PREFIXserver:8000
  server:
    image: python:3.12
    command: ["python", "-m", "http.server", "8000"]
    env:
      PORT: 8000

  # one Job per replica, aliased cache-0, cache-1, cache-2
  cache:
    image: redis:7
    command: ["redis-server", "--save", ""]
    replicas: 3
    flavor: cpu-upgrade
```

```bash
hf jobs-services uv run my_train_script.py
```

Supported keys per service: `image`, `command`, `env`, `secrets`, `flavor`, `replicas`, `expose`,
`volumes`, `ssh`, `labels`. `command` is either an argv list or a shell line (run through `sh -c`, so
pipes and `${VAR}` work). Services all start in parallel, share one network group, and reach each other
on every port - which is why `depends_on`, `networks`, `ports`, `build` and `restart` are rejected with
an explanation instead of being silently ignored.

### From the shell, without a file

```bash
hf jobs-services uv run --with-services "dask(num_workers=2)" --flavor cpu-upgrade --timeout 30m \
  -e HF_TOKEN -s MY_SECRET=abc -v hf://datasets/trl-lib/Capybara:/data \
  my_dask_script.py --num-rows 1000
```

### Check on them, stop them

```bash
$ hf jobs-services ls
GROUP                 SERVICE STAGE   ID
services-594e22f45cd7 server  RUNNING lhoestq/6aba88996b030d633f69d0d9
Hint: Stop them with: hf jobs-services stop services-594e22f45cd7

$ hf jobs-services stop services-594e22f45cd7
Cancel 1 Job(s) of services group 'services-594e22f45cd7'? [y/N]: y
✓ Job canceled
  service: server
  job: 6aba88996b030d633f69d0d9
✓ Services group stopped
  group: services-594e22f45cd7
  canceled: 1
```

`ls` only shows Jobs still scheduling or running, and `stop` prompts before cancelling (`-y` to skip it).

Every Job started by the extension carries a `services-group` label (and a `service` one), which is how
`ls` and `stop` find them back - the network group name itself is not queryable through the Jobs API.

### Lifecycle

| Situation                               | What happens                                                  |
| --------------------------------------- | ------------------------------------------------------------- |
| a service fails to start                | the other services are canceled, the Job is never started     |
| the Job ends (any stage)                | the services are canceled, the exit code follows the Job stage |
| `Ctrl+C`                                | the Job and its services are canceled                          |
| `--detach`                              | the command returns once the Job is started, services stay up |
| the terminal dies                       | services stop on their own once `--services-timeout` expires   |

`--services-timeout` defaults to `--timeout`, itself defaulting to `2h` for services, so an abandoned
run cannot bill forever.

## How the Job talks to its services

Every member of a Jobs network group is reachable by the others on every port, through two environment
variables set inside each Job:

- `${HF_NETWORK_GROUP_PREFIX}<alias>` - the hosts claiming that alias (`server`, `dask-scheduler`, ...)
- `$HF_NETWORK_GROUP_HOSTNAME` - every member of the group

Members are resolvable **before** they are ready: connect with retries
(`curl --retry 10 --retry-connrefused`, `ray.init(..., retries=...)`) rather than expecting a startup
order. The main Job claims the `main` alias unless a service already took it.

## Templates

| Template                                | Services                                    | From the Job, connect with                                  |
| --------------------------------------- | ------------------------------------------- | ----------------------------------------------------------- |
| `ray(num_workers=2, ...)`               | `ray-head` + `ray-worker-N`                 | `JobSubmissionClient("http://$HF_NETWORK_GROUP_PREFIXray-head:8265")` |
| `dask(num_workers=2, ...)`              | `dask-scheduler` + `dask-worker-N`          | `Client("http://$HF_NETWORK_GROUP_PREFIXdask-scheduler:8786")` |
| `spark(num_workers=2, ...)`             | `spark-master` + `spark-worker-N`           | `spark://$HF_NETWORK_GROUP_PREFIXspark-master:7077` (thrift on 9083) |
| `spark_connect(num_workers=2, ...)`     | `spark-master`, `spark-connect` + workers   | `SparkSession.builder.remote("sc://$HF_NETWORK_GROUP_PREFIXspark-connect:15002")` |

Run `hf jobs-services templates` for the full parameter list of each one. The Ray and Dask templates
install their Python package inside the service Jobs: pass `image=` (with the package pre-installed) to
cut a few seconds off the startup, and `ray_version=` / `dask_version=` to pin a version.

## Limits of this approach

The orchestration is client-side: one `hf jobs-services uv run` process starts the services, waits for them,
starts the Job, streams its logs and cancels the services at the end. Concretely:

- the terminal has to stay alive (or use `--detach`, and stop the services yourself afterwards);
- services are billed from the moment they start, including while the main Job is scheduling;
- a network group only spans Jobs of the same namespace and resource group, so `--resource-group-id` is
  shared between the Job and its services.

[`huggingface/huggingface_hub#4931`](https://github.com/huggingface/huggingface_hub/pull/4931) is the
server-side counterpart of this extension (`run_uv_job(..., with_services=...)`): once it ships, a
single Job API call owns the whole group, and this extension becomes a thin front-end for the
services-file format.

## Development

```bash
uv venv && uv pip install -e . pytest ruff ty
python -m pytest tests                      # 37 tests, no network (the Hub API is faked)
ruff format src tests examples && ruff check src tests examples
ty check --python .venv src
```

To try the `hf jobs-services ...` dispatch with a local checkout, install the package in a venv, then
point an extension manifest at the generated binary:

```bash
mkdir -p ~/.local/share/hf/extensions/hf-jobs-services
cat > ~/.local/share/hf/extensions/hf-jobs-services/manifest.json <<EOF
{
  "owner": "lhoestq",
  "repo": "hf-jobs-services",
  "repo_id": "lhoestq/hf-jobs-services",
  "short_name": "jobs-services",
  "executable_path": "$PWD/.venv/bin/hf-jobs-services",
  "type": "python",
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%S)",
  "description": "Run Jobs alongside long-running services"
}
EOF
hf jobs-services --help
```

`hf extensions install lhoestq/hf-jobs-services` works the same way once the repository is public: the
installer downloads `github.com/lhoestq/hf-jobs-services/archive/HEAD.zip` into an isolated venv under
`~/.local/share/hf/extensions/hf-jobs-services/venv`.
