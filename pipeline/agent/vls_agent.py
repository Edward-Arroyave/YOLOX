"""Vision Label Studio agent: runs the training pipeline for requests queued in Blob.

The web app (Vision Label Studio, spec 027) cannot enter this VM. Instead it
writes a request to Blob Storage (``<VLS_RUNS_CONTAINER>/<VLS_RUNS_PREFIX>/<runId>/request.json``)
and starts the machine if it is off. This agent runs as a systemd service of the
``jupyter`` user, from the root of this repository, and:

* takes the requests oldest first, one at a time (FIFO);
* waits, with status ``waiting``, while someone else's training runs on the machine;
* downloads the chosen base model from Blob and passes it as ``--base-checkpoint``;
* runs ``conda run -n <env> python pipeline/run_training_pipeline.py --prefix … --dataset-folder … --yes-clean``;
* uploads the last 200 log lines and ``progress.json`` (written by the trainer through
  ``VLS_PROGRESS_FILE``) every ``VLS_LOG_UPLOAD_SECONDS``;
* writes the final ``status.json`` with the published version, duration and best metrics;
* shuts the machine down if the app started it for this request and nothing else is pending.

Security: nothing read from Blob is run through a shell. Every field is validated and
the argument list is built here. The work directory must be this repository.

Configuration: the repository ``.env`` (``AZURE_STORAGE_CONNECTION_STRING`` is reused)
plus the optional ``VLS_*`` variables documented in ``pipeline/agent/README.md``.
Run: ``python -m pipeline.agent.vls_agent``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

NAME = re.compile(r"^[A-Za-z0-9._-]+$")
RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
VERSION_LINE = re.compile(r"Nueva versi[oó]n:\s*(\d+\.\d+\.\d+)")
TRAINING_PROCESS = re.compile(r"run_training_pipeline\.py|tools/train\.py")
AGENT_PROCESS = re.compile(r"pipeline\.agent\.vls_agent|vls_agent\.py")
LOG_LINES = 200
PENDING_STATUSES = (None, "waiting")
RUNS_DIR = ".vls_runs"


class InvalidRequest(Exception):
    """A request the agent refuses to run (the run is marked failed)."""


@dataclass(frozen=True)
class AgentConfig:
    container: str
    runs_prefix: str
    repository_root: Path
    conda_exe: str
    poll_seconds: int
    log_upload_seconds: int

    @property
    def runs_dir(self) -> Path:
        return self.repository_root / RUNS_DIR


def load_config(env, repository_root: Path) -> AgentConfig:
    if not env.get("AZURE_STORAGE_CONNECTION_STRING"):
        raise SystemExit("vls-agent: falta AZURE_STORAGE_CONNECTION_STRING en el .env del repositorio.")
    return AgentConfig(
        container=env.get("VLS_RUNS_CONTAINER") or "yolox",
        runs_prefix=(env.get("VLS_RUNS_PREFIX") or "vls-runs").strip("/"),
        repository_root=repository_root,
        conda_exe=env.get("VLS_CONDA_EXE") or "/opt/conda/bin/conda",
        poll_seconds=int(env.get("VLS_POLL_SECONDS") or 20),
        log_upload_seconds=int(env.get("VLS_LOG_UPLOAD_SECONDS") or 30),
    )


def _name(request, field):
    value = request.get(field)
    if not isinstance(value, str) or not NAME.match(value) or value in (".", ".."):
        raise InvalidRequest(f"{field} no válido")
    return value


def _relative(value, field, suffix=None):
    if (not isinstance(value, str) or value.startswith("/") or (suffix and not value.endswith(suffix))
            or not all(NAME.match(part) and part not in (".", "..") for part in value.split("/"))):
        raise InvalidRequest(f"{field} no válido")
    return value


def build_command(request, config: AgentConfig, base_checkpoint_path: Path | None = None):
    """Validated request → (argv, cwd). Never a shell string."""
    prefix = _name(request, "prefix")
    dataset_folder = _name(request, "datasetFolder")
    conda_env = _name(request, "condaEnv")
    script = _relative(request.get("script"), "script")
    if request.get("baseCheckpoint") is not None:
        _relative(request.get("baseCheckpoint"), "baseCheckpoint", ".pth")
    workdir = request.get("workdir")
    if not isinstance(workdir, str) or ".." in PurePosixPath(workdir).parts:
        raise InvalidRequest("workdir no válido")
    if Path(workdir).resolve() != config.repository_root.resolve():
        raise InvalidRequest(f"workdir no es el repositorio del agente ({config.repository_root})")

    argv = [
        config.conda_exe, "run", "-n", conda_env, "--no-capture-output",
        "python", script,
        "--prefix", prefix, "--dataset-folder", dataset_folder, "--yes-clean",
    ]
    if base_checkpoint_path is not None:
        argv += ["--base-checkpoint", str(base_checkpoint_path)]
    return argv, str(config.repository_root)


def _status_of(store, run_id):
    text = store.read(run_id, "status.json")
    if text is None:
        return None
    try:
        return json.loads(text).get("status") or "unknown"
    except (ValueError, AttributeError):
        return "unknown"


def find_pending(store):
    """Requests not finished nor running, oldest first (run ids sort chronologically)."""
    return [run_id for run_id in sorted(store.list_run_ids())
            if RUN_ID.match(run_id) and store.read(run_id, "request.json") is not None
            and _status_of(store, run_id) in PENDING_STATUSES]


def foreign_training(ps_output: str, own_pid: int) -> str | None:
    """`ps -eo pid=,user=,args=` → reason if someone else's training is running."""
    for line in ps_output.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid, user, args = int(parts[0]), parts[1], parts[2]
        if pid == own_pid or AGENT_PROCESS.search(args) or not TRAINING_PROCESS.search(args):
            continue
        return f"Hay otro entrenamiento en la máquina (PID {pid}, usuario {user})."
    return None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _status(run_id, status, **fields):
    return json.dumps({
        "schema": "vls-run-status-v1", "runId": run_id, "status": status,
        "waitingReason": fields.get("waitingReason"), "queuePosition": fields.get("queuePosition"),
        "startedAt": fields.get("startedAt"), "updatedAt": _now(),
        "finishedAt": fields.get("finishedAt"), "exitCode": fields.get("exitCode"),
        "error": fields.get("error"), "result": fields.get("result"),
    }, ensure_ascii=False, indent=2)


def _read_progress(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def process_run(run_id, store, config: AgentConfig, runner, *, clock=time.monotonic, busy=lambda: None,
                sleep=time.sleep, shutdown=None, download=None):
    """Claims and runs one request. Returns "succeeded"/"failed", or None if it was not taken."""
    current = _status_of(store, run_id)
    if current not in PENDING_STATUSES:
        return None
    reason = busy()
    first = _status(run_id, "waiting" if reason else "running", waitingReason=reason,
                    queuePosition=1 if reason else None, startedAt=None if reason else _now())
    # Claim: only one agent creates status.json; a run left "waiting" by this agent is resumed.
    if current is None and not store.create_if_absent(run_id, "status.json", first):
        return None
    if current == "waiting":
        store.write(run_id, "status.json", first)
    while reason:
        sleep(config.poll_seconds)
        reason = busy()
        if reason:
            store.write(run_id, "status.json", _status(run_id, "waiting", waitingReason=reason, queuePosition=1))

    started = _now()
    started_clock = clock()
    store.write(run_id, "status.json", _status(run_id, "running", startedAt=started))
    work = config.runs_dir / run_id
    progress_file = work / "progress.json"
    tail = deque(maxlen=LOG_LINES)
    version = [None]

    def upload_log():
        store.write(run_id, "log.txt", "".join(line + "\n" for line in tail))

    def progress(status, error=None):
        data = _read_progress(progress_file) or {
            "schema": "progress-v1", "epoch": None, "maxEpoch": None, "startedAt": started,
            "training_history": [], "evaluations": [], "best": None}
        data.update(runId=run_id, status=status, updatedAt=_now(), version=version[0], error=error)
        return data

    def finish(status, exit_code=None, error=None):
        final = progress(status, error)
        store.write(run_id, "progress.json", json.dumps(final, ensure_ascii=False))
        upload_log()
        result = {"version": version[0], "durationSeconds": round(clock() - started_clock),
                  "best": final.get("best")}
        store.write(run_id, "status.json", _status(run_id, status, startedAt=started, finishedAt=_now(),
                                                    exitCode=exit_code, error=error, result=result))
        shutil.rmtree(work, ignore_errors=True)
        return status

    try:
        request = json.loads(store.read(run_id, "request.json") or "")
        if not isinstance(request, dict):
            raise InvalidRequest("la solicitud no es un objeto JSON")
        build_command(request, config)  # validate everything before downloading anything
        base_local = None
        if request.get("baseCheckpoint"):
            base_local = work / "base" / PurePosixPath(request["baseCheckpoint"]).name
            try:
                download(request["baseCheckpoint"], base_local)
            except Exception as error:  # noqa: BLE001 — any download problem fails the run
                return finish("failed", error=f"No se pudo descargar el modelo base "
                                              f"{request['baseCheckpoint']}: {error}")
        argv, cwd = build_command(request, config, base_checkpoint_path=base_local)
    except (ValueError, InvalidRequest) as error:
        return finish("failed", error=f"Solicitud rechazada por el agente: {error}")

    last_upload = [clock()]

    def on_line(line):
        line = line.rstrip("\n")
        tail.append(line)
        found = VERSION_LINE.search(line)
        if found:
            version[0] = found.group(1)
        if clock() - last_upload[0] >= config.log_upload_seconds:
            upload_log()
            store.write(run_id, "progress.json", json.dumps(progress("running"), ensure_ascii=False))
            store.write(run_id, "status.json", _status(run_id, "running", startedAt=started))
            last_upload[0] = clock()

    try:
        exit_code = runner(argv, cwd, {"VLS_PROGRESS_FILE": str(progress_file)}, on_line)
    except Exception as error:  # noqa: BLE001 — the pipeline could not even start
        tail.append(f"[vls-agent] no se pudo ejecutar: {error}")
        return finish("failed", error=f"No se pudo ejecutar el pipeline: {error}")

    result = finish("succeeded" if exit_code == 0 else "failed", exit_code=exit_code,
                    error=None if exit_code == 0 else f"El pipeline terminó con código {exit_code}.")

    if request.get("stopWhenDone") is True and shutdown is not None:
        others = find_pending(store)
        someone = busy()
        if others or someone:
            why = f"quedan {len(others)} solicitudes en cola" if others else someone
            tail.append(f"[vls-agent] no se apaga la máquina: {why}")
            upload_log()
        else:
            code, output = shutdown()
            if code != 0:
                tail.append(f"[vls-agent] no se pudo apagar la máquina ({code}): {output}")
                upload_log()
    return result


# --- Real implementations (not used by the unit tests) -----------------------

def run_subprocess(argv, cwd, env_extra, on_line):
    """Runs the pipeline with stdout+stderr merged, line by line. No shell."""
    env = {**os.environ, **env_extra, "PYTHONUNBUFFERED": "1"}
    with subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as proc:
        for line in proc.stdout:
            on_line(line)
        return proc.wait()


def machine_busy():
    done = subprocess.run(["ps", "-eo", "pid=,user=,args="], capture_output=True, text=True)
    return foreign_training(done.stdout, own_pid=os.getpid())


def shutdown_machine():
    """Needs passwordless sudo for `shutdown` (the default on Workbench for `jupyter`)."""
    done = subprocess.run(["sudo", "-n", "shutdown", "-h", "now"], capture_output=True, text=True)
    return done.returncode, (done.stderr or done.stdout).strip()


class BlobStore:
    """The queue in Azure Blob Storage: `<runs_prefix>/<runId>/<file>` in the container."""

    def __init__(self, client, runs_prefix):
        self.client = client
        self.prefix = runs_prefix.rstrip("/") + "/"

    def list_run_ids(self):
        return [item.name[len(self.prefix):].rstrip("/")
                for item in self.client.walk_blobs(name_starts_with=self.prefix, delimiter="/")
                if item.name.endswith("/")]

    def read(self, run_id, name):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            return self.client.download_blob(f"{self.prefix}{run_id}/{name}").readall().decode("utf-8")
        except ResourceNotFoundError:
            return None

    def write(self, run_id, name, text):
        self.client.upload_blob(f"{self.prefix}{run_id}/{name}", text.encode("utf-8"), overwrite=True)

    def create_if_absent(self, run_id, name, text):
        from azure.core.exceptions import ResourceExistsError
        try:
            self.client.upload_blob(f"{self.prefix}{run_id}/{name}", text.encode("utf-8"), overwrite=False)
            return True
        except ResourceExistsError:
            return False

    def download(self, blob_path, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        with partial.open("wb") as output:
            self.client.download_blob(blob_path).readinto(output)
        if partial.stat().st_size == 0:
            partial.unlink()
            raise RuntimeError(f"el archivo está vacío: {blob_path}")
        partial.replace(destination)


def main():
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from tools.ingest_blob_storage import create_container_client, load_environment

    load_environment(None)
    config = load_config(os.environ, repository_root)
    store = BlobStore(create_container_client(os.environ["AZURE_STORAGE_CONNECTION_STRING"], config.container),
                      config.runs_prefix)
    print(f"vls-agent: vigilando {config.container}/{config.runs_prefix} cada {config.poll_seconds}s", flush=True)
    while True:
        try:
            for run_id in find_pending(store)[:1]:  # one at a time, oldest first
                print(f"vls-agent: entrenamiento {run_id}", flush=True)
                result = process_run(run_id, store, config, run_subprocess, busy=machine_busy,
                                     shutdown=shutdown_machine, download=store.download)
                print(f"vls-agent: entrenamiento {run_id} → {result}", flush=True)
        except Exception as error:  # noqa: BLE001 — keep the service alive across Blob errors
            print(f"vls-agent: error: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        time.sleep(config.poll_seconds)


if __name__ == "__main__":
    main()
