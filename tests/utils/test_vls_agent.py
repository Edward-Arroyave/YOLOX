"""Vision Label Studio agent: runs the training pipeline for requests queued in Blob.

No Azure, conda or GPU: the store, runner, download, busy check and clock are fakes.
"""
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.agent import vls_agent as agent

RUN_ID = "20261002-010612-c05e"


class MemoryStore:
    """The queue in Blob, in memory: keys are '<runId>/<file>'."""

    def __init__(self):
        self.files = {}
        self.history = []

    def list_run_ids(self):
        return sorted({key.split("/")[0] for key in self.files})

    def read(self, run_id, name):
        return self.files.get(f"{run_id}/{name}")

    def write(self, run_id, name, text):
        self.files[f"{run_id}/{name}"] = text
        self.history.append((name, text))

    def create_if_absent(self, run_id, name, text):
        if f"{run_id}/{name}" in self.files:
            return False
        self.write(run_id, name, text)
        return True


class FakeClock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


class AgentTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        self.config = agent.AgentConfig(
            container="yolox", runs_prefix="vls-runs", repository_root=self.root,
            conda_exe="/opt/conda/bin/conda", poll_seconds=20, log_upload_seconds=30,
        )
        self.request = {
            "schema": "vls-run-request-v1", "runId": RUN_ID,
            "dataset": "rapitestlis/training_yolox/9-2026-segundo-entrenamiento",
            "datasetFolder": "9-2026-segundo-entrenamiento", "prefix": "lis",
            "workdir": str(self.root), "condaEnv": "ia-yolox-training",
            "script": "pipeline/run_training_pipeline.py", "baseCheckpoint": None,
            "stopWhenDone": False,
        }
        self.store = MemoryStore()
        self.clock = FakeClock()
        self.runs = []
        self.downloads = []
        self.shutdowns = []
        self.sleeps = []
        self.busy_reasons = []

    def tearDown(self):
        self.directory.cleanup()

    def queue(self, run_id=RUN_ID, **changes):
        self.store.write(run_id, "request.json", json.dumps({**self.request, "runId": run_id, **changes}))
        self.store.history.clear()
        return run_id

    def runner(self, lines, code, epochs=0):
        """Prints `lines`; after each one writes a progress file like the trainer does."""
        def run(argv, cwd, env_extra, on_line):
            self.runs.append((argv, cwd, dict(env_extra)))
            progress = Path(env_extra["VLS_PROGRESS_FILE"])
            for index, line in enumerate(lines):
                self.clock.now += 20
                if index < epochs:
                    progress.parent.mkdir(parents=True, exist_ok=True)
                    best = {"epoch": index + 1, "map_50_95": 0.4 + index / 100, "ap50": 0.7,
                            "ap75": 0.45, "average_recall": 0.5}
                    progress.write_text(json.dumps({
                        "schema": "progress-v1", "status": "running", "epoch": index + 1, "maxEpoch": 80,
                        "training_history": [{"epoch": index + 1}], "evaluations": [best], "best": best,
                        "version": None, "error": None}), encoding="utf-8")
                on_line(line)
            return code
        return run

    def busy(self):
        return self.busy_reasons.pop(0) if self.busy_reasons else None

    def download(self, blob_path, destination):
        self.downloads.append((blob_path, Path(destination)))
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"ckpt")

    def process(self, run_id, runner, download=None):
        return agent.process_run(
            run_id, self.store, self.config, runner, clock=self.clock, busy=self.busy,
            sleep=self.sleeps.append, shutdown=lambda: self.shutdowns.append(True) or (0, ""),
            download=download or self.download)

    def status(self, run_id=RUN_ID):
        return json.loads(self.store.files[f"{run_id}/status.json"])

    def statuses(self):
        return [json.loads(text) for name, text in self.store.history if name == "status.json"]


class BuildCommandTest(AgentTestCase):
    def test_exact_argv_in_the_repository(self):
        argv, cwd = agent.build_command(self.request, self.config)
        self.assertEqual(argv, [
            "/opt/conda/bin/conda", "run", "-n", "ia-yolox-training", "--no-capture-output",
            "python", "pipeline/run_training_pipeline.py",
            "--prefix", "lis", "--dataset-folder", "9-2026-segundo-entrenamiento", "--yes-clean",
        ])
        self.assertEqual(Path(cwd), self.root)

    def test_the_base_checkpoint_is_the_downloaded_local_file(self):
        local = self.root / ".vls_runs" / RUN_ID / "base" / "yolox_s.pth"
        argv, _ = agent.build_command({**self.request, "baseCheckpoint": "weights/base/yolox_s.pth"},
                                      self.config, base_checkpoint_path=local)
        self.assertEqual(argv[-2:], ["--base-checkpoint", str(local)])

    def test_rejects_unsafe_requests(self):
        for change in [
            {"datasetFolder": "x; rm -rf ~"}, {"datasetFolder": ".."}, {"prefix": "lis --other"},
            {"condaEnv": "env && curl"}, {"script": "../evil.py"}, {"script": "/usr/bin/evil.py"},
            {"workdir": "/etc"}, {"workdir": str(self.root / "otra")}, {"datasetFolder": 42},
            {"baseCheckpoint": "../x.pth"}, {"baseCheckpoint": "weights/notas.txt"},
            {"baseCheckpoint": "/abs/x.pth"}, {"baseCheckpoint": "w/x.pth --yes"},
        ]:
            with self.subTest(change=change):
                with self.assertRaises(agent.InvalidRequest):
                    agent.build_command({**self.request, **change}, self.config)


class ProcessRunTest(AgentTestCase):
    def test_success_runs_once_publishes_log_progress_and_result(self):
        self.queue()
        lines = ["Pipeline configurado:", "  Nueva versión: 1.0.3", "época 1", "época 2", "época 3"]
        result = self.process(RUN_ID, self.runner(lines, 0, epochs=3))
        self.assertEqual(result, "succeeded")
        self.assertEqual(len(self.runs), 1)
        argv, cwd, env_extra = self.runs[0]
        progress_file = Path(env_extra["VLS_PROGRESS_FILE"])
        self.assertTrue(str(progress_file).startswith(str(self.root / ".vls_runs" / RUN_ID)))
        self.assertEqual(self.statuses()[0]["status"], "running")
        final = self.status()
        self.assertEqual((final["status"], final["exitCode"], final["error"]), ("succeeded", 0, None))
        self.assertEqual(final["result"]["version"], "1.0.3")
        self.assertEqual(final["result"]["durationSeconds"], 100)
        self.assertAlmostEqual(final["result"]["best"]["map_50_95"], 0.42)
        self.assertEqual(self.store.files[f"{RUN_ID}/log.txt"].splitlines()[-1], "época 3")
        # live: progress.json was uploaded while running, not only at the end
        uploads = [json.loads(text) for name, text in self.store.history if name == "progress.json"]
        self.assertGreaterEqual(len(uploads), 2)
        self.assertEqual(uploads[0]["status"], "running")
        self.assertEqual((uploads[-1]["status"], uploads[-1]["version"]), ("succeeded", "1.0.3"))
        self.assertFalse((self.root / ".vls_runs" / RUN_ID).exists())  # local files cleaned

    def test_failure_keeps_the_exit_code_and_a_readable_error(self):
        self.queue()
        self.process(RUN_ID, self.runner(["ERROR: una etapa falló con código 1"], 1))
        final = self.status()
        self.assertEqual((final["status"], final["exitCode"]), ("failed", 1))
        self.assertIn("código 1", final["error"])
        progress = json.loads(self.store.files[f"{RUN_ID}/progress.json"])
        self.assertEqual(progress["status"], "failed")

    def test_waits_while_another_training_runs_on_the_machine(self):
        self.queue()
        self.busy_reasons = ["Hay otro entrenamiento en la máquina (PID 1234, usuario pepe).",
                             "Hay otro entrenamiento en la máquina (PID 1234, usuario pepe)."]
        self.process(RUN_ID, self.runner(["ok"], 0))
        first = self.statuses()[0]
        self.assertEqual(first["status"], "waiting")
        self.assertIn("PID 1234", first["waitingReason"])
        self.assertEqual(first["queuePosition"], 1)
        self.assertEqual(self.sleeps, [20, 20])
        self.assertEqual(len(self.runs), 1)
        self.assertEqual(self.status()["status"], "succeeded")
        self.assertIsNone(self.status()["waitingReason"])

    def test_downloads_the_chosen_base_model_from_blob(self):
        self.queue(baseCheckpoint="weights/base/yolox_s.pth")
        self.process(RUN_ID, self.runner(["ok"], 0))
        blob_path, local = self.downloads[0]
        self.assertEqual(blob_path, "weights/base/yolox_s.pth")
        self.assertEqual(local.name, "yolox_s.pth")
        self.assertEqual(self.runs[0][0][-2:], ["--base-checkpoint", str(local)])

    def test_a_failed_download_fails_the_run_without_training(self):
        self.queue(baseCheckpoint="weights/base/yolox_s.pth")

        def broken(blob_path, destination):
            raise OSError("BlobNotFound")

        self.process(RUN_ID, self.runner(["ok"], 0), download=broken)
        self.assertEqual(self.runs, [])
        self.assertIn("modelo base", self.status()["error"])

    def test_invalid_or_malformed_requests_fail_without_running(self):
        self.queue(datasetFolder="x; rm -rf ~")
        self.process(RUN_ID, self.runner([], 0))
        self.assertIn("datasetFolder", self.status()["error"])
        other = "20261002-020000-aaaa"
        self.store.write(other, "request.json", "{ no json")
        self.process(other, self.runner([], 0))
        self.assertEqual(self.status(other)["status"], "failed")
        self.assertEqual(self.runs, [])

    def test_does_not_take_a_run_that_already_has_a_status(self):
        self.queue()
        self.store.write(RUN_ID, "status.json", json.dumps({"status": "running"}))
        self.assertIsNone(self.process(RUN_ID, self.runner(["x"], 0)))
        self.assertEqual(self.runs, [])

    def test_a_runner_crash_is_a_failed_run(self):
        self.queue()

        def boom(argv, cwd, env_extra, on_line):
            raise FileNotFoundError("conda")

        self.process(RUN_ID, boom)
        self.assertEqual(self.status()["status"], "failed")
        self.assertIn("conda", self.status()["error"])

    def test_log_keeps_only_the_last_200_lines(self):
        self.queue()
        self.process(RUN_ID, self.runner([f"línea {i}" for i in range(250)], 0))
        lines = self.store.files[f"{RUN_ID}/log.txt"].splitlines()
        self.assertEqual((len(lines), lines[0]), (200, "línea 50"))


class ShutdownTest(AgentTestCase):
    def test_stops_the_machine_when_asked_and_nothing_else_is_pending(self):
        self.queue(stopWhenDone=True)
        self.process(RUN_ID, self.runner(["ok"], 0))
        self.assertEqual(self.shutdowns, [True])
        self.assertEqual(self.status()["status"], "succeeded")  # final status before shutting down

    def test_does_not_stop_with_more_requests_in_the_queue(self):
        self.queue(stopWhenDone=True)
        self.store.write("20261002-030000-bbbb", "request.json", json.dumps(self.request))
        self.process(RUN_ID, self.runner(["ok"], 0))
        self.assertEqual(self.shutdowns, [])
        self.assertIn("no se apaga", self.store.files[f"{RUN_ID}/log.txt"])

    def test_does_not_stop_if_another_training_started_meanwhile(self):
        self.queue(stopWhenDone=True)
        runner = self.runner(["ok"], 0)

        def run_then_someone_starts(argv, cwd, env_extra, on_line):
            code = runner(argv, cwd, env_extra, on_line)
            self.busy_reasons = ["Hay otro entrenamiento en la máquina (PID 9, usuario pepe)."]
            return code

        self.process(RUN_ID, run_then_someone_starts)
        self.assertEqual(self.shutdowns, [])


class QueueTest(AgentTestCase):
    def test_pending_requests_are_fifo_and_include_the_waiting_one(self):
        for run_id in ("20261002-030000-cccc", "20261002-010000-aaaa", "20261002-020000-bbbb",
                       "20261002-040000-dddd"):
            self.store.write(run_id, "request.json", "{}")
        self.store.write("20261002-020000-bbbb", "status.json", json.dumps({"status": "succeeded"}))
        self.store.write("20261002-040000-dddd", "status.json", json.dumps({"status": "waiting"}))
        self.store.write("no-es-un-id", "request.json", "{}")
        self.assertEqual(agent.find_pending(self.store),
                         ["20261002-010000-aaaa", "20261002-030000-cccc", "20261002-040000-dddd"])


class ForeignTrainingTest(unittest.TestCase):
    def test_detects_a_pipeline_or_train_started_by_someone_else(self):
        ps = "\n".join([
            "  101 jupyter /opt/conda/bin/python -m pipeline.agent.vls_agent",
            "  202 jupyter /opt/conda/bin/python -m ipykernel_launcher -f kernel.json",
            " 1234 pepe    python pipeline/run_training_pipeline.py --prefix lis --yes-clean",
        ])
        self.assertEqual(agent.foreign_training(ps, own_pid=101),
                         "Hay otro entrenamiento en la máquina (PID 1234, usuario pepe).")
        self.assertIsNone(agent.foreign_training(ps.splitlines()[0] + "\n" + ps.splitlines()[1], own_pid=101))
        self.assertIn("PID 77", agent.foreign_training(" 77 root python tools/train.py -f exps/x.py", own_pid=1))


class ConfigTest(unittest.TestCase):
    def test_reuses_the_repository_env_with_defaults(self):
        root = Path("/repo")
        config = agent.load_config({"AZURE_STORAGE_CONNECTION_STRING": "x"}, root)
        self.assertEqual((config.container, config.runs_prefix), ("yolox", "vls-runs"))
        self.assertEqual((config.conda_exe, config.poll_seconds, config.log_upload_seconds),
                         ("/opt/conda/bin/conda", 20, 30))
        self.assertEqual(config.repository_root, root)
        custom = agent.load_config({"AZURE_STORAGE_CONNECTION_STRING": "x", "VLS_RUNS_CONTAINER": "otro",
                                    "VLS_RUNS_PREFIX": "cola", "VLS_POLL_SECONDS": "5"}, root)
        self.assertEqual((custom.container, custom.runs_prefix, custom.poll_seconds), ("otro", "cola", 5))
        with self.assertRaises(SystemExit):
            agent.load_config({}, root)


if __name__ == "__main__":
    unittest.main()
