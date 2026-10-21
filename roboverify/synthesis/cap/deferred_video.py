"""Persistent, bounded video jobs replay saved states without executing policies."""

import argparse
import fcntl
import hashlib
import inspect
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from .gpu_scheduler import GPUUnavailable, owner_alive, process_identity

SCHEMA = "cap-deferred-video-v1"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_metadata(env):
    """Capture the environment's rendering defaults before creating a GL context."""
    inner = env.unwrapped if hasattr(env, "unwrapped") else env
    cls = type(inner)
    sources = {}
    for base in cls.__mro__:
        if base is object:
            continue
        filename = inspect.getsourcefile(base)
        if filename and Path(filename).is_file():
            sources[base.__module__] = digest(filename)
    signature = inspect.signature(inner.render)
    render_kwargs = {
        name: signature.parameters[name].default
        for name in ("width", "height")
        if name in signature.parameters
        and signature.parameters[name].default is not inspect.Parameter.empty
    }
    model = getattr(inner.sim, "model", None)
    return {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "source_hashes": sources,
        "model_xml_sha256": (
            hashlib.sha256(model.get_xml().encode()).hexdigest() if model else None
        ),
        "num_blocks": inner.num_blocks,
        "render_kwargs": render_kwargs,
        "camera": "environment-default; fixed by source hashes and model XML",
        "visualization": {
            name: getattr(inner, name)
            for name in ("visualize_target", "visualize_mocap")
            if hasattr(inner, name)
        },
        "fps": 20,
    }


def enqueue(directory, queue_root, metadata, *, seed, task, context):
    directory, queue_root = Path(directory).resolve(), Path(queue_root).resolve()
    trajectory = directory / "trajectory.npz"
    job = {
        "schema": SCHEMA,
        "id": uuid.uuid4().hex,
        "created_epoch": time.time(),
        "trajectory": str(trajectory),
        "trajectory_sha256": digest(trajectory),
        "frame_index_sha256": digest(directory / "frame_index.jsonl"),
        "environment": metadata,
        "seed": seed,
        "task": task,
        "context": context,
        "output": str(directory / "video.mp4"),
    }
    path = directory / "render_job.json"
    atomic_json(path, job)
    atomic_json(
        directory / "render_status.json",
        {
            "status": "pending",
            "job_sha256": digest(path),
            "attempts": [],
        },
    )
    atomic_json(queue_root / "render_queue" / f"{job['id']}.json", {"job": str(path)})
    return {
        "status": "pending",
        "mode": "deferred",
        "job": str(path),
        "status_path": str(directory / "render_status.json"),
        "path": job["output"],
    }


def resolved_video(video):
    """Overlay mutable video state without rewriting immutable policy results."""
    if video and video.get("mode") == "deferred" and video.get("status_path"):
        path = Path(video["status_path"])
        if path.exists():
            state = read(path)
            fields = (
                "status",
                "reason",
                "path",
                "frames",
                "fps",
                "video_sha256",
                "queue_seconds",
                "render_seconds",
                "encode_finalize_seconds",
                "elapsed_seconds",
                "ended_epoch",
            )
            return (
                video
                | {k: state[k] for k in fields if k in state}
                | {"render_attempts": len(state.get("attempts", []))}
            )
    return video


def unclaimed_worker_alive(path):
    """Cover coordinator death between spawning a worker and its status update."""
    worker_path = str(Path(path).with_name("render_worker.json")).encode()
    if not Path(worker_path.decode()).exists():
        return False
    for command in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if worker_path in command.read_bytes().split(b"\0") and process_identity(
                int(command.parent.name)
            ):
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return False


def frame_count(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return int(result.stdout.strip())


def render_job(path, *, environment_factory=None, assignment=None):
    """Lock one job; failures are video-only. Retries require an explicit caller."""
    path = Path(path).resolve()
    directory = path.parent
    with (directory / ".render.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(directory / "render_status.json")
        if state["status"] == "completed":
            return state
        started = time.time()
        attempt = {
            "started_epoch": started,
            "owner": process_identity(),
            "gpu_assignment": assignment,
            "GPUS": os.environ.get("GPUS"),
        }
        state.update(status="running", owner=process_identity())
        state["attempts"].append(attempt)
        atomic_json(directory / "render_status.json", state)
        env = encoder = None
        temporary = directory / "video.partial.mp4"
        try:
            import numpy as np
            from synthesis.cfg.collection import VideoRecorder
            from synthesis.cfg.recordings import load_traces
            from synthesis.cfg.reset import inner_env, restore

            from .config import RunConfig
            from .tasks import prepare_environment

            job = read(path)
            if state["job_sha256"] != digest(path) or job["schema"] != SCHEMA:
                raise ValueError("Render job changed or has an incompatible schema")
            if digest(job["trajectory"]) != job["trajectory_sha256"]:
                raise ValueError("Trajectory hash mismatch")
            if digest(directory / "frame_index.jsonl") != job["frame_index_sha256"]:
                raise ValueError("Frame index hash mismatch")
            (trace,) = load_traces(job["trajectory"])
            frames = [
                json.loads(line)
                for line in (directory / "frame_index.jsonl").read_text().splitlines()
            ]
            if len(frames) != len(trace.states) or len(trace.snapshots) != len(
                trace.states
            ):
                raise ValueError("Incomplete snapshot/frame mapping")
            if any(
                row["frame"] != i or row["state_index"] != i or row["step"] != i
                for i, row in enumerate(frames)
            ):
                raise ValueError("Invalid frame/state mapping")
            config = RunConfig(
                task=job["task"],
                num_blocks=trace.num_blocks,
                seed=job["seed"],
                video_mode="off",
            )
            if (
                environment_factory is None
                and job["environment"]["class"]
                == "synthesis.cap.test_adapter.RobotFixture"
            ):
                from .test_adapter import RobotFixture

                env = RobotFixture(trace.num_blocks)
                restore(env, trace.snapshots[0])
            else:
                env = (environment_factory or prepare_environment)(
                    config, trace.snapshots[0]
                )
            inner = inner_env(env)
            if environment_metadata(env) != job["environment"]:
                raise ValueError("Environment/model/render metadata mismatch")

            # Neither a policy nor recorded actions may advance physics during rendering.
            def forbidden_step(*args, **kwargs):
                raise RuntimeError("Deferred rendering must not step the environment")

            env.step = forbidden_step
            inner.step = forbidden_step
            temporary.unlink(missing_ok=True)
            encoder = VideoRecorder(temporary)
            render_seconds = 0.0
            for i, snapshot in enumerate(trace.snapshots):
                before = time.monotonic()
                observation = restore(env, snapshot)
                if not np.allclose(observation, trace.states[i], atol=1e-8, rtol=0):
                    raise ValueError(f"Restored observation mismatch at frame {i}")
                if (
                    job["environment"]["class"]
                    == "synthesis.cap.test_adapter.RobotFixture"
                ):
                    inner.actions = [
                        None
                    ] * i  # Fixture's synthetic frame index, no action replay.
                encoder.append(
                    inner.render(
                        mode="rgb_array", **job["environment"]["render_kwargs"]
                    )
                )
                render_seconds += time.monotonic() - before
                if encoder.error:
                    raise RuntimeError(encoder.error)
            before = time.monotonic()
            video = encoder.close()
            encoder = None
            encode_seconds = time.monotonic() - before
            if video["status"] != "saved" or frame_count(temporary) != len(frames):
                raise ValueError("Video did not decode with the expected frame count")
            output = Path(job["output"])
            if output.exists():
                # A crash may occur after atomic publication but before status commit.
                if digest(output) != digest(temporary):
                    raise ValueError("Refuse to overwrite a different existing video")
                temporary.unlink()
            else:
                temporary.replace(output)
            state.pop("reason", None)
            state.update(
                status="completed",
                path=str(output),
                frames=len(frames),
                fps=20,
                video_sha256=digest(output),
                render_seconds=render_seconds,
                encode_finalize_seconds=encode_seconds,
                queue_seconds=started - job["created_epoch"],
            )
        except Exception as exc:
            state.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        finally:
            if encoder is not None:
                encoder.close()
            if env is not None:
                env.close()
            attempt.update(
                ended_epoch=time.time(),
                status=state["status"],
                reason=state.get("reason"),
            )
            state.update(ended_epoch=time.time(), elapsed_seconds=time.time() - started)
            atomic_json(directory / "render_status.json", state)
        return state


class RenderPool:
    """One queue coordinator; jobs share the experiment GPU reservation ledger."""

    def __init__(
        self, root, *, gpu=None, max_workers=2, timeout=600, retry_failed=False
    ):
        if max_workers < 1 or timeout <= 0:
            raise ValueError("Render worker count and timeout must be positive")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".render-coordinator.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        self.gpu, self.max_workers, self.timeout = gpu, max_workers, timeout
        self.retry_failed = retry_failed
        self.retried = set()
        self.active = {}
        self.started = time.time()
        previous = (
            read(self.root / "video_summary.json")
            if (self.root / "video_summary.json").exists()
            else {}
        )
        self.peak = previous.get("peak_render_workers", 0)
        self.blocked_since = None

    def jobs(self):
        return [
            Path(read(p)["job"])
            for p in sorted((self.root / "render_queue").glob("*.json"))
        ]

    def pending(self):
        result = []
        for path in self.jobs():
            if path in self.active:
                continue
            state = read(path.with_name("render_status.json"))
            if state["status"] in {"pending", "running"} and (
                owner_alive(state.get("owner")) or unclaimed_worker_alive(path)
            ):
                raise RuntimeError(
                    "An earlier render worker is still alive; refuse duplicate dispatch"
                )
            if state["status"] in {"pending", "running"} or (
                self.retry_failed
                and state["status"] == "failed"
                and path not in self.retried
            ):
                result.append(path)
        return result

    def tick(self, available_slots):
        for path, (process, handle, assignment, started) in list(self.active.items()):
            expired = time.monotonic() - started > self.timeout
            if expired and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if process.poll() is not None:
                handle.close()
                if assignment:
                    self.gpu.release(assignment["token"])
                state = read(path.with_name("render_status.json"))
                if state["status"] not in {"completed", "failed"}:
                    state.update(
                        status="failed",
                        reason=(
                            "Render deadline"
                            if expired
                            else f"Render worker exited {process.returncode}"
                        ),
                        ended_epoch=time.time(),
                    )
                    atomic_json(path.with_name("render_status.json"), state)
                del self.active[path]
        for path in self.pending():
            if len(self.active) >= min(self.max_workers, available_slots):
                break
            worker_path = path.with_name("render_worker.json")
            try:
                assignment = self.gpu.reserve(worker_path) if self.gpu else None
            except GPUUnavailable:
                self.blocked_since = self.blocked_since or time.monotonic()
                break
            if self.gpu and assignment is None:
                self.blocked_since = self.blocked_since or time.monotonic()
                break
            self.blocked_since = None
            atomic_json(worker_path, {"job": str(path), "gpu_assignment": assignment})
            handle = path.with_name("render_worker.log").open("ab")
            process = None
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "synthesis.cap.deferred_video",
                        "--worker",
                        str(worker_path),
                    ],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=(
                        dict(os.environ, GPUS=str(assignment["egl_index"]))
                        if assignment
                        else None
                    ),
                )
                if assignment:
                    self.gpu.bind(assignment["token"], process.pid)
            except BaseException as error:
                if process is not None:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                handle.close()
                if assignment:
                    self.gpu.release(assignment["token"])
                if not isinstance(error, Exception):
                    raise
                state = read(path.with_name("render_status.json"))
                reason = f"Render worker launch failed: {type(error).__name__}: {error}"
                state.update(status="failed", reason=reason, ended_epoch=time.time())
                state["attempts"].append(
                    {
                        "status": "failed",
                        "reason": reason,
                        "ended_epoch": time.time(),
                        "gpu_assignment": assignment,
                    }
                )
                atomic_json(path.with_name("render_status.json"), state)
                self.retried.add(path)
                continue
            self.retried.add(path)
            self.active[path] = (process, handle, assignment, time.monotonic())
            self.peak = max(self.peak, len(self.active))
        self.report()

    def report(self):
        jobs = [read(p.with_name("render_status.json")) for p in self.jobs()]
        counts = dict(Counter(j["status"] for j in jobs))
        result = {
            "jobs": len(jobs),
            "counts": counts,
            "peak_render_workers": self.peak,
            "all_videos_completed": bool(jobs)
            and counts.get("completed", 0) == len(jobs),
            "coordinator_elapsed_seconds": time.time() - self.started,
            "first_job_created_epoch": min(
                (read(p)["created_epoch"] for p in self.jobs()), default=None
            ),
            "last_video_completed_epoch": max(
                (j.get("ended_epoch", 0) for j in jobs if j["status"] == "completed"),
                default=None,
            ),
            "render_seconds": sum(j.get("render_seconds", 0) for j in jobs),
            "encode_finalize_seconds": sum(
                j.get("encode_finalize_seconds", 0) for j in jobs
            ),
        }
        atomic_json(self.root / "video_summary.json", result)
        return result

    def close(self):
        for path, (process, handle, assignment, _) in self.active.items():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            handle.close()
            if assignment:
                self.gpu.release(assignment["token"])
            state = read(path.with_name("render_status.json"))
            if state["status"] != "completed":
                state.update(
                    status="pending", reason="Coordinator interrupted; resumable"
                )
                atomic_json(path.with_name("render_status.json"), state)
        self.active.clear()
        self.report()
        self.lock.close()


def drain_queue(root, *, max_workers=2, gpu=None, retry_failed=False, timeout=600):
    pool = RenderPool(
        root,
        gpu=gpu,
        max_workers=max_workers,
        timeout=timeout,
        retry_failed=retry_failed,
    )
    try:
        while pool.pending() or pool.active:
            pool.tick(max_workers)
            if (
                pool.blocked_since
                and not pool.active
                and time.monotonic() - pool.blocked_since
                >= gpu.config.admission_timeout_seconds
            ):
                break
            if pool.pending() or pool.active:
                time.sleep(0.2)
        return pool.report()
    finally:
        pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.worker:
        spec = read(args.worker)
        if spec["gpu_assignment"]:
            from .gpu_scheduler import initialize_worker

            initialize_worker(spec["gpu_assignment"], args.worker)
        state = render_job(spec["job"], assignment=spec["gpu_assignment"])
        return 0 if state["status"] == "completed" else 1
    if not args.directory:
        parser.error("--directory is required")
    manifest_path = Path(args.directory) / "manifest.json"
    gpu = None
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest.get("gpu_config"):
            from .gpu_scheduler import GPUScheduler

            gpu = GPUScheduler(manifest["gpu_config"])
        args.max_workers = min(args.max_workers, manifest["max_workers"])
    result = drain_queue(
        args.directory,
        max_workers=args.max_workers,
        gpu=gpu,
        retry_failed=args.retry_failed,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2))
    return 0 if not result["jobs"] or result["all_videos_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
