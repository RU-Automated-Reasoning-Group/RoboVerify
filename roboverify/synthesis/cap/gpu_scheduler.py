"""Host-local, cooperative GPU admission. No simulator imports in the parent."""

import ctypes as C
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .config import GPUConfig


class GPUUnavailable(RuntimeError):
    pass


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_identity(pid=None):
    """Linux start ticks + boot identity defeat PID reuse; zombies have exited."""
    pid = os.getpid() if pid is None else pid
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": pid,
            "start_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def owner_alive(owner):
    if not owner:
        return False
    try:
        return process_identity(owner["pid"]) == owner
    except PermissionError:
        return True  # Inaccessible is not confirmed dead.


def egl_uuid_mapping():
    """Match EGL persistent device UUIDs, not CUDA ordinals, to NVML UUIDs.

    The installed mujoco_py shim enumerates at most eight devices. Use that
    same prefix. Unsupported/ambiguous mappings fail closed.
    """
    lib = C.CDLL("libEGL.so.1")
    lib.eglGetProcAddress.argtypes = [C.c_char_p]
    lib.eglGetProcAddress.restype = C.c_void_p

    def function(name, result, *arguments):
        address = lib.eglGetProcAddress(name.encode())
        if not address:
            raise GPUUnavailable(f"Missing EGL function: {name}")
        return C.CFUNCTYPE(result, *arguments)(address)

    query = function(
        "eglQueryDevicesEXT",
        C.c_uint,
        C.c_int,
        C.POINTER(C.c_void_p),
        C.POINTER(C.c_int),
    )
    binary = function(
        "eglQueryDeviceBinaryEXT",
        C.c_uint,
        C.c_void_p,
        C.c_int,
        C.c_int,
        C.c_void_p,
        C.POINTER(C.c_int),
    )
    devices, count = (C.c_void_p * 8)(), C.c_int()
    if not query(8, devices, C.byref(count)):
        raise GPUUnavailable("EGL device enumeration failed")
    mapping = {}
    for index in range(min(count.value, 8)):
        value, size = (C.c_ubyte * 16)(), C.c_int()
        if (
            binary(devices[index], 0x335C, 16, value, C.byref(size))
            and size.value == 16
        ):
            ident = "GPU-" + str(uuid.UUID(bytes=bytes(value)))
            if ident in mapping:
                raise GPUUnavailable(f"Ambiguous EGL UUID: {ident}")
            mapping[ident] = {
                "uuid": ident,
                "egl_index": index,
                "mapping_method": "EGL_EXT_device_persistent_id",
            }
    return mapping


def probe_mapping():
    """Keep even EGL enumeration out of the scheduler process."""
    result = subprocess.run(
        [sys.executable, "-m", "synthesis.cap.gpu_scheduler", "--mapping"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


class NVMLMonitor:
    """Use the installed driver directly; no Python dependency installation."""

    def sample(self, allowlist):
        try:
            lib = C.CDLL("libnvidia-ml.so.1")
        except OSError as exc:
            raise GPUUnavailable(f"NVML unavailable: {exc}") from exc

        def check(code):
            if code:
                raise GPUUnavailable(f"NVML error {code}")

        class Memory(C.Structure):
            _fields_ = [(key, C.c_ulonglong) for key in ("total", "free", "used")]

        class Utilization(C.Structure):
            _fields_ = [(key, C.c_uint) for key in ("gpu", "memory")]

        class PCI(C.Structure):
            _fields_ = (
                [("legacy", C.c_char * 16)]
                + [
                    (key, C.c_uint)
                    for key in ("domain", "bus", "device", "device_id", "subsystem_id")
                ]
                + [("bus_id", C.c_char * 32)]
            )

        class Process(C.Structure):
            _fields_ = [
                ("pid", C.c_uint),
                ("used_memory", C.c_ulonglong),
                ("gpu_instance", C.c_uint),
                ("compute_instance", C.c_uint),
            ]

        def processes(handle):
            result = {}
            for kind in ("Compute", "Graphics"):
                try:
                    query = getattr(lib, f"nvmlDeviceGet{kind}RunningProcesses_v2")
                    count = C.c_uint(1024)
                    values = (Process * count.value)()
                    check(query(handle, C.byref(count), values))
                    result[kind.lower()] = [
                        {
                            "pid": v.pid,
                            "used_memory_mb": (
                                v.used_memory / 2**20
                                if v.used_memory != 2**64 - 1
                                else None
                            ),
                        }
                        for v in values[: count.value]
                    ]
                except (GPUUnavailable, AttributeError) as exc:
                    result[kind.lower()] = {"unavailable": str(exc)}
            return result

        check(lib.nvmlInit_v2())
        rows = {}
        try:
            for ident in allowlist:
                try:
                    handle = C.c_void_p()
                    check(
                        lib.nvmlDeviceGetHandleByUUID(ident.encode(), C.byref(handle))
                    )
                    memory, utilization, pci, index = (
                        Memory(),
                        Utilization(),
                        PCI(),
                        C.c_uint(),
                    )
                    check(lib.nvmlDeviceGetMemoryInfo(handle, C.byref(memory)))
                    check(
                        lib.nvmlDeviceGetUtilizationRates(handle, C.byref(utilization))
                    )
                    check(lib.nvmlDeviceGetPciInfo_v3(handle, C.byref(pci)))
                    check(lib.nvmlDeviceGetIndex(handle, C.byref(index)))
                    rows[ident] = {
                        "uuid": ident,
                        "monitor_index": index.value,
                        "pci_bus_id": pci.bus_id.decode(),
                        "free_memory_mb": memory.free / 2**20,
                        "used_memory_mb": memory.used / 2**20,
                        "utilization_percent": utilization.gpu,
                        "processes": processes(handle),
                        "healthy": True,
                        "health_scope": "NVML memory/utilization/PCI queries succeeded",
                    }
                except GPUUnavailable as exc:
                    rows[ident] = {"uuid": ident, "healthy": False, "error": str(exc)}
        finally:
            lib.nvmlShutdown()
        return rows


class GPUScheduler:
    def __init__(self, config, *, monitor=None, mapping=None):
        self.config = config if isinstance(config, GPUConfig) else GPUConfig(**config)
        self.directory = Path(self.config.reservation_directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor or NVMLMonitor()
        self.mapping = probe_mapping() if mapping is None else mapping
        missing = set(self.config.gpu_allowlist) - set(self.mapping)
        if missing:
            raise GPUUnavailable(f"No verified EGL mapping for {sorted(missing)}")
        self.last_sample = None
        self.last_reason = None
        self.telemetry = {}
        self.worker_telemetry = {}
        self.latest_sample = None

    @contextmanager
    def locked(self):
        with (self.directory / "scheduler.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.directory / "reservations.json"
            data = (
                json.loads(path.read_text())
                if path.exists()
                else {
                    "host": socket.gethostname(),
                    "reservations": {},
                    "samples": [],
                }
            )
            if data["host"] != socket.gethostname():
                raise GPUUnavailable(
                    "Reservation directory belongs to a different host"
                )
            yield data
            atomic_json(path, data)

    def _reclaim(self, data):
        for token, reservation in list(data["reservations"].items()):
            if owner_alive(reservation.get("worker")):
                continue
            if reservation.get("worker") or not owner_alive(reservation.get("parent")):
                # A parent can die between spawn and recording its child's PID.
                # Such a child claims the token before importing/using the GPU.
                if not reservation.get("worker"):
                    found = False
                    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
                        try:
                            if reservation[
                                "job_path"
                            ].encode() in cmdline.read_bytes().split(b"\0"):
                                found = True
                                break
                        except (FileNotFoundError, ProcessLookupError):
                            pass
                        except PermissionError:
                            found = True  # Fail closed when exit cannot be confirmed.
                            break
                    if found:
                        continue
                del data["reservations"][token]

    def _sample(self, data):
        now = time.time()
        if (
            self.last_sample is None
            or now - self.last_sample >= self.config.gpu_poll_interval_seconds
        ):
            try:
                rows = self.monitor.sample(self.config.gpu_allowlist)
            except (GPUUnavailable, OSError, AttributeError) as exc:
                if self.config.gpu_policy != "fixed":
                    raise GPUUnavailable(f"GPU monitoring unavailable: {exc}") from exc
                rows = {
                    u: {"uuid": u, "monitoring_unavailable": str(exc)}
                    for u in self.config.gpu_allowlist
                }
            data["samples"].append({"epoch": now, "devices": rows})
            self.last_sample = now
        data["retention_seconds"] = max(
            data.get("retention_seconds", 0), self.config.gpu_sample_window_seconds
        )
        latest = data.setdefault("latest_devices", {})
        for sample in data["samples"]:
            latest.update(sample["devices"])
        cutoff = now - data["retention_seconds"]
        data["samples"] = [s for s in data["samples"] if s["epoch"] >= cutoff]
        for sample in data["samples"]:
            for ident, row in sample["devices"].items():
                if ident not in self.config.gpu_allowlist:
                    continue
                stats = self.telemetry.setdefault(ident, {})
                for key in ("used_memory_mb", "utilization_percent"):
                    if key in row:
                        stats["peak_" + key] = max(
                            stats.get("peak_" + key, 0), row[key]
                        )
                stats["latest"] = row
        self.latest_sample = {
            "epoch": self.last_sample,
            "devices": {u: latest[u] for u in self.config.gpu_allowlist if u in latest},
        }
        return latest

    def poll(self):
        with self.locked() as data:
            self._reclaim(data)
            rows = self._sample(data)
            for reservation in data["reservations"].values():
                owner = reservation.get("worker")
                if not owner or reservation["uuid"] not in self.config.gpu_allowlist:
                    continue
                stats = self.worker_telemetry.setdefault(reservation["job_path"], {})
                try:
                    fields = (
                        Path(f"/proc/{owner['pid']}/stat")
                        .read_text()
                        .rsplit(")", 1)[1]
                        .split()
                    )
                    stats["peak_rss_mb"] = max(
                        stats.get("peak_rss_mb", 0),
                        int(fields[21]) * os.sysconf("SC_PAGE_SIZE") / 2**20,
                    )
                    stats["cpu_seconds"] = (
                        int(fields[11]) + int(fields[12])
                    ) / os.sysconf("SC_CLK_TCK")
                    io = dict(
                        line.split(": ", 1)
                        for line in Path(f"/proc/{owner['pid']}/io")
                        .read_text()
                        .splitlines()
                    )
                    stats["io_bytes"] = {
                        k: int(io[k]) for k in ("read_bytes", "write_bytes")
                    }
                except (FileNotFoundError, ProcessLookupError, PermissionError) as exc:
                    stats["process_sample_unavailable"] = type(exc).__name__
                processes = rows.get(reservation["uuid"], {}).get("processes", {})
                memory = [
                    p["used_memory_mb"]
                    for group in processes.values()
                    if isinstance(group, list)
                    for p in group
                    if p["pid"] == owner["pid"] and p["used_memory_mb"] is not None
                ]
                if memory:
                    stats["peak_gpu_memory_mb"] = max(
                        stats.get("peak_gpu_memory_mb", 0), *memory
                    )
                stats["sampled_epoch"] = time.time()

    def reserve(self, job_path):
        job_path = str(Path(job_path).resolve())
        with self.locked() as data:
            self._reclaim(data)
            rows = self._sample(data)
            if any(r["job_path"] == job_path for r in data["reservations"].values()):
                raise GPUUnavailable("Trial already has a live GPU reservation")
            candidates = []
            for ident in self.config.gpu_allowlist:
                row = rows.get(ident, {})
                reservations = [
                    r for r in data["reservations"].values() if r["uuid"] == ident
                ]
                limit = min(
                    [self.config.max_workers_per_gpu]
                    + [r["slot_limit"] for r in reservations]
                )
                if len(reservations) >= limit:
                    continue
                samples = [
                    s["devices"][ident]
                    for s in data["samples"]
                    if ident in s["devices"]
                    and s["epoch"]
                    >= time.time() - self.config.gpu_sample_window_seconds
                ]
                samples = samples or [row]
                if (
                    row.get("monitoring_unavailable")
                    and self.config.gpu_policy == "fixed"
                ):
                    mean_load, free = 0, None
                elif row.get("healthy"):
                    valid = [s for s in samples if s.get("healthy")]
                    free = min(s["free_memory_mb"] for s in valid)
                    # Conservative: reserve each entire worker allowance even if NVML
                    # already sees part of it. Never spend the same headroom twice.
                    reserved = sum(r["memory_mb"] for r in reservations)
                    if (
                        free - reserved - self.config.worker_memory_mb
                        < self.config.min_free_memory_mb
                    ):
                        continue
                    mean_load = sum(s["utilization_percent"] for s in valid) / len(
                        valid
                    )
                else:
                    continue
                candidates.append(
                    (len(reservations), mean_load, -(free or 0), ident, row)
                )
            if not candidates:
                self.last_reason = "No allowed GPU has a healthy sample, memory headroom, and free worker slot"
                return None
            _, _, _, ident, row = min(candidates)
            token = uuid.uuid4().hex
            occupied = {
                r["slot"] for r in data["reservations"].values() if r["uuid"] == ident
            }
            reservation = self.mapping[ident] | {
                "token": token,
                "slot": next(
                    i
                    for i in range(self.config.max_workers_per_gpu)
                    if i not in occupied
                ),
                "slot_limit": self.config.max_workers_per_gpu,
                "memory_mb": self.config.worker_memory_mb,
                "parent": process_identity(),
                "worker": None,
                "state": "reserved",
                "job_path": job_path,
                "reserved_epoch": time.time(),
                "admission_sample": row,
                "reservation_directory": str(self.directory),
                "config": asdict(self.config),
                "environment": {
                    "GPUS": str(self.mapping[ident]["egl_index"]),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
            }
            data["reservations"][token] = reservation
            self.last_reason = None
            return reservation

    def bind(self, token, pid):
        with self.locked() as data:
            reservation = data["reservations"][token]
            worker = process_identity(pid)
            if worker:
                reservation.update(worker=worker, state="running")

    def release(self, token):
        with self.locked() as data:
            row = data["reservations"].get(token)
            if row and owner_alive(row.get("worker")):
                raise GPUUnavailable("Cannot release a live worker's rendering slot")
            data["reservations"].pop(token, None)


def initialize_worker(assignment, job_path):
    """Claim before any simulator import; reverify the actual worker environment."""
    if os.environ.get("GPUS") != str(assignment["egl_index"]):
        raise GPUUnavailable("Worker GPUS does not match its reservation")
    scheduler = GPUScheduler(assignment["config"], mapping=egl_uuid_mapping())
    if scheduler.mapping[assignment["uuid"]]["egl_index"] != assignment["egl_index"]:
        raise GPUUnavailable(
            "EGL mapping changed between admission and worker initialization"
        )
    with scheduler.locked() as data:
        row = data["reservations"].get(assignment["token"])
        if not row or row["job_path"] != str(Path(job_path).resolve()):
            raise GPUUnavailable(
                "Worker reservation missing or belongs to another trial"
            )
        if row.get("worker") and row["worker"] != process_identity():
            raise GPUUnavailable("Reservation is owned by another process")
        row.update(worker=process_identity(), state="running")
    return {
        "mapping_verified": True,
        "identity": process_identity(),
        "uuid": assignment["uuid"],
        "egl_index": assignment["egl_index"],
        "environment": {
            "GPUS": os.environ.get("GPUS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def rendering_probe():
    """Small real MuJoCo/EGL context, before any model call or trial budget."""
    if os.environ.get("MUJOCO_PY_FORCE_CPU"):
        raise GPUUnavailable("Clear MUJOCO_PY_FORCE_CPU before GPU rendering")
    import mujoco_py

    model = mujoco_py.load_model_from_xml(
        '<mujoco><worldbody><geom type="sphere" size="0.1" rgba="1 0 0 1"/></worldbody></mujoco>'
    )
    sim = mujoco_py.MjSim(model)
    context = mujoco_py.MjRenderContextOffscreen(sim, device_id=int(os.environ["GPUS"]))
    pixels = sim.render(32, 32)
    result = {
        "shape": list(pixels.shape),
        "pixel_sum": int(pixels.sum()),
        "context_device_id": (
            context.opengl_context.device_id
            if hasattr(context.opengl_context, "device_id")
            else int(os.environ["GPUS"])
        ),
    }
    del context, sim, model
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mapping", action="store_true")
    mode.add_argument("--render-probe", action="store_true")
    mode.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    if args.mapping:
        print(json.dumps(egl_uuid_mapping()))
    elif args.render_probe:
        print(json.dumps(rendering_probe()))
    else:
        mapping = probe_mapping()
        try:
            monitoring = NVMLMonitor().sample(list(mapping))
        except (GPUUnavailable, OSError, AttributeError) as exc:
            monitoring = {"unavailable": str(exc)}
        print(json.dumps({"mapping": mapping, "monitoring": monitoring}, indent=2))
