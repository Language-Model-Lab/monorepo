"""GPU telemetry and run conditions."""

import datetime
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

WORKER = Path(__file__).with_name("_nvml_worker.py")
REPO = Path(__file__).resolve().parents[3]

# NVML clock-event reason bits (nvml.h, nvmlClocksEventReason*)
REASON_SW_POWER_CAP = 0x4
REASON_HW_SLOWDOWN = 0x8
REASON_SW_THERMAL = 0x20
REASON_HW_THERMAL = 0x40
REASON_HW_POWER_BRAKE = 0x80
POWER_REASONS = REASON_SW_POWER_CAP | REASON_HW_POWER_BRAKE
THERMAL_REASONS = REASON_SW_THERMAL | REASON_HW_THERMAL | REASON_HW_SLOWDOWN


@dataclass(frozen=True)
class Sample:
    t: float  # time.monotonic(), comparable across processes
    sm_mhz: int
    mem_mhz: int
    temp_c: int
    power_w: float
    reasons: int


class TelemetrySampler:
    """Samples NVML every interval_ms in a separate process while the context is open.

    The sampler process runs a standalone script, so it never competes with the benchmark for the GIL.
    Samples go to a file; samples() reads whatever has been written so far.
    NVML device 0 is assumed to be the CUDA device in use (single-GPU machines).
    """

    def __init__(self, interval_ms: float = 100, device_index: int = 0):
        self.interval_ms = interval_ms
        self.device_index = device_index
        self._dir = None
        self._proc = None
        self._final = None

    def __enter__(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "samples.csv"
        self._proc = subprocess.Popen(
            [sys.executable, str(WORKER), str(self.path), str(self.interval_ms), str(self.device_index)]
        )
        deadline = time.monotonic() + 10
        while not self.samples():
            if self._proc.poll() is not None:
                raise RuntimeError("NVML sampler exited before writing a sample")
            if time.monotonic() > deadline:
                raise RuntimeError("NVML sampler wrote no sample within 10 s")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self._proc.terminate()
        self._proc.wait(timeout=5)
        self._final = self.samples()
        self._dir.cleanup()
        return False

    def wait_until(self, t: float, timeout: float = 2.0) -> None:
        """Blocks until a sample taken after time t exists, so a window ending at t has its right neighbor."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            samples = self.samples()
            if samples and samples[-1].t > t:
                return
            time.sleep(self.interval_ms / 1000 / 4)

    def samples(self) -> list[Sample]:
        if self._final is not None:
            return self._final
        try:
            lines = self.path.read_text().splitlines()
        except FileNotFoundError:
            return []
        out = []
        for line in lines:
            parts = line.split(",")
            if len(parts) != 6:  # a line still being written
                continue
            out.append(Sample(float(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4]), int(parts[5])))
        return out


def window(samples: list[Sample], t0: float, t1: float) -> list[Sample]:
    """Samples taken during [t0, t1]. Fewer than two inside (a run shorter than the sampling interval)
    adds the nearest sample on each side."""
    inside = [s for s in samples if t0 <= s.t <= t1]
    if len(inside) >= 2:
        return inside
    before = [s for s in samples if s.t < t0][-1:]
    after = [s for s in samples if s.t > t1][:1]
    return before + inside + after


def stability(samples: list[Sample]) -> dict:
    if not samples:
        return {"n_samples": 0}
    sm = [s.sm_mhz for s in samples]
    mem = [s.mem_mhz for s in samples]
    sm_median, mem_median = statistics.median(sm), statistics.median(mem)
    return {
        "n_samples": len(samples),
        "sm_median_mhz": sm_median,
        "sm_min_mhz": min(sm),
        "sm_max_mhz": max(sm),
        "sm_spread": (max(sm) - min(sm)) / sm_median if sm_median else float("inf"),
        "mem_median_mhz": mem_median,
        "mem_spread": (max(mem) - min(mem)) / mem_median if mem_median else float("inf"),
        "temp_max_c": max(s.temp_c for s in samples),
        "power_mean_w": statistics.fmean(s.power_w for s in samples),
        "power_max_w": max(s.power_w for s in samples),
        "power_capped": any(s.reasons & POWER_REASONS for s in samples),
        "thermal_slowdown": any(s.reasons & THERMAL_REASONS for s in samples),
    }


def is_stable(stab: dict, threshold: float = 0.05) -> bool:
    """Plan section 6.2: reject runs whose SM clock varies beyond the threshold."""
    return stab.get("n_samples", 0) > 0 and stab["sm_spread"] <= threshold


def clock_lock_check(samples: list[Sample], requested_mhz: float | None) -> dict:
    """Whether a clock lock the user applied (sudo nvidia-smi -lgc) held: median SM clock within 1% of the
    request, and spread within 1%."""
    if requested_mhz is None:
        return {"requested_mhz": None, "attempted": False}
    stab = stability(samples)
    held = (
        stab.get("n_samples", 0) > 0
        and abs(stab["sm_median_mhz"] - requested_mhz) <= 0.01 * requested_mhz
        and stab["sm_spread"] <= 0.01
    )
    return {"requested_mhz": requested_mhz, "attempted": True, "held": held, **stab}


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def on_ac_power() -> bool | None:
    supplies = [p for p in Path("/sys/class/power_supply").glob("*") if _read(p / "type") == "Mains"]
    if not supplies:
        return None
    return any(_read(p / "online") == "1" for p in supplies)


#Run conditions for the JSON sidecar. Warns when not on AC power or not in the performance profile.
def environment(device=None) -> dict:
    import pynvml
    import transformers

    device = torch.device(device if device is not None else torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    bus_bits = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
    max_mem_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    env = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "l2_bytes": props.L2_cache_size,
        "memory_bytes": props.total_memory,
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "max_sm_mhz": pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM),
        "max_mem_mhz": max_mem_mhz,
        "memory_bus_bits": bus_bits,
        # GDDR6: NVML reports half the per-pin data rate
        "theoretical_bw_gbs": 2 * max_mem_mhz * 1e6 * bus_bits / 8 / 1e9,
        "enforced_power_limit_w": pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000,
        "on_ac_power": on_ac_power(),
        "power_profile": _read("/sys/firmware/acpi/platform_profile"),
        "git_commit": commit,
    }
    env["warnings"] = []
    if env["on_ac_power"] is False:
        env["warnings"].append("not on AC power")
    if env["power_profile"] not in (None, "performance"):
        env["warnings"].append(f"power profile is {env['power_profile']!r}, not 'performance'")
    for message in env["warnings"]:
        warnings.warn(message, stacklevel=2)
    return env
