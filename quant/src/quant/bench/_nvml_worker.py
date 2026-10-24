"""NVML sampling loop, run as a standalone script in its own process (see telemetry.TelemetrySampler).

It imports nothing from the quant package, so it starts fast and never competes with the benchmark for the
Python GIL. One CSV line per sample: monotonic time, SM MHz, memory MHz, temperature C, power W, clock-event
reasons bitmask.
"""

import sys
import time

import pynvml


def main() -> None:
    out_path, interval_ms, index = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    interval = interval_ms / 1000
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    pynvml.nvmlDeviceGetPowerUsage(handle)  # the first power reading after init can be invalid
    with open(out_path, "w", buffering=1) as f:
        while True:
            t = time.monotonic()
            f.write(
                f"{t:.6f},"
                f"{pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)},"
                f"{pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)},"
                f"{pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)},"
                f"{pynvml.nvmlDeviceGetPowerUsage(handle) / 1000:.3f},"
                f"{pynvml.nvmlDeviceGetCurrentClocksEventReasons(handle)}\n"
            )
            time.sleep(max(0.0, interval - (time.monotonic() - t)))


if __name__ == "__main__":
    main()
