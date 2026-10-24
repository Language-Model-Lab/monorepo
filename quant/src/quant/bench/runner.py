import time
from collections.abc import Callable

from quant.bench.telemetry import TelemetrySampler, is_stable, stability, window


def timed_block(
    measure: Callable[[], list[float]],
    sampler: TelemetrySampler,
    *,
    threshold: float = 0.05,
    max_attempts: int = 3,
    min_duration_s: float = 0.5,
) -> list[dict]:
    """Calls measure() (a timing function returning per call milliseconds, e.g. time_cuda or
    time_cuda_rotating bound to its arguments) until a block's SM clock spread is within the threshold, or
    the attempts run out. Returns every attempt: attempt index, per-call times (ms), telemetry stability,
    and whether it was accepted. The last attempt is the one to report.

    Each block repeats measure() until it lasts min_duration_s, so it spans several telemetry samples.
    A block shorter than the sampling interval would be judged by the samples around it, which are taken
    while the GPU is idle. One untimed call of measure() runs first: one-time library initialization (for
    example cuBLAS handles on the first F.linear) takes CPU time with the GPU idle, and would otherwise fill
    the first block's telemetry window with idle readings.
    """
    measure()
    attempts = []
    for attempt in range(max_attempts):
        t0 = time.monotonic()
        times = []
        while not times or time.monotonic() - t0 < min_duration_s:
            times += measure()
        t1 = time.monotonic()
        sampler.wait_until(t1)
        stab = stability(window(sampler.samples(), t0, t1))
        accepted = is_stable(stab, threshold)
        attempts.append({"attempt": attempt, "times_ms": times, "block_s": t1 - t0, "stability": stab,
                         "accepted": accepted})
        if accepted:
            break
    return attempts
