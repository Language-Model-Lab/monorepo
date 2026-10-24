import math
from collections.abc import Callable, Sequence

import torch


class L2Flusher:
    """Evicts the L2 cache by reading a buffer twice its size.

    A read pass leaves clean lines in L2. A write pass (e.g. zero_()) would leave dirty lines, whose write-back
    would then add DRAM traffic inside the next timed region.
    """

    def __init__(self, device=None, factor: int = 2):
        device = torch.device(device if device is not None else torch.cuda.current_device())
        l2 = torch.cuda.get_device_properties(device).L2_cache_size
        self.nbytes = factor * l2
        self.buffer = torch.zeros(self.nbytes // 4, dtype=torch.int32, device=device)
        self._out = torch.empty((), dtype=torch.int64, device=device)

    def flush(self) -> None:
        torch.sum(self.buffer, dim=0, dtype=torch.int64, out=self._out)


def time_cuda(
    fn: Callable[[], object],
    warmup: int = 10,
    iters: int = 50,
    flush: L2Flusher | None = None,
    queue_ahead_ms: float = 20.0,
) -> list[float]:
    """Milliseconds per call of fn, from one CUDA event pair per iteration. The L2 flush, if given, runs
    before each start event, outside the timed region. One synchronize at the end.

    A sleep kernel queued before the timed loop keeps the GPU busy while the CPU enqueues the iterations.
    Without it, a short kernel's start event fires while the GPU waits for the CPU to launch fn, and the
    measured time includes launch overhead.
    """
    for _ in range(warmup):
        if flush is not None:
            flush.flush()
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    if queue_ahead_ms > 0:
        torch.cuda._sleep(int(queue_ahead_ms * 1e-3 * 2e9))  # cycles; at least queue_ahead_ms below 2 GHz
    for start, end in zip(starts, ends):
        if flush is not None:
            flush.flush()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


#Buffer copies needed so that, cycling through them, each copy is read again only after at least
#factor x L2 bytes of other copies: every call then reads cold data from DRAM.
def rotation_count(bytes_per_call: int, l2_bytes: int | None = None, factor: int = 2) -> int:
    if l2_bytes is None:
        l2_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    return math.ceil(factor * l2_bytes / bytes_per_call) + 1


def time_cuda_rotating(
    fns: Sequence[Callable[[], object]], warmup: int = 3, iters: int = 50, queue_ahead_ms: float = 20.0
) -> list[float]:
    """Milliseconds per call, for kernels too short to time one call at a time.

    Each sample times all of fns back to back between one event pair and divides by len(fns). Each fn should
    read its own copy of the data (see rotation_count), so every call reads from DRAM without an L2 flush.
    CUDA event readings on this GPU cluster at multiples of 1.024 us; timing len(fns) calls per pair divides
    that granularity by len(fns).
    """
    for _ in range(warmup):
        for fn in fns:
            fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    if queue_ahead_ms > 0:
        torch.cuda._sleep(int(queue_ahead_ms * 1e-3 * 2e9))
    for start, end in zip(starts, ends):
        start.record()
        for fn in fns:
            fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) / len(fns) for start, end in zip(starts, ends)]
