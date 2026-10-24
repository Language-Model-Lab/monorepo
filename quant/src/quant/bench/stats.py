import statistics

#n, median, quartiles, IQR, mean, std, min, max. Quartiles interpolate linearly (numpy's default).
def summarize(values) -> dict:
    v = sorted(values)
    if not v:
        raise ValueError("no values to summarize")
    q25, median, q75 = statistics.quantiles(v, n=4, method="inclusive") if len(v) > 1 else (v[0],) * 3
    return {
        "n": len(v),
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "mean": statistics.fmean(v),
        "std": statistics.stdev(v) if len(v) > 1 else 0.0,
        "min": v[0],
        "max": v[-1],
    }


def bytes_gemv(n: int, k: int, weight_bytes_per_elem: float, act_bytes_per_elem: int, scale_bytes: int = 0) -> int:
    """Bytes a batch-1 linear layer moves (plan section 6.3): weights + scales + x + y."""
    return int(n * k * weight_bytes_per_elem) + scale_bytes + (k + n) * act_bytes_per_elem


def gbs(nbytes: int, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9
