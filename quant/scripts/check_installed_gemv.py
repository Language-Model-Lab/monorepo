"""Isolated installed-wheel check of the GEMV kernels (K-PLAN-002 stage 4).

Copy this file next to an environment that has the quant wheel installed, outside the checkout, and run it
with that environment's `python -I` and a fresh TORCH_EXTENSIONS_DIR. It checks that:
  - quant, its CUDA sources and every source handed to the JIT compiler come from the installation;
  - no sys.path entry and no loaded module lies under --forbid (the checkout);
  - the extension cache started empty, so the kernels were compiled from the installed sources;
  - the installation has no workspace source identity (it is identified by the wheel's sha256);
  - both kernels match their FP32 references in FP16 and BF16, with and without bias.
Writes a JSON report to --out and exits non-zero if any check fails.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import zipfile
from pathlib import Path

MODEL_SHAPES = [(2048, 1024), (1024, 1024), (1024, 2048), (3072, 1024), (1024, 3072)]  # (N, K)
CONTROL_SHAPES = [(1024, 1024)]
REL_L2 = {"float16": 1e-3, "bfloat16": 1e-2}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wheel_metadata(wheel: Path) -> dict:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        wheel_file = next(n for n in names if n.endswith(".dist-info/WHEEL"))
        generator = next((line.split(":", 1)[1].strip() for line in archive.read(wheel_file).decode().splitlines()
                          if line.startswith("Generator:")), None)
    return {"generator": generator, "csrc_members": sorted(n for n in names if "/csrc/" in n)}


def _under(path: str | Path, root: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forbid", type=Path, required=True, help="the checkout; nothing may be imported from it")
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ext_dir = Path(os.environ["TORCH_EXTENSIONS_DIR"])
    cache_was_empty = not ext_dir.exists() or not any(ext_dir.iterdir())

    from importlib.resources import files

    import torch
    import torch.nn.functional as F

    import quant
    import quant.ext
    from quant.gemv import gemv_16bit_control, int8_gemv
    from quant.precision import ieee_fp32_matmul
    from quant.provenance import workspace_root

    forbid = args.forbid.resolve()
    package = Path(quant.__file__).resolve().parent
    prefix = Path(sys.prefix).resolve()
    csrc = Path(str(files("quant").joinpath("csrc"))).resolve()

    compiled = []
    original_load = quant.ext.load

    def recording_load(*load_args, **kwargs):
        compiled.extend(str(Path(s).resolve()) for s in kwargs.get("sources", ()))
        return original_load(*load_args, **kwargs)

    quant.ext.load = recording_load

    def rel_l2(y, ref):
        return ((y.float() - ref).norm() / ref.norm()).item()

    cases = []
    torch.manual_seed(0)
    with torch.no_grad(), ieee_fp32_matmul():
        for dtype_name, limit in REL_L2.items():
            dtype = getattr(torch, dtype_name)
            for kernel, shapes in (("int8_gemv", MODEL_SHAPES), ("gemv_16bit_control", CONTROL_SHAPES)):
                for n, k in shapes:
                    for with_bias in (False, True):
                        x = torch.randn(1, k, device="cuda").to(dtype)
                        bias = torch.randn(n, device="cuda") * 0.1 if with_bias else None
                        if kernel == "int8_gemv":
                            q = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
                            scales = torch.rand(n, device="cuda") * 1e-2 + 4e-5
                            y = int8_gemv(x, q, scales, bias)
                            ref = F.linear(x.float(), q.float()) * scales
                        else:
                            w = (torch.randn(n, k, device="cuda") * 0.02).to(dtype)
                            y = gemv_16bit_control(x, w, bias)
                            ref = F.linear(x.float(), w.float())
                        if bias is not None:
                            ref = ref + bias
                        err = rel_l2(y, ref)
                        cases.append({"kernel": kernel, "dtype": dtype_name, "N": n, "K": k, "bias": with_bias,
                                      "rel_l2": err, "passed": math.isfinite(err) and err < limit})

    modules_from_checkout = sorted(
        name for name, module in list(sys.modules.items())
        if (f := getattr(module, "__file__", None)) and _under(f, forbid)
    )
    path_entries_in_checkout = [p for p in sys.path if p and _under(p, forbid)]
    checks = {
        "quant_installed_in_environment": package.is_relative_to(prefix) and not _under(package, forbid),
        "csrc_inside_installed_package": csrc.is_relative_to(package),
        "compiled_sources_from_installation": bool(compiled) and all(Path(s).is_relative_to(csrc) for s in compiled),
        "extension_cache_started_empty": cache_was_empty,
        "no_sys_path_entry_in_checkout": not path_entries_in_checkout,
        "no_module_loaded_from_checkout": not modules_from_checkout,
        "no_workspace_source_identity": workspace_root(package) is None,
        "kernels_match_references": all(c["passed"] for c in cases),
    }
    wheel_meta = _wheel_metadata(args.wheel)
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "wheel": {"file": args.wheel.name, "sha256": _sha256(args.wheel), "bytes": args.wheel.stat().st_size,
                  **wheel_meta},
        "sdist": None if args.sdist is None else {"file": args.sdist.name, "sha256": _sha256(args.sdist),
                                                  "bytes": args.sdist.stat().st_size},
        "paths": {"sys_prefix": str(prefix), "quant_package": str(package), "csrc": str(csrc),
                  "compiled_sources": compiled, "extensions_dir": str(ext_dir),
                  "sys_path_entries_in_checkout": path_entries_in_checkout,
                  "modules_from_checkout": modules_from_checkout},
        "source_identity": None,  # an installed wheel is identified by its sha256, not by a workspace manifest
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0)},
        "cases": cases,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "checks": checks, "wheel_sha256": result["wheel"]["sha256"],
                      "cases": f"{sum(c['passed'] for c in cases)}/{len(cases)}"}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
