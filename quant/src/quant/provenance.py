"""Source identity for evidence runs without version control (K-PLAN-002 stage 4).

A manifest lists the sha256 of every source file, one `<sha256>  <path>` line per file with workspace-relative
POSIX paths sorted bytewise (the `sha256sum -c` format). Its aggregate is the sha256 of the manifest bytes.
A deterministic archive of exactly those files keeps the bytes themselves, so the identity stays
reconstructable after the working tree changes. Bytecode, caches, build products and the output directories
are never part of the identity, so importing, testing or writing results cannot change it.
"""

import gzip
import hashlib
import io
import tarfile
import tempfile
import tomllib
from pathlib import Path

INCLUDE = (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "backend/pyproject.toml",
    "backend/README.md",
    "backend/.python-version",
    "backend/src",
    "backend/tests",
    "backend/JSON_examples",
    "quant/pyproject.toml",
    "quant/README.md",
    "quant/src",
    "quant/scripts",
    "quant/tests",
    "documentation/plans/2026-10-25-kernel-phase-1.md",
    "documentation/plans/2026-10-28-kernel-phase-1-implementation.md",
)
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".venv", "build", "dist"}
EXCLUDED_DIR_SUFFIXES = (".egg-info",)
EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo", ".so")
OUTPUT_DIRS = ("quant/results", "documentation/artifacts")


def workspace_root(start: str | Path | None = None) -> Path | None:
    """The nearest ancestor of start (default: this file) whose pyproject.toml declares a uv workspace, or
    None, as for an installed wheel outside any checkout."""
    here = Path(start if start is not None else __file__).resolve()
    for directory in (here, *here.parents):
        candidate = directory / "pyproject.toml"
        if not candidate.is_file():
            continue
        try:
            config = tomllib.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if "workspace" in config.get("tool", {}).get("uv", {}):
            return directory
    return None


def _excluded(rel: Path) -> bool:
    if any(part in EXCLUDED_DIRS or part.endswith(EXCLUDED_DIR_SUFFIXES) for part in rel.parts[:-1]):
        return True
    if rel.name.endswith(EXCLUDED_FILE_SUFFIXES):
        return True
    posix = rel.as_posix()
    return any(posix == d or posix.startswith(d + "/") for d in OUTPUT_DIRS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sorted(paths) -> list[str]:
    return sorted(paths, key=lambda p: p.encode("utf-8"))


#Workspace-relative POSIX paths of the source files, sorted bytewise. A missing include entry is an error.
def source_files(root: Path, include: tuple[str, ...] = INCLUDE) -> list[str]:
    root = Path(root)
    files = set()
    for entry in include:
        path = root / entry
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = [p for p in path.rglob("*") if p.is_file()]
        else:
            raise FileNotFoundError(f"snapshot entry {entry!r} does not exist under {root}")
        files.update(p.relative_to(root).as_posix() for p in candidates if not _excluded(p.relative_to(root)))
    return _sorted(files)


def manifest(root: Path, include: tuple[str, ...] = INCLUDE) -> str:
    root = Path(root)
    return "".join(f"{_sha256(root / rel)}  {rel}\n" for rel in source_files(root, include))


def aggregate(manifest_text: str) -> str:
    return hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()


def source_identity(root: Path | None = None, include: tuple[str, ...] = INCLUDE) -> dict | None:
    """Aggregate hash of the current source, or None outside a workspace."""
    root = Path(root) if root is not None else workspace_root()
    if root is None:
        return None
    text = manifest(root, include)
    lock = root / "uv.lock"
    return {
        "root": str(root),
        "files": text.count("\n"),
        "aggregate_sha256": aggregate(text),
        "uv_lock_sha256": _sha256(lock) if lock.is_file() else None,
    }


def write_snapshot(root: Path, out_dir: Path, include: tuple[str, ...] = INCLUDE) -> dict:
    """Writes source-<aggregate[:12]>.tar.gz (fixed order, timestamps, owners and modes, so identical sources
    give identical bytes), MANIFEST.sha256 and AGGREGATE.sha256 into out_dir."""
    root, out_dir = Path(root), Path(out_dir)
    text = manifest(root, include)
    digest = aggregate(text)
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel in source_files(root, include):
            data = (root / rel).read_bytes()
            info = tarfile.TarInfo(rel)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"source-{digest[:12]}.tar.gz"
    with open(archive, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(tar_bytes.getvalue())
    (out_dir / "MANIFEST.sha256").write_text(text, encoding="utf-8")
    (out_dir / "AGGREGATE.sha256").write_text(digest + "\n", encoding="utf-8")
    return {
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "files": text.count("\n"),
        "aggregate_sha256": digest,
    }


def verify_snapshot(archive: Path, manifest_path: Path) -> dict:
    """Extracts the archive into a temporary directory and recomputes the manifest from what it contains:
    passes only if every entry, and nothing else, matches the saved manifest."""
    expected = Path(manifest_path).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        extracted = _sorted(p.relative_to(tmp).as_posix() for p in Path(tmp).rglob("*") if p.is_file())
        recomputed = "".join(f"{_sha256(Path(tmp) / rel)}  {rel}\n" for rel in extracted)
    return {
        "passed": recomputed == expected,
        "files": len(extracted),
        "aggregate_sha256": aggregate(recomputed),
        "expected_aggregate_sha256": aggregate(expected),
    }
