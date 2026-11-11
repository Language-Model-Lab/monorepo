import tarfile
from pathlib import Path

import pytest

from quant.provenance import (
    aggregate,
    manifest,
    source_files,
    source_identity,
    verify_snapshot,
    workspace_root,
    write_snapshot,
)

INCLUDE = ("pyproject.toml", "pkg", "quant")


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    (root / "pkg" / "src").mkdir(parents=True)
    (root / "quant" / "src").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["pkg"]\n')
    (root / "pkg" / "src" / "mod.py").write_text("VALUE = 1\n")
    (root / "quant" / "src" / "kernel.cu").write_text("// kernel\n")
    return root


def test_workspace_root_is_found_from_inside_and_absent_outside(workspace, tmp_path):
    assert workspace_root(workspace / "pkg" / "src" / "mod.py") == workspace
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert workspace_root(outside) is None
    identity = source_identity(workspace, INCLUDE)
    assert identity["files"] == 3 and identity["aggregate_sha256"] == aggregate(manifest(workspace, INCLUDE))


def test_bytecode_caches_and_outputs_do_not_change_the_identity(workspace):
    before = manifest(workspace, INCLUDE)
    for junk in ("pkg/src/__pycache__/mod.cpython-314.pyc", "pkg/.pytest_cache/v/cache/nodeids",
                 "pkg/src/pkg.egg-info/PKG-INFO", "pkg/build/lib/mod.py", "pkg/src/ext.so",
                 "quant/results/run.json", "pkg/.venv/bin/python"):
        path = workspace / junk
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated")
    assert manifest(workspace, INCLUDE) == before
    assert source_files(workspace, INCLUDE) == ["pkg/src/mod.py", "pyproject.toml", "quant/src/kernel.cu"]


def test_editing_a_source_file_changes_the_aggregate(workspace):
    before = aggregate(manifest(workspace, INCLUDE))
    (workspace / "pkg" / "src" / "mod.py").write_text("VALUE = 2\n")
    assert aggregate(manifest(workspace, INCLUDE)) != before


def test_snapshot_round_trip_is_deterministic(workspace, tmp_path):
    first = write_snapshot(workspace, tmp_path / "a", INCLUDE)
    second = write_snapshot(workspace, tmp_path / "b", INCLUDE)
    assert first["archive_sha256"] == second["archive_sha256"]
    assert first["aggregate_sha256"] == aggregate(manifest(workspace, INCLUDE))
    result = verify_snapshot(Path(first["archive"]), tmp_path / "a" / "MANIFEST.sha256")
    assert result["passed"] and result["files"] == 3
    assert (tmp_path / "a" / "AGGREGATE.sha256").read_text().strip() == first["aggregate_sha256"]


def test_tampered_archive_fails_verification(workspace, tmp_path):
    original = write_snapshot(workspace, tmp_path / "a", INCLUDE)
    (workspace / "pkg" / "src" / "mod.py").write_text("VALUE = 3\n")
    tampered = write_snapshot(workspace, tmp_path / "b", INCLUDE)
    assert not verify_snapshot(Path(tampered["archive"]), tmp_path / "a" / "MANIFEST.sha256")["passed"]
    # An archive with an extra entry fails too.
    extra = tmp_path / "extra.tar.gz"
    with tarfile.open(original["archive"], "r:gz") as src, tarfile.open(extra, "w:gz") as dst:
        for member in src.getmembers():
            dst.addfile(member, src.extractfile(member))
        dst.add(workspace / "pyproject.toml", arcname="unexpected.toml")
    assert not verify_snapshot(extra, tmp_path / "a" / "MANIFEST.sha256")["passed"]


def test_missing_include_entry_is_an_error(workspace):
    with pytest.raises(FileNotFoundError, match="missing.md"):
        source_files(workspace, (*INCLUDE, "missing.md"))


def test_real_checkout_identity():
    root = workspace_root()
    assert root == Path(__file__).resolve().parents[2]
    files = source_files(root)
    assert "quant/src/quant/csrc/gemv.cu" in files and "uv.lock" in files
    assert "documentation/plans/2026-10-28-kernel-phase-1-implementation.md" in files
    assert not any(f.endswith(".pyc") or "/results/" in f or "egg-info" in f for f in files)
