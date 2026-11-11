"""Source snapshot for evidence runs (K-PLAN-002 stage 4).

  create --out DIR                      write the archive, MANIFEST.sha256 and AGGREGATE.sha256 into DIR
  verify --archive A --manifest M       re-extract A and check it reproduces M exactly (non-zero exit if not)
  aggregate                             print the current source aggregate hash
"""

import argparse
import json
import sys
from pathlib import Path

from quant.provenance import source_identity, verify_snapshot, workspace_root, write_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    commands.add_parser("aggregate")
    args = parser.parse_args()

    root = workspace_root()
    if args.command != "verify" and root is None:
        print("not inside a uv workspace checkout", file=sys.stderr)
        return 1
    if args.command == "create":
        print(json.dumps(write_snapshot(root, args.out), indent=2))
        return 0
    if args.command == "verify":
        result = verify_snapshot(args.archive, args.manifest)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    print(source_identity(root)["aggregate_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
