import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[3] / "results" / "bench"


#One row per dict; columns in order of first appearance across all rows.
def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")
