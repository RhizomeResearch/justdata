from __future__ import annotations

import csv
import json
from pathlib import Path


def _read_manifest(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        columns = set()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row = {str(k): "" if v is None else str(v) for k, v in row.items()}
                rows.append(row)
                columns.update(row)
        return rows, columns

    with path.open("r", encoding="utf-8", newline="") as f:
        dialect = csv.excel_tab if suffix in {".tsv", ".tab"} else csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = [
            {str(k): "" if v is None else str(v) for k, v in row.items()}
            for row in reader
        ]
        return rows, set(reader.fieldnames or [])
