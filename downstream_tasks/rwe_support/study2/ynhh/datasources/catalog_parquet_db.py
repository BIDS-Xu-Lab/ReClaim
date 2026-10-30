#!/usr/bin/env python3
"""Summarize a parquet database (each subfolder = a table) into a JSON catalog.

Reads only parquet footers, so the data is never loaded. num_rows and per-column
non_null counts come from footer statistics. If a file was written without
statistics, that column's non_null is reported as null (unknown).

Usage: python catalog_parquet_db.py "data base" [output.json]
"""
import json
import sys
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq


def summarize_table(table_dir):
    dataset = ds.dataset(table_dir, format="parquet", partitioning="hive")
    num_rows = 0
    nulls = {}
    unknown = set()                              # columns whose footer lacked null stats
    for frag in dataset.get_fragments():
        md = pq.read_metadata(frag.path)         # footer only, never reads the data
        num_rows += md.num_rows
        for rg in range(md.num_row_groups):
            row_group = md.row_group(rg)
            for c in range(row_group.num_columns):
                col = row_group.column(c)
                stats = col.statistics
                if stats is not None and stats.has_null_count:
                    nulls[col.path_in_schema] = nulls.get(col.path_in_schema, 0) + stats.null_count
                else:
                    unknown.add(col.path_in_schema)
    columns = [
        {"name": f.name, "type": str(f.type),
         "non_null": None if f.name in unknown else num_rows - nulls.get(f.name, 0)}
        for f in dataset.schema
    ]
    return {"num_rows": num_rows, "columns": columns}


def build_catalog(db_dir):
    tables = {}
    for sub in sorted(p for p in db_dir.iterdir() if p.is_dir()):
        if any(sub.rglob("*.parquet")):
            tables[sub.name] = summarize_table(sub)
    return {"database": db_dir.name, "tables": tables}


# No default DB: the OMOP root differs per site; pass it on the command line.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: catalog_parquet_db.py <omop_root_dir> [out_json]")
    db = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / f"{db.name}_catalog.json"
    if not db.is_dir():
        sys.exit(f"not a folder: {db}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(build_catalog(db), f, indent=2, ensure_ascii=False)
    print(f"wrote {out}")
