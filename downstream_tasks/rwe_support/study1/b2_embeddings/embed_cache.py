#!/usr/bin/env python3
"""
embed_cache.py -- per-person cache helpers for the embedding stage (b2_1_embeddings.sh).

The embedding parquets are cached per person, not per file: downstream readers join on person_id
and refuse partial coverage, so when a cohort gains members only the missing people are embedded
and appended; everyone already embedded keeps their exact vector.

  coverage  <cohort_csv> <parquet>                      -> prints the number of cohort ids missing
  filter    <prevseq> <cohort_csv> <existing> <out>     -> prevseq rows for the missing ids only
  merge     <existing> <delta> <cohort_csv> <out>       -> existing + delta, carried arm/outcome
                                                           refreshed from cohort.csv, atomic write
"""
import os, sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def ids_csv(path):
    return set(pd.read_csv(path, usecols=["person_id"], dtype={"person_id": str})["person_id"])


def ids_parquet(path):
    return set(pq.read_table(path, columns=["person_id"]).column("person_id").to_pandas().astype(str))


def cmd_coverage(cohort_csv, parquet):
    if not os.path.exists(parquet):
        print(len(ids_csv(cohort_csv))); return
    print(len(ids_csv(cohort_csv) - ids_parquet(parquet)))


def cmd_filter(prevseq, cohort_csv, existing, out):
    missing = ids_csv(cohort_csv) - (ids_parquet(existing) if os.path.exists(existing) else set())
    t = pq.read_table(prevseq)
    pid = t.column("person_id").to_pandas().astype(str)
    mask = pa.array(pid.isin(missing).values)
    sub = t.filter(mask)
    if sub.num_rows != len(missing):
        sys.exit(f"ERROR: prevseq {prevseq} covers {sub.num_rows}/{len(missing)} missing ids -- rebuild the prevseq first")
    pq.write_table(sub, out)
    print(f"delta prevseq: {sub.num_rows:,} rows -> {out}")


def cmd_merge(existing, delta, cohort_csv, out):
    old = pq.read_table(existing); new = pq.read_table(delta)
    if set(old.column_names) != set(new.column_names):
        sys.exit(f"ERROR: schema mismatch old={old.column_names[:5]}... new={new.column_names[:5]}...")
    new = new.select(old.column_names)
    both = pa.concat_tables([old, new])
    pid = both.column("person_id").to_pandas().astype(str)
    if pid.duplicated().any():
        sys.exit("ERROR: duplicate person_id after merge")
    coh = pd.read_csv(cohort_csv, dtype={"person_id": str}).set_index("person_id")
    cols = {}
    for c in both.column_names:
        if c in ("arm", "outcome") and c in coh.columns:   # refresh carried cohort columns
            vals = coh[c].reindex(pid.values)
            arr = both.column(c).to_pandas()
            arr = np.where(vals.notna().values, vals.values, arr)
            cols[c] = pa.array(arr.astype(both.schema.field(c).type.to_pandas_dtype()))
        else:
            cols[c] = both.column(c)
    tmp = out + ".tmp.parquet"
    pq.write_table(pa.table(cols), tmp)
    os.replace(tmp, out)
    missing = ids_csv(cohort_csv) - set(pid)
    if missing:
        sys.exit(f"ERROR: merged parquet still misses {len(missing)} cohort ids")
    print(f"merged: {old.num_rows:,} existing + {new.num_rows:,} new = {both.num_rows:,} rows -> {out}")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"coverage": cmd_coverage, "filter": cmd_filter, "merge": cmd_merge}[cmd](*args)
