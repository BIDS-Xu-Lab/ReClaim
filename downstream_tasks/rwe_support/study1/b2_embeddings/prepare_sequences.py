#!/usr/bin/env python3
"""
prepare_sequences.py  --  build the embedding-arm input (pre-index token sequence).

The embedding arm is built from the ReClaim token sequence (month-block resolution), because
rep_* must match what the model was trained on. Reads the cohort.csv, sets the embedding entry
date to trunc(index_date, 'month') so the filter `timestamp < entry_date` blacks out the index
month (the hdPS/outcome arm keeps the day-precise index_date), and constructs `pre_entry_seq`
from the processed token store.

Output: intermediate/mdcr/embeddings/{cohort}_{cell}/prevseq_{data_version}/cohort_with_pre_entry_seq.parquet
        Feed to embed_reclaim (ReClaim) / embed_delphi (Delphi).
Usage: python prepare_sequences.py --cohort_csv <csv> --out_dir <dir> --processed_root <root>
           --data_version <label> --run_name <token-store run folder> --id_prefix CR
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
import argparse
import logging
from functools import reduce
from typing import List, Union

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logging, time_execution  # noqa: E402

INDENT1, INDENT2, INDENT3 = "  ", "    ", "      "

DEFAULT_PROCESSED_ROOT = os.environ.get("MS_PROCESSED_ROOT", "")   # token-store root; set in env.sh


def parse_args():
    p = argparse.ArgumentParser(description="Prepare pre-index token sequence for the embedding arm")
    p.add_argument("--cohort_csv", required=True, help="intermediate/mdcr/cohorts/.../cohort.csv (has person_id, index_date, entry_date_embed)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--processed_root", default=DEFAULT_PROCESSED_ROOT)
    p.add_argument("--data_version", default=os.environ.get("DATA_VERSION"),
                   required=not os.environ.get("DATA_VERSION"),
                   help="Version label of the processed token store, a subdirectory of the processed root (env.sh DATA_VERSION); must match the tokens the checkpoints were trained on")
    # Token-store run folder under {processed_root}/{data_version}/ (env.sh MS_RUN_NAME, as written by
    # preprocessing/pipeline_marketscan); it must be the store the cohorts were gated on. A mismatch does
    # not crash (a vocab mismatch silently degrades rep_*), which is why it comes from env.sh, never guessed.
    p.add_argument("--run_name", default=os.environ.get("MS_RUN_NAME"),
                   required=not os.environ.get("MS_RUN_NAME"),
                   help="Token-store run folder under the processed root and data version (env.sh MS_RUN_NAME)")
    # cohort.csv keys on a bare person_id (MarketScan ENROLID); the token store keys on
    # 'CR'+ENROLID -- the prefix builds the join key.
    p.add_argument("--id_prefix", default="",
                   help="Prefix prepended to cohort person_id to form the token-store key (MarketScan: 'CR').")
    p.add_argument("--token_id_col", default=None,
                   help="Id column in the token store. Default: auto (enrollee_id if present, else person_id).")
    return p.parse_args()


def create_spark_session(app_name: str) -> SparkSession:
    local_dir = os.environ.get("SPARK_LOCAL_DIRS") or os.environ.get("SLURM_TMPDIR") or "/tmp"
    logging.info("%sspark scratch dir: %s", INDENT2, local_dir)
    return (SparkSession.builder.appName(app_name).master("local[*]")
            .config("spark.local.dir", local_dir)
            # Keep auto-broadcast conservative: a large threshold lets Spark auto-broadcast the
            # full demo_seq table, which exceeds the broadcast cap. Explicit F.broadcast() hints
            # are honoured regardless, so the cohort-side broadcasts still happen.
            .config("spark.sql.autoBroadcastJoinThreshold", str(32 * 1024 * 1024))
            # Bigger cohorts need a bigger driver and more shuffle partitions, so neither is hardcoded.
            .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEM", "50g"))
            .config("spark.driver.maxResultSize", os.environ.get("SPARK_MAX_RESULT", "10g"))
            .config("spark.executor.memory", "350g").config("spark.executor.cores", "10")
            .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "240"))
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer").getOrCreate())


def read_and_union_parquet(spark, roots: Union[str, List[str]], subpath: str) -> DataFrame:
    paths = roots if isinstance(roots, list) else [roots]
    return reduce(DataFrame.unionByName, [spark.read.parquet(os.path.join(p, subpath)) for p in paths])


@time_execution
def construct_pre_entry_seq(spark, roots, cohort_csv, out_dir, id_prefix="", token_id_col=None):
    """One pre_entry_seq per cohort person, index month blacked out.

    The join key is built, not assumed: cohort.csv keys on `person_id` (MarketScan ENROLID); the
    token store keys on `enrollee_id` = 'CR' + ENROLID. Both sides are renamed to `token_key`, and
    the original unprefixed person_id is carried through (estimate_ladder merges on `person_id`).
    """
    # The string cast lives only in token_key: person_id keeps its inferred dtype so the parquet
    # output carries the type estimate_ladder reads from cohort.csv.
    cohort = (spark.read.csv(cohort_csv, header=True, inferSchema=True)
              .select("person_id",
                      F.trunc(F.col("index_date"), "month").alias("entry_date"),
                      "arm", "outcome", "age_at_index")
              .withColumn("token_key", F.concat(F.lit(id_prefix), F.col("person_id").cast("string"))))
    demo_seq = read_and_union_parquet(spark, roots, "processed_data/demo_seq")
    ev = read_and_union_parquet(spark, roots, "processed_data/event_seq_per_timestamp")

    if token_id_col is None:
        token_id_col = "enrollee_id" if "enrollee_id" in ev.columns else "person_id"
    if token_id_col not in ev.columns:
        raise ValueError(f"token id column '{token_id_col}' not in event_seq_per_timestamp {ev.columns}")
    logging.info("%sjoin: cohort.person_id -> '%s%s' (token col '%s')",
                 INDENT2, id_prefix or "", "{person_id}", token_id_col)
    ev = ev.withColumnRenamed(token_id_col, "token_key")
    demo_seq = demo_seq.withColumnRenamed(token_id_col, "token_key")

    # Shrink demo_seq to cohort members before any downstream join.
    cohort_keys = F.broadcast(cohort.select("token_key").distinct())
    demo_seq = demo_seq.join(cohort_keys, on="token_key", how="left_semi")

    # Broadcast the cohort: without the hint Spark shuffles the entire event table.
    pre = (ev.join(F.broadcast(cohort), on="token_key", how="inner")
           .filter(F.col("timestamp") < F.col("entry_date")))   # strict '<' == M0 blackout

    # Leakage guard: ATT is TRAILING (it encodes `time_to_next_timestamp`), so the LAST retained
    # group's <ATT-N> would measure the gap to the first POST-index event, leaking when the
    # patient next sought care. Only the final row is affected (the cut keeps a contiguous time
    # prefix), so it uses the ATT-free `seq_per_timestamp`; earlier rows keep the _with_ATT form.
    order_cols = [c for c in ("timestamp", "payer_order", "event_type") if c in pre.columns]
    w_last = Window.partitionBy("token_key").orderBy(*[F.col(c).desc() for c in order_cols])
    pre = (pre.withColumn("_rn", F.row_number().over(w_last))
           .withColumn("_tok", F.when(F.col("_rn") == 1, F.col("seq_per_timestamp"))
                                .otherwise(F.col("seq_per_timestamp_with_ATT"))))

    ev_seq = (pre.groupBy("token_key")
              .agg(F.concat_ws(" ", F.array_sort(F.collect_list(
                  F.struct(*[F.col(c) for c in order_cols], F.col("_tok").alias("tok"))))
                  .getItem("tok")).alias("pre_entry_event_seq"))
              .withColumn("pre_entry_event_seq", F.trim(F.col("pre_entry_event_seq"))))
    # Drive off the cohort so nobody is lost: a person with no pre-index events gets a header-only
    # sequence rather than being dropped ("no recent care" tracks health status, not missing at
    # random; estimate_ladder would otherwise zero-fill an off-manifold vector into the PS).
    seq = (cohort.select("token_key", "person_id", "arm", "outcome", "age_at_index")
           .join(demo_seq, on="token_key", how="left")
           .join(ev_seq, on="token_key", how="left")
           .withColumn("_hdr_only", F.col("pre_entry_event_seq").isNull() |
                                    (F.trim(F.col("pre_entry_event_seq")) == ""))
           # Header-only persons get an <AGE-at-index> token: Delphi's parser reads the age from
           # fixed position 3; ReClaim is position-agnostic. "No recent care" stays encoded as the
           # absence of event tokens.
           .withColumn("pre_entry_event_seq",
                       F.when(F.col("_hdr_only"),
                              F.concat(F.lit("<AGE-"),
                                       F.col("age_at_index").cast("int").cast("string"),
                                       F.lit(">")))
                        .otherwise(F.col("pre_entry_event_seq")))
           .withColumn("pre_entry_seq", F.trim(F.concat_ws(" ", F.lit("<sos>"), F.col("demo_seq"),
                                                           F.col("pre_entry_event_seq")))))
    n_header_only = seq.filter(F.col("_hdr_only")).count()
    if n_header_only:
        logging.info("%s%d person(s) had no pre-index events -> header-only sequence "
                     "(+<AGE-at-index> for the fixed-position Delphi parser)",
                     INDENT2, n_header_only)
    seq = seq.select("person_id", "pre_entry_seq", "arm", "outcome")

    os.makedirs(out_dir, exist_ok=True)
    # Parquet, not CSV: much smaller for one long token-string column and preserves person_id's dtype.
    out_path = os.path.join(out_dir, "cohort_with_pre_entry_seq.parquet")
    logging.info("%sWriting -> %s", INDENT2, out_path)
    seq.toPandas().to_parquet(out_path, compression="snappy", index=False)


@time_execution
def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("prepare_sequences PREP PREVSEQ STARTING...")
    args = parse_args()
    roots = os.path.join(args.processed_root, args.data_version, args.run_name)
    spark = create_spark_session("h4a_prep_prevseq")
    try:
        construct_pre_entry_seq(spark, roots, args.cohort_csv, args.out_dir,
                                id_prefix=args.id_prefix, token_id_col=args.token_id_col)
        logging.info("prepare_sequences COMPLETED. Log: %s", log_file)
    except Exception as e:
        logging.error("prepare_sequences failed: %s", str(e)); raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
