#!/usr/bin/env python3
"""
prepare_sequences.py -- pre-index token sequences for the embedding pass (ty2_1).

Joins the union cohort to the ReClaim token store (demo_seq + event_seq_per_timestamp under
{processed_root}/{data_version}/{store}; ids key on a bare person_id) and writes one row per cohort
person: person_id, drug_class, pre_entry_seq = <sos> + demographic header + all events with
`timestamp < entry_date` (entry_date = the index month, so no index-month event leaks into the
sequence). The last retained row uses the token without the ATT (time-gap) suffix, since no
later event follows it. Persons with no pre-index events get a header-only sequence.

Output: <out_dir>/cohort_with_pre_entry_seq.parquet.
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
import argparse
import logging

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from utils import setup_logging, time_execution

INDENT2 = "    "


def parse_args():
    p = argparse.ArgumentParser(description="Prepare pre-index token sequences (OMOP TTE)")
    p.add_argument("--cohort_csv", required=True)
    p.add_argument("--out_dir", default="intermediate/embeddings")
    p.add_argument("--processed_root",
                   default=os.environ.get("PROCESSED_ROOT", ""),
                   help="Token-store root (env.sh PROCESSED_ROOT; <data_version>/ynhh_full beneath)")
    p.add_argument("--data_version", default=os.environ.get("DATA_VERSION"),
                   required=not os.environ.get("DATA_VERSION"),
                   help="Version label of the processed token store, a subdirectory of the processed root (env.sh DATA_VERSION); must match the tokens the checkpoints were trained on")
    p.add_argument("--store", default="ynhh_full",
                   help="Store directory under {processed_root}/{data_version}/ (default ynhh_full)")
    return p.parse_args()


def create_spark_session(app_name):
    local_dir = os.environ.get("SPARK_LOCAL_DIRS") or os.environ.get("SLURM_TMPDIR") or "/tmp"
    return (SparkSession.builder.appName(app_name).master("local[*]")
            .config("spark.local.dir", local_dir)
            .config("spark.sql.autoBroadcastJoinThreshold", str(32 * 1024 * 1024))
            .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEM", "120g"))
            .config("spark.driver.maxResultSize", os.environ.get("SPARK_MAX_RESULT", "16g"))
            .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "240"))
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer").getOrCreate())


@time_execution
def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("PREPARE SEQUENCES STARTING...")
    args = parse_args()
    root = os.path.join(args.processed_root, args.data_version, args.store)

    spark = create_spark_session("ty2_prepare_sequences")
    try:
        cohort = (spark.read.csv(args.cohort_csv, header=True, inferSchema=True)
                  .select("person_id",
                          F.trunc(F.col("index_date"), "month").alias("entry_date"),
                          "drug_class", "age_at_index")
                  .withColumn("token_key", F.col("person_id").cast("string")))
        demo_seq = spark.read.parquet(os.path.join(root, "demo_seq"))
        ev = spark.read.parquet(os.path.join(root, "event_seq_per_timestamp"))

        token_id_col = "enrollee_id" if "enrollee_id" in ev.columns else "person_id"
        ev = ev.withColumnRenamed(token_id_col, "token_key") \
               .withColumn("token_key", F.col("token_key").cast("string"))
        demo_seq = demo_seq.withColumnRenamed(token_id_col, "token_key") \
                           .withColumn("token_key", F.col("token_key").cast("string"))

        cohort_keys = F.broadcast(cohort.select("token_key").distinct())
        demo_seq = demo_seq.join(cohort_keys, on="token_key", how="left_semi")

        # Tokenization-coverage guard: with build_cohort's gate on trajectory/, every cohort
        # person must be in the store; a miss means a wrong store path or a cohort built
        # without the gate. Fail loudly rather than embed <sos>-only sequences.
        _n_cohort = cohort.select("token_key").distinct().count()
        _n_tok = demo_seq.select("token_key").distinct().count()
        if _n_tok < _n_cohort:
            raise ValueError(f"{_n_cohort - _n_tok} of {_n_cohort} cohort person(s) missing from "
                             f"the token store demo_seq at {root} -- wrong store path, or the "
                             f"cohort was built without the --restrict_ids gate.")
        logging.info("%stokenization coverage: %d/%d cohort persons (100%%)", INDENT2, _n_tok, _n_cohort)

        pre = (ev.join(F.broadcast(cohort), on="token_key", how="inner")
               .filter(F.col("timestamp") < F.col("entry_date")))   # strictly before the index month

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

        seq = (cohort.select("token_key", "person_id", "drug_class", "age_at_index")
               .join(demo_seq, on="token_key", how="left")
               .join(ev_seq, on="token_key", how="left")
               .withColumn("_hdr_only", F.col("pre_entry_event_seq").isNull() |
                                        (F.trim(F.col("pre_entry_event_seq")) == ""))
               # Header-only patients (no pre-index events; admitted because there is no
               # observability gate) get an <AGE-at-index> token appended so every sequence
               # carries true demographics. The age is the cohort's age_at_index, and an age
               # token with no trailing events is valid token grammar.
               .withColumn("pre_entry_event_seq",
                           F.when(F.col("_hdr_only"),
                                  F.concat(F.lit("<AGE-"),
                                           F.col("age_at_index").cast("int").cast("string"),
                                           F.lit(">")))
                            .otherwise(F.col("pre_entry_event_seq")))
               .withColumn("pre_entry_seq",
                           F.trim(F.concat_ws(" ", F.lit("<sos>"), F.col("demo_seq"),
                                              F.col("pre_entry_event_seq")))))
        n_header_only = seq.filter(F.col("_hdr_only")).count()
        if n_header_only:
            logging.info("%s%d person(s) had no pre-index events -> header-only sequence "
                         "(+<AGE-at-index> appended)", INDENT2, n_header_only)
        out = seq.select("person_id", "drug_class", "pre_entry_seq").toPandas()
        logging.info("%s%s sequences", INDENT2, f"{len(out):,}")

        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "cohort_with_pre_entry_seq.parquet")
        out.to_parquet(out_path, compression="snappy", index=False)
        logging.info("Wrote %s. Log: %s", out_path, log_file)
        logging.info("PREPARE SEQUENCES COMPLETED.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
