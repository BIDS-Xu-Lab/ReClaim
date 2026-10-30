#!/usr/bin/env python3
"""
prepare_sequences.py  --  pre-index token sequences for the embedding pass (union cohort).

Joins the cohort to the token store (val+test split) and builds one sequence per patient:
  * entry_date = trunc(index_date, 'month') -> M0 blackout (strict `timestamp < entry_date`),
    so the index month itself never enters the sequence
  * the LAST retained row uses the ATT-free `seq_per_timestamp`: its trailing <ATT-N> token
    would encode time-to-first-POST-index-event, i.e. leak post-index care seeking
  * cohort-driven LEFT join: patients with no pre-index events get a header-only sequence
    (<sos> <SEX-..> <DOBYR-..> <AGE-..>) rather than being dropped or zero-filled downstream
  * the cohort is broadcast; the event table is never shuffled

Output: intermediate/embeddings/cohort_with_pre_entry_seq.parquet
Usage : python tm2_embeddings/prepare_sequences.py --cohort_csv intermediate/cohorts/cohort.csv
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
import argparse
import logging
from functools import reduce
from typing import List, Union

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F

from utils import setup_logging, time_execution

INDENT2 = "    "

DEFAULT_PROCESSED_ROOT = os.environ.get("MS_PROCESSED_ROOT", "")   # token-store root; set in env.sh


def parse_args():
    p = argparse.ArgumentParser(description="Prepare pre-index token sequences (TTE embedding pass)")
    p.add_argument("--cohort_csv", default="intermediate/cohorts/cohort.csv")
    p.add_argument("--out_dir", default="intermediate/embeddings")
    p.add_argument("--processed_root", default=DEFAULT_PROCESSED_ROOT)
    p.add_argument("--data_version", default=os.environ.get("DATA_VERSION"),
                   required=not os.environ.get("DATA_VERSION"),
                   help="Version label of the processed token store, a subdirectory of the processed root (env.sh DATA_VERSION); must match the tokens the checkpoints were trained on")
    p.add_argument("--test_split", default="both", choices=["val", "test", "both", "1000k"])
    p.add_argument("--id_prefix", default="",
                   help="Token-store key prefix. EMPTY by default: build_cohort namespaces "
                        "person_id at read time ('CR'+ENROLID for CCAE/MDCR, 'MD'+ENROLID for "
                        "MDCD), so the cohort key IS the token-store key.")
    return p.parse_args()


def create_spark_session(app_name: str) -> SparkSession:
    local_dir = os.environ.get("SPARK_LOCAL_DIRS") or os.environ.get("SLURM_TMPDIR") or "/tmp"
    return (SparkSession.builder.appName(app_name).master("local[*]")
            .config("spark.local.dir", local_dir)
            .config("spark.sql.autoBroadcastJoinThreshold", str(32 * 1024 * 1024))
            .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEM", "120g"))
            .config("spark.driver.maxResultSize", os.environ.get("SPARK_MAX_RESULT", "16g"))
            .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "240"))
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer").getOrCreate())


def read_and_union_parquet(spark, roots: Union[str, List[str]], subpath: str) -> DataFrame:
    paths = roots if isinstance(roots, list) else [roots]
    return reduce(DataFrame.unionByName, [spark.read.parquet(os.path.join(p, subpath)) for p in paths])


@time_execution
def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("PREPARE SEQUENCES STARTING...")
    args = parse_args()
    if args.test_split == "both":
        roots = [os.path.join(args.processed_root, args.data_version, "marketscan_val"),
                 os.path.join(args.processed_root, args.data_version, "marketscan_test")]
    else:
        roots = os.path.join(args.processed_root, args.data_version, f"marketscan_{args.test_split}")

    spark = create_spark_session("tm2_prepare_sequences")
    try:
        cohort = (spark.read.csv(args.cohort_csv, header=True, inferSchema=True)
                  .select("person_id",
                          F.trunc(F.col("index_date"), "month").alias("entry_date"),
                          "drug_class", "age_at_index")
                  .withColumn("token_key",
                              F.concat(F.lit(args.id_prefix), F.col("person_id").cast("string"))))
        demo_seq = read_and_union_parquet(spark, roots, "processed_data/demo_seq")
        ev = read_and_union_parquet(spark, roots, "processed_data/event_seq_per_timestamp")

        token_id_col = "enrollee_id" if "enrollee_id" in ev.columns else "person_id"
        ev = ev.withColumnRenamed(token_id_col, "token_key")
        demo_seq = demo_seq.withColumnRenamed(token_id_col, "token_key")

        cohort_keys = F.broadcast(cohort.select("token_key").distinct())
        demo_seq = demo_seq.join(cohort_keys, on="token_key", how="left_semi")

        pre = (ev.join(F.broadcast(cohort), on="token_key", how="inner")
               .filter(F.col("timestamp") < F.col("entry_date")))   # strict '<' == M0 blackout

        # ATT-leak guard: the last retained row takes the ATT-free token
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
               # header-only patients get an <AGE-at-index> token so every sequence carries
               # true demographics (valid token grammar with no trailing events)
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
        logging.info("%s%s sequences (cohort rows must match)", INDENT2, f"{len(out):,}")
        # 100% TOKEN COVERAGE ASSERT: a member absent from the token store would get a
        # header-only sequence and a content-free embedding without any error.
        n_no_demo = int(seq.filter(F.col("demo_seq").isNull()).count())
        if n_no_demo:
            raise RuntimeError(
                f"{n_no_demo} of {out.shape[0]} cohort members have NO token-store record "
                "(demo_seq missing). Check the ENROLID namespacing (CR/MD prefixes) and the "
                "token-store filters (drug coverage, dual-eligible, member-days) against build_cohort's cohort.")

        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "cohort_with_pre_entry_seq.parquet")
        out.to_parquet(out_path, compression="snappy", index=False)
        logging.info("Wrote %s. Log: %s", out_path, log_file)
        logging.info("PREPARE SEQUENCES COMPLETED.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
