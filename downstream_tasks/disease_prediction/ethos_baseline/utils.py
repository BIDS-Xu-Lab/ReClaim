"""
Shared Utilities

Logging and timing for a long run, the two file reads that happen outside Spark, and the helpers
that locate and summarise an OMOP table.

Every script of this baseline writes a timestamped log under logs/ and sends its progress bars
there as well, which leaves ERROR to mean that something actually went wrong.
"""

import inspect
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

INDENT = "    "

# How often a progress bar may redraw. tqdm redraws several times a second, which reads well on a
# terminal but writes thousands of lines into the log of a stage that runs for hours.
PROGRESS_MIN_INTERVAL_SECONDS = 60


class StreamToLogger:
    """Redirect print() output to logger."""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

    def isatty(self):
        """A logger is never a terminal, so anything drawing on one is told to keep it plain."""
        return False


def progress(iterable, **kwargs):
    """
    Wrap an iterable in a progress bar that reports as progress rather than as a failure.

    tqdm writes to stderr, which setup_logging hands to the logger at ERROR level, so every redraw of
    a bar would be logged as though something had gone wrong and would bury the errors that really
    did. Writing to stdout instead keeps progress at INFO and leaves ERROR to mean what it says.

    Args:
        iterable: The iterable to report progress over
        **kwargs: Passed through to tqdm

    Returns:
        The tqdm-wrapped iterable
    """
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("mininterval", PROGRESS_MIN_INTERVAL_SECONDS)
    return tqdm(iterable, **kwargs)


def setup_logging(script_path=None, capture_print=True, level=logging.INFO):
    """
    Set up logging to a timestamped file and to the console.

    Args:
        script_path: Script the log is named after, the caller's own file when None
        capture_print: Whether to redirect print() to the logger
        level: Logging level, e.g. logging.DEBUG or logging.INFO

    Returns:
        str: Path to the log file
    """
    os.makedirs("logs", exist_ok=True)

    if script_path is None:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        script_path = module.__file__

    script_name = os.path.splitext(os.path.basename(script_path))[0]

    # Timestamps are the machine's local time, so a log can be lined up against anything else it ran
    # alongside
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{script_name}_{timestamp}.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger()

    # redirect print() to logger if requested
    if capture_print:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
        logger.info("Logger initialized: %s (print capture enabled)", log_file)
    else:
        logger.info("Logger initialized: %s", log_file)

    return log_file


def format_time_delta(seconds: float) -> str:
    """
    Write a duration the way it reads most easily at its own scale.

    Args:
        seconds: Duration to format

    Returns:
        str: The duration, e.g. "1h 30m 45s" or "45.23s"
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{seconds:.2f}s"


def time_execution(func):
    """
    Decorator to measure and log the execution time of a function.

    Args:
        func: The function to be timed

    Returns:
        The wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        formatted_time = format_time_delta(execution_time)
        logging.info("Function '%s' executed in %s", func.__name__, formatted_time)
        return result
    return wrapper


def spark_log_contingency_table(df, column_name, top_n=100, total_count=None):
    """
    Log the distribution of a categorical variable as a contingency table.
    Shows count and percentage for each unique value, sorted by count descending.

    - Uses a single aggregation to get both total and distribution
    - Only collects top N rows (default 100) instead of all unique values
    - Accepts pre-computed total_count to avoid redundant scans

    Args:
        df: Spark DataFrame
        column_name: Name of the column to analyze
        top_n: Limit on number of rows to display (default 100, None = all - NOT RECOMMENDED)
        total_count: Optional pre-computed total count to avoid recomputation

    Returns:
        dict with distribution info or None if no data
    """
    # pyspark is needed only by the stages that build the timelines, so importing it here keeps the
    # evaluation stage runnable without it
    import pyspark.sql.functions as F

    start_time = time.time()
    logging.info("%sLogging contingency table for variable %s", INDENT, column_name)

    # Get distribution grouped by column, sorted by count desc
    dist_df = df.groupBy(F.col(column_name)).agg(F.count("*").alias("count")).orderBy(F.col("count").desc())

    # Get unique count and total in a single pass if total_count not provided
    if total_count is None:
        # Use a single aggregation to get both total and unique count
        stats = df.agg(
            F.count("*").alias("total"),
            F.countDistinct(column_name).alias("unique")
        ).collect()[0]
        total_count = stats["total"]
        unique_count = stats["unique"]
    else:
        # Only need unique count
        unique_count = dist_df.count()

    if total_count == 0:
        logging.info("%sNo data available for %s", INDENT, column_name)
        return None

    logging.info("%s%s distribution (Total: %s, Unique values: %s)", INDENT, column_name, f"{total_count:,}", unique_count)

    # Only collect top N rows - avoid collecting all data for high-cardinality columns
    if top_n is not None:
        logging.info("%sDisplaying top %s items", INDENT, top_n)
        dist_rows = dist_df.limit(top_n).collect()
    else:
        # Warning: This can be slow for high-cardinality columns
        logging.warning("%sCollecting ALL unique values - this may be slow for high-cardinality columns", INDENT)
        dist_rows = dist_df.collect()

    logging.info("%s%-6s %-20s %-15s %-15s", INDENT, "Rank", "Value", "Count", "Percentage (%)")
    logging.info("%s%s", INDENT, "-" * 60)

    distribution_list = []
    for rank, row in enumerate(dist_rows, start=1):
        value = row[column_name]
        count_val = row["count"]
        count_formatted = f"{count_val:,}"
        pct = (count_val / total_count * 100)
        value_str = str(value) if value is not None else 'NULL'
        logging.info("%s%-6s %-20s %-15s %-15.2f", INDENT, rank, value_str, count_formatted, pct)
        distribution_list.append({
            'value': value,
            'count': count_val,
            'percentage': pct
        })

    if top_n is not None and unique_count > top_n:
        logging.info("%s... and %s more values", INDENT, unique_count - top_n)

    execution_time = time.time() - start_time
    logging.info("Function 'spark_log_contingency_table' executed in %s", format_time_delta(execution_time))


@time_execution
def omop_table_path(omop_dir, table):
    """
    Resolve the location of an OMOP table.

    EHRSHOT ships each table either as a directory of parquet parts (<table>/part-*.parquet) or
    as a single parquet file (<table>.parquet). Both are readable with spark.read.parquet once
    the right path is chosen.

    Args:
        omop_dir: Directory containing the OMOP tables
        table: OMOP table name without extension (e.g. "person")

    Returns:
        str: Path to the parquet directory or file holding the table
    """
    directory = os.path.join(omop_dir, table)
    if os.path.isdir(directory):
        return directory
    return f"{directory}.parquet"


def omop_table_exists(omop_dir, table):
    """
    Check whether an OMOP table is present as a directory of parts or as a single parquet file.

    Args:
        omop_dir: Directory containing the OMOP tables
        table: OMOP table name without extension (e.g. "concept_relationship")

    Returns:
        bool: True when the table can be read
    """
    directory = os.path.join(omop_dir, table)
    return os.path.isdir(directory) or os.path.exists(f"{directory}.parquet")


def load_icd_cm_code_to_name(ref_dir) -> dict:
    """
    Load the mapping from a dotless ICD-10-CM code to the title it is spelled out with.

    Several rows can describe the same code, in which case the shortest title is kept so that the
    choice stays the same between runs.

    Args:
        ref_dir: Directory containing the code system reference tables

    Returns:
        dict: Mapping of an ICD-10-CM code, including a 3-character category, to its title
    """
    df = pd.read_csv(Path(ref_dir) / "icd10cm-order-Jan-2021.csv.gz", dtype=str)
    return df.groupby("code").long.agg(lambda titles: min(titles, key=len)).to_dict()


def read_hdf5(path) -> dict:
    """
    Read a timelines file into arrays.

    Args:
        path: HDF5 file holding the timelines

    Returns:
        dict: The six datasets of the file, keyed by name
    """
    with h5py.File(path, "r") as f:
        return {
            "times": f["times"][:],
            "tokens": f["tokens"][:],
            "patient_data_offsets": f["patient_data_offsets"][:],
            "patient_ids": f["patient_ids"][:],
            "patient_context": f["patient_context"][:],
            "age_reference": f["age_reference"][:],
        }
