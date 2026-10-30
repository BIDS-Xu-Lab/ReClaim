#!/usr/bin/env python3
"""Logging and timing helpers shared by the Python stages."""

import os
import time
os.environ['TZ'] = 'America/New_York'   # one fixed zone for log timestamps
time.tzset()

import sys
import logging
import inspect
from datetime import datetime
import pyspark.sql.functions as F

INDENT = "    "


class StreamToLogger:
    """Redirect print() output to the logger."""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def setup_logging(script_path=None, capture_print=True, level=logging.INFO):
    """Log to logs/<script>_<timestamp>.log and stdout; optionally capture print().
    Returns the log file path."""
    os.makedirs("logs", exist_ok=True)

    if script_path is None:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        script_path = module.__file__

    script_name = os.path.splitext(os.path.basename(script_path))[0]

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

    if capture_print:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
        logger.info("Logger initialized: %s (print capture enabled)", log_file)
    else:
        logger.info("Logger initialized: %s", log_file)

    return log_file


def format_time_delta(seconds):
    """Seconds -> "1h 30m 45s" / "5m 3s" / "45.23s"."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{seconds:.2f}s"


def time_execution(func):
    """Decorator: log the wall time of a function."""
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
    """Log the distribution of a categorical Spark column (count and percentage per value,
    sorted by count). Only the top_n rows are collected; total_count can be passed in to
    avoid a second scan."""
    start_time = time.time()
    logging.info("%sLogging contingency table for variable %s", INDENT, column_name)

    dist_df = df.groupBy(F.col(column_name)).agg(F.count("*").alias("count")).orderBy(F.col("count").desc())

    if total_count is None:
        stats = df.agg(
            F.count("*").alias("total"),
            F.countDistinct(column_name).alias("unique")
        ).collect()[0]
        total_count = stats["total"]
        unique_count = stats["unique"]
    else:
        unique_count = dist_df.count()

    if total_count == 0:
        logging.info("%sNo data available for %s", INDENT, column_name)
        return None

    logging.info("%s%s distribution (Total: %s, Unique values: %s)", INDENT, column_name, f"{total_count:,}", unique_count)

    if top_n is not None:
        logging.info("%sDisplaying top %s items", INDENT, top_n)
        dist_rows = dist_df.limit(top_n).collect()
    else:
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
