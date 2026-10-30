#!/usr/bin/env python3

# Set timezone to Eastern Standard Time
import os
import time
os.environ['TZ'] = 'America/New_York'
time.tzset()  # Apply the timezone change

import sys
import logging
import inspect
from datetime import datetime
import pyspark.sql.functions as F

INDENT = "    "


class StreamToLogger:
    """Redirect print() output to logger"""
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
    """
    Set up logging to file and console with Eastern Time timestamps
    
    Args:
        script_path (str, optional): Path to the script. If None, __file__ will be used
                                    from the calling script.
        capture_print (bool): Whether to redirect print() to logger. Default True.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING).
                    Default is logging.INFO.
    
    Returns:
        str: Path to the log file
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Get the script name without extension
    if script_path is None:
        # Get the frame of the caller
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        script_path = module.__file__
    
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    
    # Generate default log filename with script name and timestamp
    # Since we set TZ=America/New_York at the top of the file,
    # datetime.now() will automatically use Eastern Time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{script_name}_{timestamp}.log"
    
    # Configure logging to use Eastern Time
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


def format_time_delta(seconds):
    """
    Format time delta in a readable format
    
    Args:
        seconds (float): Number of seconds
    
    Returns:
        str: Formatted time string (e.g. "1h 30m 45s" or "45.23s")
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
    Decorator to measure and log the execution time of a function
    
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


def cache_with_count(df, name=None):
    """
    Cache a DataFrame and return both the cached DataFrame and its count.
    This avoids recomputing the DAG when you need both caching and counting.
    
    Args:
        df: Spark DataFrame to cache
        name: Optional name for logging purposes
    
    Returns:
        tuple: (cached_df, count)
    """
    cached_df = df.cache()
    count = cached_df.count()  # This triggers caching
    if name:
        logging.info("%sCached '%s': %s rows", INDENT, name, f"{count:,}")
    return cached_df, count


def spark_log_contingency_table(df, column_name, top_n=100, total_count=None):
    """
    Log the distribution of a categorical variable as a contingency table.
    Shows count and percentage for each unique value, sorted by count descending.
    
    OPTIMIZED VERSION:
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
def spark_log_continuous_distribution(df, column_name):
    """
    Log the distribution of a continuous variable including mean, stddev, and percentiles.
    WARNING: Can be expensive on large datasets.
    
    Args:
        df: Spark DataFrame
        column_name: Name of the column to analyze
    """
    logging.info("%sCalculating continuous variable %s distribution...", INDENT, column_name)
    logging.info("%s!!! VERY EXPENSIVE TO RUN ON LARGE DATA SET!!!", INDENT)

    stats = df.select(
        F.count(F.lit(1)).alias("total_count"),
        F.count(F.col(column_name)).alias("non_null_count"),
        F.mean(F.col(column_name)).alias("mean"),
        F.stddev(F.col(column_name)).alias("stddev"),
        F.percentile_approx(F.col(column_name), 0.05).alias("p05"),
        F.percentile_approx(F.col(column_name), 0.25).alias("p25"),
        F.percentile_approx(F.col(column_name), 0.50).alias("p50"),
        F.percentile_approx(F.col(column_name), 0.75).alias("p75"),
        F.percentile_approx(F.col(column_name), 0.95).alias("p95"),
        F.percentile_approx(F.col(column_name), 0.99).alias("p99"),
        F.min(F.col(column_name)).alias("min"),
        F.max(F.col(column_name)).alias("max")
    ).collect()[0]

    non_missing_rate = stats['non_null_count'] / stats['total_count'] * 100 if stats['total_count'] > 0 else 0.0

    logging.info("%sDistribution for (%s):******", INDENT, column_name)
    logging.info("%s  Non null count: %s", INDENT, f"{stats['non_null_count']:,}")
    logging.info("%s  Non-missing rate: %.2f%%", INDENT, non_missing_rate)
    logging.info("%s  Mean: %.2f", INDENT, stats['mean'])
    logging.info("%s  STDDEV: %.2f", INDENT, stats['stddev'])
    logging.info("%s  Min: %.2f", INDENT, stats['min'])
    logging.info("%s  Max: %.2f", INDENT, stats['max'])
    logging.info("%s  p05: %s", INDENT, f"{stats['p05']:,}")
    logging.info("%s  p25: %s", INDENT, f"{stats['p25']:,}")
    logging.info("%s  p50: %s", INDENT, f"{stats['p50']:,}")
    logging.info("%s  p75: %s", INDENT, f"{stats['p75']:,}")
    logging.info("%s  p95: %s", INDENT, f"{stats['p95']:,}")
    logging.info("%s  p99: %s", INDENT, f"{stats['p99']:,}")

    logging.info("%sContinuous variable %s distribution calculation completed", INDENT, column_name)

