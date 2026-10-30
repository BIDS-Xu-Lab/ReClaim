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
    # force=True ensures this works even if logging was already configured by imports
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True,
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




# =============================================================================
# Content-keyed sampling (layout-independent determinism) — DuckDB-backed
# =============================================================================
# Every random selection in the pipeline is keyed on a salted sha256 of the
# enrollee id — never on file membership or row position — so the result is
# identical across partitionings, row orders, file counts and clusters as long
# as the rows themselves are the same.  (Sampling by file or row position is not:
# a re-partitioned copy of the same corpus yields a different sample.)
#
# The selection "the n smallest (hash, id) pairs" is expressed ONCE, in SQL,
# and evaluated by DuckDB (mature, parallel, streaming, spills to a temp dir):
#     ORDER BY <hash64(salt, id)>, id  LIMIT n
# hash64 = the leading 8 bytes of sha256(f"{salt}:{id}") as an unsigned 64-bit
# integer — deliberately NOT DuckDB's native hash() nor python's hash(), both of
# which are version/process dependent.  (Independent check, if ever needed:
# int.from_bytes(hashlib.sha256(f"{salt}:{id}".encode()).digest()[:8], "big").)

ID_COL = "enrollee_id"
CORPUS_FP_SALT = "corpus-fingerprint"   # seed-INDEPENDENT: comparable across clusters
DUCKDB_VERSION = "1.3.2"                # pinned in env_build; sha256()/casts are stable, hash() is not


def hash_salt(seed: int, purpose: str) -> str:
    """One salt per (seed, purpose) so the pool / split / pairs draws are independent."""
    return f"{seed}:{purpose}"


def hash64_sql(salt: str, col: str = ID_COL) -> str:
    """SQL for the 64-bit content key: leading 8 bytes of sha256(salt ':' id)."""
    return f"('0x' || substr(sha256('{salt}:' || {col}), 1, 16))::UBIGINT"


def smallest_n_sql(source_sql: str, n: int, salt: str, col: str = ID_COL, select: str = "*") -> str:
    """SQL selecting EXACTLY n rows of `source_sql`: the n smallest (key, id) pairs.
    Ties on the 64-bit key (astronomically rare) break by id, so the result is a
    pure function of the id multiset."""
    return (f"SELECT {select} FROM {source_sql} "
            f"ORDER BY {hash64_sql(salt, col)}, {col} LIMIT {int(n)}")


def duck_connect(threads=None, memory_limit_gb=None, temp_dir=None):
    """A DuckDB connection with job-appropriate settings (threads, memory cap,
    node-local spill dir).  Import is lazy: only sampling stages need duckdb."""
    import duckdb
    assert duckdb.__version__ == DUCKDB_VERSION, f"duckdb {duckdb.__version__} != pinned {DUCKDB_VERSION}"
    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads={int(threads)}")
    if memory_limit_gb:
        con.execute(f"SET memory_limit='{int(memory_limit_gb)}GB'")
    if temp_dir:
        con.execute(f"SET temp_directory='{temp_dir}'")
    return con


def select_smallest_ids(con, ids, n: int, salt: str) -> set:
    """In-memory variant (split / pairs): the n ids with the smallest keys."""
    import pyarrow as pa
    con.register("_ids", pa.table({ID_COL: list(ids)}))
    try:
        return {r[0] for r in con.execute(smallest_n_sql("_ids", n, salt, select=ID_COL)).fetchall()}
    finally:
        con.unregister("_ids")


def corpus_fingerprint(con, source_sql: str, col: str = ID_COL):
    """Order-invariant fingerprint of an id multiset: (rows, sum64, xor64) of the
    fixed-salt key.  Equal fingerprints => same rows, whatever the partitioning.
    sum64 catches duplicates that xor would cancel."""
    n, x, s = con.execute(
        f"SELECT count(*), bit_xor(h), (sum(h)::HUGEINT % 18446744073709551616)::UBIGINT "
        f"FROM (SELECT {hash64_sql(CORPUS_FP_SALT, col)} AS h FROM {source_sql})").fetchone()
    return int(n), int(s), int(x)


def fingerprint_str(fp) -> str:
    n, s, x = fp
    return f"rows={n:,} sum64=0x{s:016x} xor64=0x{x:016x}"
