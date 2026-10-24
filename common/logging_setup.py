import logging
import os
import sys

# Every module logs under "algo.*" so one set of handlers captures all output.
ROOT_LOGGER_NAME = "algo"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_logging(default_log_name: str) -> None:
    log_file = os.getenv("LOG_FILE", os.path.join(PROJECT_ROOT, default_log_name))
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(ROOT_LOGGER_NAME)
    log.setLevel(logging.INFO)
    log.handlers.clear()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    log.addHandler(stream)

    try:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError as e:
        log.warning("Could not open log file %s: %s", log_file, e)
