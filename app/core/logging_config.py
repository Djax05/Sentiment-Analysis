import logging


LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT
    )


def get_logger(name: str):
    return logging.getLogger(name)
