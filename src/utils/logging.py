import logging


_logger = None


def get_logger(name: str = "nyc"):
    global _logger
    if _logger:
        return _logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    _logger = logger
    return logger