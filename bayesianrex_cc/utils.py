"""General utility routines."""
import logging


def setup_root_logging(level: int = logging.INFO):
    """
    Root entry point to configure logging, called once from the main entry point.

    :param level: The minimum logging level of displayed messages, defaults to
        logging.INFO (everything below is suppressed).
    """
    logging.basicConfig(
        level=level,
        format="p%(process)s [%(asctime)s] [%(levelname)s]  %(message)s  (%(name)s:%(lineno)s)",
        datefmt="%y-%m-%d %H:%M",
    )
