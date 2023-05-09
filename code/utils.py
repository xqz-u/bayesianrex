import logging


def write_line(_sequence, _writer, newline=True):
    for i, f in enumerate(_sequence):
        if i < len(_sequence) - 1:
            _writer.write(str(f) + ",")
        else:
            if newline:
                _writer.write(str(f) + "\n")
            else:
                _writer.write(str(f))


def setup_root_logging(level: int = logging.INFO):
    """
    Root entry point to configure logging. Should be called once from the
    program's main entry point.

    :param level: The minimum logging level of displayed messages, defaults to
        logging.INFO.
    """

    logging.basicConfig(
        level=level,
        format="p%(process)s [%(asctime)s] [%(levelname)s]  %(message)s  (%(name)s:%(lineno)s)",
        datefmt="%y-%m-%d %H:%M",
    )
