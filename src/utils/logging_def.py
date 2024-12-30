import logging

# this must be called before any other loggers are instantiated to take effect
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s]  %(message)s')


def get_logger(name: str):
    """
    All modules should use this function to get a logger.
    This way, we ensure all loggers are instantiated after basicConfig() call and inherit the same config.
    """
    return logging.getLogger(name)
