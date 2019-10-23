import logging

logger = logging.getLogger("GraphRicciCurvature")


def set_verbose(verbose="ERROR"):
    if verbose == "INFO":
        logger.setLevel(logging.INFO)
    elif verbose == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif verbose == "ERROR":
        logger.setLevel(logging.ERROR)
    else:
        print('Incorrect verbose level, option:["INFO","DEBUG","ERROR"], use "ERROR instead."')
        logger.setLevel(logging.ERROR)
