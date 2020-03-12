import logging

logger = logging.getLogger("GraphRicciCurvature")


def set_verbose(verbose="ERROR"):
    """Set up the verbose level of the GraphRicciCurvature.

    Parameters
    ----------
    verbose: {"INFO","DEBUG","ERROR"}
        Verbose level. (Default value = "ERROR")
            - "INFO": show only iteration process log.
            - "DEBUG": show all output logs.
            - "ERROR": only show log if error happened.
    """
    if verbose == "INFO":
        logger.setLevel(logging.INFO)
    elif verbose == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif verbose == "ERROR":
        logger.setLevel(logging.ERROR)
    else:
        print('Incorrect verbose level, option:["INFO","DEBUG","ERROR"], use "ERROR instead."')
        logger.setLevel(logging.ERROR)
