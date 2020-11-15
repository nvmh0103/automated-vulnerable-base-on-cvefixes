# -*- coding: utf-8 -*-

import logging
from logging.handlers import RotatingFileHandler
import sys

def configureLogging(logLevel, console):
    """
    logs the error messages and info to a file
    
    Parameters
    ----------
    mode : int, mode = 0 by default logs the info to file,
        mode = 1 outputs log info to terminal
    """
    logger = logging.getLogger('root')
    FORMAT = "%(levelname)s: %(asctime)s - %(filename)s:%(lineno)s - %(funcName)s : %(message)s"
    logHandler = RotatingFileHandler('SVD_Approach1.log', maxBytes=1024*1024*10, backupCount=20)
    logHandler.setLevel(logging.INFO)
    logger.addHandler(logHandler)

    if(console):
        logging.basicConfig(stream=sys.stdout, level= logLevel, format=FORMAT)
    else:
        logging.basicConfig(level= logLevel, format=FORMAT)
    return logger