# -*- coding: utf-8 -*-

import logging
from logging.handlers import RotatingFileHandler
import sys
import h5py
import pandas as pd

def configureLogging(logFile, logLevel, console):
    """
    logs the error messages and info to a file
    
    Parameters
    ----------
    mode : int, mode = 0 by default logs the info to file,
        mode = 1 outputs log info to terminal
    """
    logger = logging.getLogger('root')
    FORMAT = "%(levelname)s: %(asctime)s - %(filename)s:%(lineno)s - %(funcName)s : %(message)s"
    logHandler = RotatingFileHandler(logFile, maxBytes=1024*1024*10, backupCount=20)
    logHandler.setLevel(logging.INFO)
    logger.addHandler(logHandler)

    if(console):
        logging.basicConfig(stream=sys.stdout, level= logLevel, format=FORMAT)
    else:
        logging.basicConfig(level= logLevel?logLevel:"DEBUG", format=FORMAT)
    return logger


def toPickle(dataset, dataSetType):
    # List all groups
    data = h5py.File("dataset/"+dataset+"_"+dataSetType+".hdf5",'r')
    data.visit(print)

    mydf = pd.DataFrame(list(data['functionSource']))
    mydf['CWE-119']=list(data['CWE-119']); 
    mydf['CWE-120']=list(data['CWE-120']); 
    mydf['CWE-469']=list(data['CWE-469']); 
    mydf['CWE-476']=list(data['CWE-476']); 
    mydf['CWE-other']=list(data['CWE-other'])
    
    mydf.rename(columns={0:'functionSource'},inplace=True)
    mydf.iloc[0:5,0:]
    mydf.to_pickle("pickle_file/"+dataset+"_"+dataSetType+".pickle")

def convert2Pickle(dataset):
    # dataset has 3 parts
    toPickle(dataset, "train");
    toPickle(dataset, "validate");
    toPickle(dataset, "test");