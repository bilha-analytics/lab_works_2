'''
author: bg
goal: proper and consistent logging
type: util 
how: use std logging module, format to liking, 
learn: https://www.loggly.com/ultimate-guide/python-logging-basics/ , https://docs.python.org/3.5/howto/logging-cookbook.html , 
refactors: Do we want this as a class? What form; Singelton? 
'''

import os
import sys, traceback 
from datetime import datetime
import logging 
from termcolor import colored
os.system('color') 

DEFAULT_LOGGING_LEVEL = logging.NOTSET 

LOGGER = None 
APP_NAME = None 

def startLogger(name, level=DEFAULT_LOGGING_LEVEL):
    '''
    Input: 
        name: Name of logger, say app name 
        level: Logging level. Default is everything @ NOTSET 
    Return: None
    TODO: review at module Vs app level usage @ LOGGER object instance + basicConfig  
    '''
    global LOGGER, APP_NAME 
    
    APP_NAME = name #"UNNAMED" if name is None else name 
    
    if LOGGER is None:
        LOGGER = logging.getLogger( APP_NAME )
        LOGGER.addHandler( logging.StreamHandler() )
        logging.basicConfig()  
    setLogLevel( level ) 
    print("Logger is started")
        

def setLogLevel( level=DEFAULT_LOGGING_LEVEL):
    '''
    Input: level: Logging level. Default is everything @ NOTSET
    Return: None
    '''
    if LOGGER is not None:
        LOGGER.setLevel( level ) 


def log(src, msg, ltype=logging.INFO, appName='zmoi'): 
    '''
    For now using once instance for entire app and it's modules. Doing some name_formating hack
    The only call needed to get things working
    Input: 
        src: App or module making the request 
        msg: Message to log. Can be any object type 
        type: level 
        appName: overarching app name. Very first time used
    Return: None
    ''' 
    if LOGGER is None:
        startLogger(appName) 

    logit = {
        logging.DEBUG : LOGGER.debug,
        logging.WARNING : LOGGER.warning,
        logging.ERROR : LOGGER.error,
        logging.CRITICAL : LOGGER.critical ,
    } #INFO @ default;all else

    colorit = {
        logging.WARNING : 'yellow',
        logging.ERROR : 'red',
        logging.CRITICAL : 'red' ,
    }# default = blue 

    
    nameit = {
        logging.WARNING : "WARNING ",
        logging.ERROR : "ERROR   ",
        logging.CRITICAL : "CRITICAL" ,
    } # default = INFOR

    nm = nameit.get( ltype, "INFOR   ") if APP_NAME is None else ""
    msg_str = "{}: {} [{}] {}".format( 
                                    nm, 
                                    datetime.now(), 
                                    colored(src, colorit.get(ltype, 'blue') ), 
                                    msg )

    log_ = logit.get(ltype, LOGGER.info)
    log_(msg_str) 
    print(msg_str)


def logError(src, msg):
    '''
    Specific formatting for errors and provide stack trace on exceptions etc 
    Input: 
        src : source of log request
        msg : message to go with exception output 
    Return: None
    '''
    e = sys.exc_info()[0] 
    log("{}".format(src), "{}: {}".format(msg, e), ltype=logging.ERROR ) 
    print( traceback.format_exc() ) 
     

if __name__ == "__main__":
    log(__name__, "Trying out the logger")  
    log("Main.MyModule", "The quick brown fox jumped over the lazy dogs!", logging.WARN)
    log( __name__, "Yet another message here with a very very very very ong long long string string", logging.ERROR)
    log( __name__, "Yet another message here", logging.CRITICAL)