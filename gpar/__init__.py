#-----------------------------------------------
#  Author: Guanning Pang
#-----------------------------------------------
"""
gPar: A Python Toolbox for array processing
================================================
"""

# python 2 and 3 compatibility imports 

from __future__ import print_function, absolute_import, unicode_literals
from __future__ import with_statement, nested_scopes, division, generators


# General imports
import os
import inspect
import logging
import logging.handlers
import sys
from importlib import reload

# gPar import
import gpar
import gpar.getdata
import gpar.arrayProcess
import gpar.build
import gpar.postProcess
import gpar.util

logging.basicConfig()

version = os.path.join(os.path.dirname(gpar.__file__), 'version.py')

with open(version) as vfil:
	__version__ = vfil.read().strip()


maxSize = 10 * 1024* 1024 # max size log file can be in bytes (10 mb default)
verbose = True # set to false to avoid printing to screen
makeLog = True # set to false to not make log file

def setLogger(fileName='gpar_log.log',deleteOld=False):
	"""
	Function to set up the logger used across Detex

	Parameters
	----------
	fileName: str
		Path to log file to be create
	deleteOld: bool
		If True, delete any file of fileName if exists
	"""

	reload(logging) # reload to reconfigure default ipython log

	global makeLog
	makeLog = True
	cwd = os.getcwd()
	fil = os.path.join(cwd, fileName)
	if os.path.exists(fil):
		if os.path.getsize(fil) > maxSize:
			print ('old log file %s exceeds size limit, deleting' % fil)
			# os.path.remove(fil)
			os.remove(fil)
		elif deleteOld:
			os.path.realpath(fil)
	fh = logging.FileHandler(fil)
	fh.setLevel(logging.DEBUG)
	fmat = '%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s'
	formatter = logging.Formatter(fmat)
	fh.setFormatter(formatter)
	global logger
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)
	logger.addHandler(fh)
	lpath = os.path.abspath(fileName)
	logger.info('Startting logging, path to log file: %s' % fil)

# define basic log function
def log(name, msg, level='info', pri=False, close=False, e=None):
	"""
	Function to log important events as gpar runs
	Parameters:
		name: the __name__ statement
			should always be set to __name__ from the log call. This will enable inspect to 
			trace back to where the call initially came from 
			and record it in the log
		msg: str
			A message to log
		level: str
			level of event (info, debug, warning, critical, or error)
		pri: bool
			If True, print msg to screen without entire log info
		close: bool
			if Ture close the logger so the log file can be opened
		e: Exception class or None
			If level == "error" and Exception if raised, e if the type of exception
	"""

	if not verbose:
		pri = False

	cfun = inspect.getouterframes(inspect.currentframe())[1][0].f_code.co_name
	log = logging.getLogger(name+'.'+cfun)
	if level == 'info' and makeLog:
		log.info(msg)
	elif level == 'debug' and makeLog:
		log.debug(msg)
	elif level == 'critical' and makeLog:
		log.critical(msg)
	elif level == 'warning' and makeLog:
		log.warning(msg)
	elif level == 'error':
		if makeLog:
			log.error(msg)
		if makeLog: 
			closeLogger()
		if e is None:
			e = Exception
		raise e(msg)
	else:
		if makeLog:
			raise Exception('level input not understood, acceptable values are' 
                        ' "debug","info","warning","error","critical"')

	if pri:
		print (msg)
	if close and makeLog:
		closeLogger()


def closeLogger():

	handlers = logger.handlers[:]
	for handler in handlers:
		handler.close()
		logger.removeHandler(handler)


def deb(*varlist):

	global de
	de = varlist
	sys.exit(1)

if makeLog:
	setLogger()





