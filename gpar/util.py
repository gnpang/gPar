import pandas as pd
import numpy as np
import os
import sys
import time
import obspy
import gpar

def readList(flist, list_type='array',sep='\s+'):
	"""
	Function to read a key files and performs checks for required columns parameters
	Parameters:
		flist: str or pandas dataframe
			A path to the key file or DataFrame itself
		list_type: str
			"array" for array list, "event" for earthquake list, "coda" for coda strip or 'twoline' for twoline strip
		sep: str
			seperate symbol for file reading
	"""

	read_array = set(['NAME','LAT','LON'])
	read_eq = set(['TIME','LAT','LON','DEP','Mw','Del','Az','Baz','BB','Rayp','DIR'])
	read_coda = set(['TIME','LAT','LON','DEP','Mw','Del','Az','Baz','BB','Rayp','RES','lnA','B','C'])
	read_two = set(['TIME','LAT','LON','DEP','Mw','Del','Az','Baz','BB','Rayp','RES','m1','b1','m2','b2'])

	req_type = {'array':read_array,'event':read_eq,'coda':read_coda,'towline':read_two}

	list_types = req_type.keys()

	if list_type not in list_types:
		msg = ('Not valid list type, choose from %s' % list_types)
		gpar.log(__name__,msg,level='error',e='ValueError',pri=True)

	if isinstance(flist, str):
		if not os.path.exists(flist):
			msg = ('%s does not exists' % flist)
			gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
		else:
			df = pd.read_csv(flist,sep=sep)
	elif isinstance(flist, pd.DataFrame):
		df = flist
	else:
		msg = ('Not supported data type for flist')
		gpar.log(__name__,msg,level='error',e='ValueError',pri=True)

	if not req_type[list_type].issubset(df.columns):
		msg = ('Required columns not in %s, required columns for %s key are %s'
			   % (df.columns, list_type, req_type[list_type]))
		gpar.log(__name__,msg,level='error',e='ValueError',pri=True)

	tdf = df.loc[:, list(req_type[list_type])]

	condition = [all([x != '' for item, x in row.iteritems()])
				 for ind, row in tdf.iterrows()]
	df = df[condition]

	df.reset_index(drop=True, inplace=True)

	return df

def loadArray(fileName='ILAR.pkl'):
	"""
	Function to uses pandas.read_pickle to load a pickled array instance
	"""

	ar = pd.read_pickle(fileName)
	if not isinstance(ar, gpar.arrayProcess.Array):
		msg = ('%s is not a Array instance' % fileName)
		gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
	return ar