import pandas as pd
import numpy as np
import os
import sys
import time
import obspy
import gpar
import re

if sys.version_info.major == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest

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

	read_array = set(['NAME','LAT','LON','CODE'])
	read_eq = set(['TIME','LAT','LON','DEP','Mw','Del','Az','Baz','BB','Rayp','DIR'])
	read_db = set(['DoubleID', 'DIR1', 'TIME1', 'LAT1', 'LON1', 'DEP1', 'M1', 'DIR2', 'TIME2', 'LAT2', 'LON2', 'DEP2', 'M2'])
	read_coda = set(['TIME','LAT','LON','DEP','Mw','Del','Az','Baz','BB','Rayp','RES','lnA','B','C'])
	read_two = set(['TIME','LAT','LON','DEP','Mw','Del','Az','Baz','BB','Rayp','RES','m1','b1','m2','b2'])

	req_type = {'array':read_array,'event':read_eq,'coda':read_coda,'towline':read_two, 'doublet':read_db}

	list_types = req_type.keys()

	if list_type not in list_types:
		msg = ('Not valid list type, choose from %s' % list_types)
		gpar.log(__name__,msg,level='error',e='ValueError',pri=True)

	if isinstance(flist, str):
		if not os.path.exists(flist):
			msg = ('%s does not exists' % flist)
			gpar.log(__name__,msg,level='error',e=ValueError,pri=True)
		else:
			df = pd.read_csv(flist,sep=sep)
	elif isinstance(flist, pd.DataFrame):
		df = flist
	else:
		msg = ('Not supported data type for flist')
		gpar.log(__name__,msg,level='error',pri=True)

	if not req_type[list_type].issubset(df.columns):
		msg = ('Required columns not in %s, required columns for %s key are %s'
			   % (df.columns, list_type, req_type[list_type]))
		gpar.log(__name__,msg,level='error',e=ValueError,pri=True)

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

def readNDK(fileName='gcmt_1996_2018.ndk'):
	"""
	A simple function to read in a ndk file and pharse data into a dataframe
	For more detail see https://docs.obspy.org/master/_modules/obspy/io/ndk/core.html#_read_ndk
	"""
	ndkDf = pd.DataFrame()
	with open(fileName, "rt") as fp:
		data = fp.read()
	def lines_iter():
		prev_line = -1
		while True:
			next_line = data.find("\n", prev_line + 1)
			if next_line < 0:
				break
			yield data[prev_line+1 : next_line]
			prev_line = next_line
		if len(data) > prev_line + 1:
			yield data[prev_line+1 :]

	for _i, lines in enumerate(zip_longest(*[lines_iter()] * 5)):
		if None in lines:
			msg = "Skipped last %i lines. Not a multiple of 5 lines."%(lines.count(None))
			gpar.log(__name__, msg, level='warning', pri=True)
			continue
		try:
			record = _read_lines(*lines)
			ndkDf = ndkDf.append(record, ignore_index=True)
		except:
			msg = "Could not parse event %i. Will be skipped. Lines of the event:\n\t%s\n"%(_i+1, "\n\t".join(lines))
			gpar.log(__name__,msg,level='warning', pri=True)
	ndkDf = ndkDf.astype({"exp":int})

	return ndkDf

def _read_lines(line1, line2, line3, line4, line5):
	# First line: Hypocenter line
    # [1-4]   Hypocenter reference catalog (e.g., PDE for USGS location,
    #         ISC for #ISC catalog, SWE for surface-wave location,
    #         [Ekstrom, BSSA, 2006])
    # [6-15]  Date of reference event
    # [17-26] Time of reference event
    # [28-33] Latitude
    # [35-41] Longitude
    # [43-47] Depth
    # [49-55] Reported magnitudes, usually mb and MS
    # [57-80] Geographical location (24 characters)
	rec = {}

	rec["TIME"] = line1[5:15].strip().replace("/","-") + 'T' + line1[16:26]
	rec["DIR"] = line1[5:15].strip().replace("/",".") + '.' + line1[16:24].replace(':','.')
	rec["LAT"] = float(line1[27:33])
	rec["LON"] = float(line1[34:41])
	rec["DEP"] = float(line1[42:47])
    # rec["mb"], rec["MS"] = map(float, line1[48:55].split())

    # Fourth line: CMT info (3)
    # [1-2]   The exponent for all following moment values. For example, if
    #         the exponent is given as 24, the moment values that follow,
    #         expressed in dyne-cm, should be multiplied by 10**24.
    # [3-80]  The six moment-tensor elements: Mrr, Mtt, Mpp, Mrt, Mrp, Mtp,
    #         where r is up, t is south, and p is east. See Aki and
    #         Richards for conversions to other coordinate systems. The
    #         value of each moment-tensor element is followed by its
    #         estimated standard error. See note (4) below for cases in
    #         which some elements are constrained in the inversion.
    # Exponent converts to dyne*cm. To convert to N*m it has to be decreased
    # seven orders of magnitude.
	exponent = int(line4[:2]) - 7
	rec['exp'] = exponent
    # Directly set the exponent instead of calculating it to enhance
    # precision.

	rec["m_rr"], rec["m_rr_error"], rec["m_tt"], rec["m_tt_error"], \
        rec["m_pp"], rec["m_pp_error"], rec["m_rt"], rec["m_rt_error"], \
        rec["m_rp"], rec["m_rp_error"], rec["m_tp"], rec["m_tp_error"] = \
        map(lambda x: float("%s" % (x)), line4[2:].split())

	rec["mxx"] = rec["m_tt"]
	rec["myy"] = rec["m_pp"]
	rec["mxy"] = -rec["m_tp"]
	rec["myz"] = -rec["m_rp"]
	rec["mzx"] = rec["m_rt"]
	rec["mzz"] = rec["m_rr"]

	m = np.zeros([3,3])

	m[0][0] = rec["mxx"]
	m[1][1] = rec["myy"]
	m[2][2] = rec["mzz"]
	m[0][1] = rec["mxy"]
	m[0][2] = rec['mzx']
	m[1][0] = rec["mxy"]
	m[1][2] = rec['myz']
	m[2][0] = rec['mzx']
	m[2][1] = rec['myz']
	K = normalize(m)
	rec["mxx"] = rec["mxx"]/K
	rec["myy"] = rec["myy"]/K
	rec["mzz"] = rec["mzz"]/K
	rec["mxy"] = rec["mxy"]/K 
	rec['mzx'] = rec['mzx']/K
	rec['myz'] = rec['myz']/K

	a = ["m_rr","m_rr_error","m_tt","m_tt_error","m_pp","m_pp_error",
     	"m_rt","m_rt_error","m_rp","m_rp_error","m_tp","m_tp_error"]

	for i in a:
		rec.pop(i)
	scale = float(line5[49:56]) * (10 ** exponent)
	rec['Mw'] = float("{0:.2f}".format(2.0 / 3.0 * (np.log10(scale) - 9.1)))

	return rec

def normalize(X):
	"""
	X is 3 by 3 array
	"""

	eigval, eigvec = np.linalg.eig(X)

	return np.max(eigval)



