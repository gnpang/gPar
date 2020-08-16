"""
@author guanning
"""
# python 2 and 3 compatibility imorts
from __future__ import print_function, absolute_import
from __future__ import with_statement, nested_scopes, division, generators
from six import string_types

import os
import glob
import obspy
import pandas as pd
import numpy as np
import json
import itertools

import gpar
from gpar.util import util


# client imports
import obspy.clients.fdsn
import obspy.clients.neic
import obspy.clients.earthworm

from obspy.core import UTCDateTime
from obspy.core import AttribDict
from obspy.taup import TauPyModel

# default waveform directory
eveDirDefault = 'Data'

PI = np.pi

#ormatkey = {'mseed':'msd', 'pickle':'pkl','sac':'sac'}

def read(path):
	"""
	Function to read a file from a path. If IOError or TypeError simply try appending os.sep to start
	"""

	try:
		st = obspy.core.read(path)
	except (IOError, TypeError):
		try:
			st = obspy.core.read(os.path.join(os.path.sep,path))
		except(IOError, TypeError):
			msg = 'Cannot read %s, the file may be corrupt, skipping it'%(path)
			gpar.log(__name__,msg,level='warn',pri=True)
			return None
	return st


def makeEventList(ndk='gcmt_1976_2017.ndk',array='ILAR',
				Olat=64.7714,Olon=-146.8861,
				mindis=50, maxdis=75, minmag=None,
				model='ak135',phase=['PKiKP']):
	"""
	Function to generate a event list form gcmt.ndk

	Parameters
	-----------
	ndkfile: str
		The path or URL to the ndk file
	array: str
		The name for the array
	Olat: float
		Latitude for the reference point in the array processing
	Olon: float
		Longitude for the reference point in the array processing
	model: str
		model for calculate the ray parameter

	Return
	----------
	catdf: DataFrame
		Catalog informations
	"""
	pd.set_option('precision',3)
	model = TauPyModel(model)
	infocolumn = ['TIME','LAT','LON','DEP','Mw','mxx','mxy','mzx','myy','myz','mzz','DIR']
	if isinstance(ndk, str):
		evedf = util.readNDK(ndk)
	elif isinstance(ndk, obspy.core.event.Catalog):
		eves = ndk
		msg = 'Building dataframe for array %s'%(array)
		gpar.log(__name__, msg, level='info',pri=True)
		for eve in eves:
			origin = eve.origins[0]
			focal = eve.focal_mechanisms[0]
			tensor = focal.moment_tensor.tensor
			mag = eve.magnitudes[0]

			#event origin information
			time = origin.time
			lat = origin.latitude
			lon = origin.longitude
			dep = origin.depth/1000.0

			Mw = mag.mag

			# Event moment tensor
			# original in RPT coordinate, transform to XYZ coordinate
			mxx = tensor.m_tt
			myy = tensor.m_pp
			mxy = -tensor.m_tp
			myz = -tensor.m_rp
			mzx = tensor.m_rt
			mzz = tensor.m_rr

			newRow = {'TIME':time, 'LAT':lat,'LON':lon,'DEP':dep,
					  'Mw':Mw,'DIR':str(time),
					  'mxx':mxx,'mxy':mxy,'mzx':mzx,
					  'myy':myy,'myz':myz,'mzz':mzz}

			evedf = evedf.append(newRow, ignore_index=True)
	elif isinstance(ndk, pd.DataFrame):
		evedf = ndk.copy()
	else:
		msg = 'Not a valid type of ndk, must be a path to a ndk file or a obspy Catalog instance'%(ndk)
		gpar.log(__name__,msg,level='error',pri=True)

	# Great circle distance from events to array
	msg = 'Calculating distance for earthquakes to array %s'%(array)
	gpar.log(__name__, msg, level='info',pri=True)
	great_circle_distance_in_degree, azimuth, bakAzimuth = calc_Dist_Azi(evedf.LAT,evedf.LON,Olat,Olon)

	evedf['DIS'] = np.around(great_circle_distance_in_degree, decimals=2)
	evedf['Az'] = np.around(azimuth, decimals=2)
	evedf['Baz'] = np.around(bakAzimuth, decimals=2)
	msg = 'Selecting distance from %.2f to %.2f'%(mindis, maxdis)
	gpar.log(__name__, msg, level='info',pri=True)
	evedf = evedf[(evedf.DIS >= mindis) & (evedf.DIS <= maxdis)]

	if minmag is not None:
		msg = 'Selecting magnitude larger than %s'%(minmag)
		gpar.log(__name__, msg, level='info',pri=True)
		evedf = evedf[evedf.Mw > minmag]
	# calculate ray parameters using obspy.taup
	evedf.reset_index(drop=True, inplace=True)
	ray = [' ']*len(evedf)
	ray_radian = [' ']*len(evedf)
	takeoff_angle = [' ']*len(evedf)
	for ind, row in evedf.iterrows():
		# print('getting travel time for %dth event in array %s'%(ind, array))
		dep = row.DEP
		dis = row.DIS

		try:
			arr = model.get_travel_times(source_depth_in_km=dep,distance_in_degree=dis,phase_list=phase)[0]
			ray[ind] = float("{0:.3f}".format(arr.ray_param_sec_degree))
			ray_radian[ind] = float("{0:.3f}".format(arr.ray_param))
			takeoff_angle[ind] = float("{0:.3f}".format(arr.takeoff_angle))
		except:
			msg = ('Problem to calculate ray parameter for eve %s to array %s, set to -999'%(row.DIR, array))
			gpar.log(__name__, msg, level='warning',pri=True)
			ray[ind] = -999
			ray_radian[ind] = -999
			takeoff_angle[ind] = -999



	evedf['Rayp'] = ray
	evedf['Rayp_rad'] = ray_radian
	evedf['Angle'] = takeoff_angle

	# Calculater radiation pattern from moment tensor, ray parameter and takeoff angle
	radiation_pattern = _calRadiation(evedf.mxx,evedf.myy,evedf.mzz,
										evedf.mxy,evedf.myz,evedf.mzx,
										evedf.Angle,evedf.Az)

	evedf['BB'] = np.around(radiation_pattern, decimals=2)

	#df = evedf.copy()

	cols = ['TIME', 'LAT', 'LON', 'DEP', 'Mw', 'DIS', 'Az', 'Baz','Rayp', 'BB', 'Angle','DIR']
	evedf.drop(columns=['mxx','mxy','mzx','myy','myz','mzz','Rayp_rad','exp'], inplace=True)
	#df.reset_index(inplace=True)
	name = 'eq.'+array + '.list'
	name = os.path.join(array,name)
	evedf.to_csv(name, sep=str('\t'), index=False)

	return evedf



def calc_Dist_Azi(source_latitude_in_deg, source_longitude_in_deg,
				  receiver_latitude_in_deg, receiver_longitude_in_deg):
	"""
	Function to calcualte the azimuth, back azimuth and great circle distance.
	Need to update, has problem to calculate single earthquake
	Parameters:
		source_longitude_in_deg: float or numpy.array
			source longitude
		source_longitude_in_deg: float or numpy.array
			source longitude

		receiver_longitude_in_deg: float
			receiver longitude
		receiver_longitude_in_deg: float
			receiver longitude
	Returns:
		azimuth
		bakAzimuth
		distance
	"""

	source_latitude = source_latitude_in_deg*PI/180.0
	source_longitude = source_longitude_in_deg*PI/180.0
	if isinstance(source_latitude,float):
		n=1
	else:
		n = len(source_latitude)
	receiver_latitude = receiver_latitude_in_deg*PI/180.0
	receiver_longitude = receiver_longitude_in_deg*PI/180.0
	# correct for ellipticity
	source_latitude = np.arctan(0.996647*np.tan(source_latitude))
	receiver_latitude = np.arctan(0.996647*np.tan(receiver_latitude))
	# source vector unnder spherical coordinates
	e1 = np.cos(source_longitude) * np.cos(source_latitude)
	e2 = np.sin(source_longitude) * np.cos(source_latitude)
	e3 = np.sin(source_latitude)

	source_vector = np.transpose(np.array([e1,e2,e3]))
	# receiver vector unnder spherical coordinates
	s1 = np.cos(receiver_longitude) * np.cos(receiver_latitude)
	s2 = np.sin(receiver_longitude) * np.cos(receiver_latitude)
	s3 = np.sin(receiver_latitude)

	receiver_vector = np.array([s1,s2,s3])
	if n == 1:
		great_circle_distance_cos = np.sum(np.multiply(source_vector, receiver_vector))
	else:
		great_circle_distance_cos = np.sum(np.multiply(source_vector, receiver_vector),axis=1)

	sum_vector = np.add(source_vector, receiver_vector)
	sub_vector = np.subtract(receiver_vector,source_vector)
	if n > 1:
		great_circle_distance_sin = np.sqrt(np.sum(sum_vector**2,axis=1))*\
									np.sqrt(np.sum(sub_vector**2,axis=1))/2.0
	else:
		great_circle_distance_sin = np.sqrt(np.sum(sum_vector**2))*\
									np.sqrt(np.sum(sub_vector**2))/2.0

	great_circle_distance_in_radian = np.arctan2(great_circle_distance_sin,great_circle_distance_cos)
	great_circle_distance_in_degree = great_circle_distance_in_radian * 180.0 / PI


	azimuth_sin = np.cos(receiver_latitude)*np.cos(source_latitude)*np.sin(receiver_longitude - source_longitude)
	azimuth_cos = np.sin(receiver_latitude) - great_circle_distance_cos * np.sin(source_latitude)

	azimuth_in_radian = np.arctan2(azimuth_sin,azimuth_cos)
	azimuth_in_degree = azimuth_in_radian * 180.0 / PI

	if isinstance(azimuth_in_degree, float):
		if azimuth_in_degree < 0:
			azimuth_in_degree = azimuth_in_degree + 360.0
	else:
		for ind, azi in enumerate(azimuth_in_degree):
			if azi < 0:
				azi = azi + 360.0
				azimuth_in_degree[ind] = azi

	bakAzimuth_sin = -np.cos(receiver_latitude) * np.cos(source_latitude) * np.sin(receiver_longitude - source_longitude)
	bakAzimuth_cos = np.sin(source_latitude) - great_circle_distance_cos * np.sin(receiver_latitude)

	bakAzimuth_in_radian = np.arctan2(bakAzimuth_sin,bakAzimuth_cos)
	bakAzimuth_in_degree = bakAzimuth_in_radian * 180.0 / PI

	if isinstance(bakAzimuth_in_degree, float):
		if bakAzimuth_in_degree < 0:
			bakAzimuth_in_degree = bakAzimuth_in_degree + 360.0
	else:
		for ind, bakazi in enumerate(bakAzimuth_in_degree):
			if bakazi < 0:
				bakazi = bakazi + 360.0
				bakAzimuth_in_degree[ind] = bakazi



	return great_circle_distance_in_degree,azimuth_in_degree,bakAzimuth_in_degree


def _calRadiation(mxx,myy,mzz,mxy,myz,mzx,takeoff,azimuth):

	takeoff = takeoff * PI / 180.0
	azimuth = azimuth * PI / 180.0

	val = np.sin(takeoff)**2*\
		  (np.cos(azimuth)**2*mxx+np.sin(2*azimuth)*mxy+\
		  np.sin(azimuth)**2*myy-mzz) +\
		  mzz + 2.0*np.sin(takeoff)*np.cos(takeoff)*\
		  (np.cos(azimuth)*mzx+np.sin(azimuth)*myz)

	return val

def quickFetch(fetch_arg,**kwargs):
	"""
	Instantiate a DataFetcher using as little information as possible

	"""

	if isinstance(fetch_arg, DataFetcher):
		dat_fet = fetch_arg
	elif isinstance(fetch_arg, string_types):
		if fetch_arg in DataFetcher.subMethods:
			if fetch_arg == 'dir':
				msg = 'If using method dir, please pass a path to directory'
				gpar.log(__name__, msg, level='error', pri=True)
			dat_fet = DataFetcher(fetch_arg, **kwargs)
		else:
			if not os.path.exists(fetch_arg):
				msg = 'Directory %s does not exist' % fetch_arg
				gpar.log(__name__, msg, level='error', pri=True)
			dat_fet = DataFetcher(method='dir', arrayName=fetch_arg, **kwargs)
	else:
		msg = 'Input not understand'
		gpar.log(__name__, msg, level='error', pri=True)
	return dat_fet

def makeDataDirectory(arraylist='array.list', fetch='IRIS',
					  timeBeforeOrigin=0, timeAfterOrigin=3600,
					  buildArray=False, outsta=True, minlen=1500.0,
					  **kwargs):
	ardf = util.readList(arraylist, list_type='array', sep='\s+')

	if isinstance(fetch, gpar.getdata.DataFetcher):
		fetcher = fetch
	else:
		fetcher = gpar.getdata.DataFetcher(fetch, timeBeforeOrigin=timeBeforeOrigin, timeAfterOrigin=timeAfterOrigin, minlen=minlen)

	for ind, row in ardf.iterrows():
		edf, stadf = fetcher.getEqData(row,channel='*')
		edf.dropna(inplace=True)
		edf.reset_index(drop=True, inplace=True)
		arDir = os.path.join(row.NAME, 'Data')
		if not os.path.isdir(arDir):
			os.mkdir(arDir)
		for ind, eve in edf.iterrows():
			eDir = os.path.join(arDir, eve.DIR)
			if not os.path.isdir(eDir):
				os.mkdir(eDir)
			_id = eve.DIR[:-6] + '.' + row.NETWORK
			st = eve.Stream
			for tr in st:
				sacname = _id + '.' + tr.stats.station + '.' + tr.stats.channel + '.sac'
				sacname = os.path.join(eDir, sacname)
				tr.write(sacname, format='SAC')

		if buildArray :
			refpoint = [row.LAT, row.LON]
			array = gpar.arrayProcess.Array(row.NAME, refpoint, edf, coordsys=kwargs['coordsys'], phase=kwargs['phase'])
			msg = ('using default parameters sll_x=-15, sll_y=-15, sl_s=0.1,grdpts_x=301, grdpts_y=301, unit="deg" for time table')
			gpar.log(__name__,msg,level='info',pri=True)
			array.getTimeTable()
			array.write()
		if outsta:
			stafile = os.path.join(row.NAME, row.NAME+'.sta')
			stadf.to_csv(stafile, sep=str('\t'), index=False)

class DataFetcher(object):

	"""
	Class to hangle data acquisition
	Support downloaded sac format data only in this version
	"""
	subMethods = ['dir', 'iris']
	def __init__(self, method, arrayName=None, client=None, removeResponse=False,
		         prefilt=None, timeBeforeOrigin=0.0, timeAfterOrigin=3600.0, minlen=1500,
		         checkData=True):
		# self.arrayName = arrayName
		self.__dict__.update(locals())
		self._checkInputs()

	def _checkInputs(self):
		if self.arrayName is not None and isinstance(self.arrayName, string_types):
			self.arrayName = self.arrayName.upper()
		if not isinstance(self.method, string_types):
			msg = 'method must be a string. options:\n%s'%self.supMethods
			gpar.log(__name__,msg,level='error',e=ValueError)
		self.method = self.method.lower()
		if not self.method in DataFetcher.subMethods:
			msg = 'method %s not support. Options:\n%s'%(self.metho,self.supMethods)
			gpar.log(__name__,msg,level='error',e=ValueError)
		if self.method == 'dir':

			dirPath = glob.glob(self.arrayName)
			if len(dirPath) <1:
				msg = ('directory %s not found make sure path is correct' % self.arrayName)
				gpar.log(__name__,msg,level='error',e=IOError)
			else:
				self.directory = dirPath[0]
				self._getStream = _loadDirectoryData
		elif self.method == 'iris':
			self.client = obspy.clients.fdsn.Client('IRIS')
			self._getStream = _loadFromFDSN

	def getEqData(self, ar, timebefore=None, timeafter=None, phase=['PKiKP'], minlen=None, mode='eq',verb=False,**kwargs):
		if timebefore is None:
			timebefore = self.timeBeforeOrigin
		if timeafter is None:
			timeafter = self.timeAfterOrigin
		if minlen is None:
			minlen = self.minlen

		net = ar.NETWORK
		sta = ar.Station
		chan = ar.Channel.split('-')
		arDir = os.path.join(ar.NAME, 'Data')

		eqlist = os.path.join(ar.NAME, mode+'.'+ar.NAME+'.list')
		if not os.path.exists(eqlist):
			msg = ('Earthquake list for array %s is not exists, building from ndk file first' % ar.NAME)
			gpar.log(__name__,msg,level='warning',pri=True)
			if 'ndkfile' not in kwargs.keys():
				msg = ('input the ndkfile and related parameters (see getdata.makeEventList) for building earthquake list')
				gpar.log(__name__, msg, level='warning', pri=True)
				return None
			if not os.path.isfile(ndkfile):
				msg = ('ndk file does not exist')
				gpar.log(__name__,msg, level='warning', pri=True)
				return None
		if mode == 'eq':
			eqdf = util.readList(eqlist,list_type='event', sep='\s+')
		elif mode == 'db':
			eqdf = util.readList(eqlist,list_type='doublet', sep='\s+')
		ndf, stadf = self.getStream(ar, eqdf, timebefore, timeafter, net, sta, chan, verb=verb,
									loc='??', minlen=minlen, mode=mode,channel=kwargs['channel'])

		return ndf, stadf

	def getStream(self, ar, df, stime, etime, net, staele, chan, verb, 
				loc='??', minlen=1500, mode='eq',channel='Z'):
		if not isinstance(chan, (list, tuple)):
			if not isinstance(chan, string_types):
				msg = 'chan must be a string or list of string'
				gpar.log(__name__, msg, level=error, pri=True)
			chan = [chan]
		if self.method == 'dir':
			df, stadf = self._getStream(ar.NAME, df, mode,minlen,verb,channel)
		else:
			stadf = pd.DataFrame(columns=['STA','LAT','LON'])
			client = self.client
			stream = [''] * len(df)
			for ind, row in df.iterrows():
				time = UTCDateTime(row.TIME)
				start = time - stime
				end = time + etime
				st = self._getStream(self, start, end, net, staele, chan, loc)
				st = _checkData(st,minlen)
				if st is None or len(st) < 1:
					stream[ind] = pd.NaT
				else:
					nst = obspy.Stream()
					for tr in st:
						sta = tr.stats.station
						station = client.get_stations(network=net, station=sta).networks[0].stations[0]
						latitude = station.latitude
						longitude = station.longitude
						sac = AttribDict({'b':-stime, 'e':etime, 'o':0,
										  'evla':row.LAT, 'evlo': row.LON,
										  'delta': tr.stats.delta, 'npts': tr.stats.npts,
										  'stla': latitude, 'stlo':longitude,
										  'iztype': 11, 'lcalda': True, 'knetwk': net,
										  'kcmpnm':tr.stats.channel,'kstnm':sta,
										  'nzyear':time.year, 'nzjday':time.julday,
										  'nzhour':time.hour, 'nzmin':time.minute,
										  'nzsec':time.second, 'nzmsec': time.microsecond/1000})
						tr.stats.sac = sac
						nst.append(tr)
						stadf = _checkSta(tr, stadf)
					stream[ind] = nst

			df['Stream'] = stream

		return df, stadf


# Functions to get stream data based on selected method

def _loadDirectoryData(arrayName, df, mode,minlen,verb,channel='Z'):

	staDf = pd.DataFrame(columns=['STA', 'LAT', 'LON'])
	if mode == 'eq':
		stream = [''] * len(df)
	elif mode == 'db':
		stream1 = [''] * len(df)
		stream2 = [''] * len(df)
	else:
		msg = ('Not a valid option for earthquake type, choose eq or db')
		gpar.log(__name__, msg, level='error', pri=True)
	for ind, eve in df.iterrows():
		if mode == 'eq':
			_id = eve.DIR[:-6]
			if verb:
				print('loading data %s:%s'%(arrayName, eve.DIR))
			sacfiles = os.path.join(arrayName, 'Data', eve.DIR, _id+'*'+channel+'.sac')
			try:
				st = read(sacfiles)
			except:
				msg = 'data from %s in array %s is not found, skipping' % (arrayName,eve.DIR)
				gpar.log(__name__,msg,level='warning',pri=True)
				stream[ind] = pd.NaT
				continue
			for tr in st:
				if not hasattr(tr.stats, 'sac'):
					msg = ("Trace for %s in station %s doesn's have SAC attributes, removing" % (eve, tr.stats.station))
					st.remove(tr)
					gpar.log(__name__,msg,level='warning',e=ValueError)
					continue
				if not hasattr(tr.stats.sac, 'stla') or not hasattr(tr.stats.sac, 'stlo'):
					msg = ("Trace for %s in station %s doesn's have station information, removing" % (eve.DIR, tr.stats.station))
					st.remove(tr)
					gpar.log(__name__,msg,level='warning',e=ValueError)
					continue
				data = tr.data
				if np.std(data) < 0.1:
					msg = ("Trace data for %s in station %s has std smaller than 0.1, removing" % (eve.DIR, tr.stats.station))
					st.remove(tr)
					gpar.log(__name__,msg,level='warning',e=ValueError)
					continue

				staDf = _checkSta(tr, staDf)

			if len(st) == 0:
				msg = ("Waveforms for event %s have problem" % eve.DIR)
				gpar.log(__name__,msg,level='warning')
				st = None
			st = _checkData(st,minlen)
			stream[ind] = st
		elif mode == 'db':
			if verb:
				print('loading data %s: %s'%(arrayName, eve.DoubleID))
			_id1 = eve.DIR1[:-6]
			_id2 = eve.DIR2[:-6]
			sacfile1 = os.path.join(arrayName, 'Data', eve.DIR1, _id1+'*'+channel+'.sac')
			sacfile2 = os.path.join(arrayName, 'Data', eve.DIR2, _id2+'*'+channel+'.sac')
			try:
				st1 = read(sacfile1)
			except:
				msg = 'data from %s in array %s is not found, skipping' % (arrayName,eve.DIR1)
				gpar.log(__name__,msg,level='warning',pri=True)
				stream1[ind] = pd.NaT
				continue
			try:
				st2 = read(sacfile2)
			except:
				msg = 'data from %s in array %s is not found, skipping' % (arrayName,eve.DIR2)
				gpar.log(__name__,msg,level='warning',pri=True)
				stream2[ind] = pd.NaT
				continue
			st1 = _checkData(st1,minlen)
			st2 = _checkData(st2,minlen)
			if st1 != None:
				for tr in st1:
					staDf = _checkSta(tr,staDf)
			if st2 != None:
				for tr in st2:
					staDf = _checkSta(tr,staDf)
			stream1[ind] = st1
			stream2[ind] = st2
	if mode == 'eq':
		df['Stream'] = stream
	elif mode == 'db':
		df['ST1'] = stream1
		df['ST2'] = stream2
	df.dropna(inplace=True)
	df.reset_index(drop=True, inplace=True)

	return df, staDf

def _loadFromFDSN(fet, start, end, net, sta, chan, loc):

	client = fet.client

	if not isinstance(chan, string_types):
		chan = ','.join(chan)
	else:
		if '-' in chan:
			chan = ','.join(chan.split('-'))
	if '-' in sta:
		sta = ','.join(sta.split('-'))
	try:
		st = client.get_waveforms(net, sta, loc, chan, start, end)
	except:
		msg = ('Could not fetch data on %s in channel %s from %s to %s' %
				(net+'.'+sta, chan, start, end))
		gpar.log(__name__, msg, level='warning', pri=True)
		st = None

	return st

def _checkData(st, minlen):

	if st is None or len(st)<1:
		return None

	sample_rates = []
	for tr in st:
		stats = tr.stats
		lasttime = stats.npts * stats.delta
		sample = stats.sampling_rate
		if sample not in sample_rates:
			sample_rates.append(sample)
		if lasttime < minlen:
			msg = ('Trace in station %s starting from %s is shrter than require, removing'%(stats.station, stats.starttime))
			gpar.log(__name__, msg, level='info')
			st.remove(tr)
		if np.std(tr.data) < 0.1:
			msg = ("Trace data for %s in station %s has std smaller than 0.1, removing" % (stats.starttime, tr.stats.station))
			st.remove(tr)
			gpar.log(__name__,msg,level='warning',e=ValueError)
			continue
	if len(st) == 0:
		st = None
	if len(sample_rates) > 1 and st != None:
		resample = np.min(sample_rates)
		msg ("Traces have different sampling rate %s\nnow resample them to the samllest sample %s"%(sample_rates, resample))
		gpar.log(__name__, msg, level='info')
		st.resample(sampling_rate)

	return st

def _checkSta(tr, stadf):

	stats = tr.stats
	sta = stats.station
	tmp = stadf[stadf.STA == sta]
	if len(tmp) == 0:
		lat = stats.sac.stla
		lon = stats.sac.stlo
		newSta = {'STA':sta,'LAT':lat, 'LON':lon}
		stadf = stadf.append(newSta, ignore_index=True)

	return stadf
