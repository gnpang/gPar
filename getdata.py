"""
@author guanning
"""
# python 2 and 3 compatibility imorts
from __future__ import print_function, absolute_import, unicode_literals
from __future__ import with_statement, nested_scopes, division, generators
from six import string_types

import os
import glob
import obspy
import pandas as pd 
import numpy as np 
import json
import itertools

from six import string_types

import gpar


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
	Function to read a file form a path. If IOError or TypeError simply try appending os.sep to start
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


def makeEventList(ndkfile='gcmt_1976_2017.ndk',array='ILAR',Olat=64.7714,Olon=-146.8861,model='ak135',phase='PKiKP'):
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
	eves = obspy.read_events(ndkfile)
	evedf = pd.DataFrame(columns=infocolumn)

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

	# Great circle distance from events to array
	great_circle_distance_in_degree, azimuth, bakAzimuth = calc_Dist_Azi(evedf.LAT,evedf.LON,Olat,Olon)

	evedf['Del'] = great_circle_distance_in_degree
	evedf['Azi'] = azimuth 
	evedf['Baz'] = bakAzimuth

	# calculate ray parameters using obspy.taup
	ray = [' ']*len(evedf)
	ray_radian = [' ']*len(evedf)
	takeoff_angle = [' ']*len(evedf)
	for ind, row in evedf.iterrows():
		dep = row.DEP
		dis = row.Del

		arr = model.get_travel_times(source_depth_in_km=dep,distance_in_degree=dis,phase_list=phase)[0]
		ray[ind] = arr.ray_param_sec_degree
		ray_radian[ind] = arr.ray_param
		takeoff_angle = arr.takeoff_angle

	evedf['Rayp'] = ray
	evedf['Rayp_rad'] = ray_radian
	evedf['Angle'] = takeoff_angle

	# Calculater radiation pattern from moment tensor, ray parameter and takeoff angle
	radiation_pattern = __calRadiation(evedf.mxx,evedf.myy,evedf.mzz,
										evedf.mxy,evedf.myz,evedf.mzx,
										evedf.Angle,evedf.Rayp_rad)

	evedf['BB'] = radiation_pattern

	df = evedf.copy()

	df.drop(columns=['mxx','mxy','mzx','myy','myz','mzz','Rayp_rad'])
	name = 'eq.'+array + '.list'
	name = os.path.join(array,name)
	df.to_csv(name, sep='\t',index=False)

	return evedf, df


	
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


def __calRadiation(mxx,myy,mzz,mxy,myz,mzx,takeoff,azimuth):

	takeoff = takeoff * PI / 180.0
	azimuth = azimuth * PI / 180.0

	val = np.sin(takeoff)**2*\
		  (np.cos(azimuth)*mxx+np.sin(2*azimuth)*mxy+\
		  np.sin(azimuth)**2*myy-mzz) +\
		  mzz + 2.0*np.sin(takeoff)*cos(takeoff)*\
		  (np.cos(azimuth)*mzx+np.sin(azimuth)*myz)

	return val

def quickFetch(fetch_arg,**kwargs):
	"""
	Instantiate a DataFetcher using as little information as possible

	"""

	if not isinstance(fetch_arg, string_types):
		msg = 'Input must be a string, please try again'
		gpar.log(__name__, msg, level='error',pri=True)
	dat_fet = DataFetcher(fetch_arg,**kwargs)
	return dat_fet


class DataFetcher(object):

	"""
	Class to hangle data acquisition
	Support downloaded sac format data only in this version
	"""

	def __init__(self,arrayName):
		self.arrayName = arrayName
		self._checkInputs()

	def _checkInputs(self):
		if not isinstance(self.arrayName, string_types):
			msg = 'arrayName must be a string. only data dir now'
			gpar.log(__name__,msg,level='error',e=ValueError)
		self.arrayName = self.arrayName.upper()

		dirPath = glob.glob(self.arrayName)
		if len(dirPath) <1:
			msg = ('directory %s not found make sure path is correct' % self.arrayName)
			gpar.log(__name__,msg,level='error',e=IOError)
		else:
			self.directory = dirPath[0]

	def getStream(self,eve):
		"""
		Function for loading data from directory, sac format only
		"""

		ID = eve[:-6]
		datafile = os.path.join(self.arrayName,'Data',eve,ID+'*.sac')
		if len(datafile) < 1:
			msg = 'data from %s in array %s is not found' % (self.arrayName,eve)
			gpar.log(__name__,msg,level='warning',pri=True)
			return None
		st = read(datafile)

		# check if contain neccesarry information

		for tr in st:
			try:
				sac = tr.stats.sac
			except ValueError:
				msg = ("Trace for %s in station %s doesn's have SAC attributes, removing" % eve, tr.stats.station)
				st.remove(tr)
				gpar.log(__name__,msg,level='warning',e=ValueError)
				continue
			try:
				stala = tr.stats.sac.stla 
				stalo = tr.stats.sac.stlo 
			except ValueError:
				msg = ("Trace for %s in station %s doesn's have station information, removing" % eve, tr.stats.station)
				st.remove(tr)
				gpar.log(__name__,msg,level='warning',e=ValueError)

		if len(st) == 0:
			msg = ("Waveforms for event %s have problem" % eve)
			gpar.log(__name__,msg,level='warning,e=ValueError')
			return None

		return st




