from __future__ import print_function, absolute_import
from __future__ import with_statement, nested_scopes, division, generators

import os
import time
import numpy as np 
import pandas as pd 
import obspy 
import scipy
from obspy.core import UTCDateTime
from obspy.core import AttribDict
from obspy.taup import TauPyModel
from obspy.signal.util import util_geo_km
import matplotlib.pyplot as plt
import matplotlib.cm as cm
try:
	import cPickle
except ImportError:
	import pickle as cPickle


PI = np.pi
deg2rad = PI/180.0
rad2deg = 180.0/PI
deg2km = 2*PI*6371.0/360.0

import gpar
from gpar.header import clibarray


##### Beamforming functions and class ######

class Array(object):
	"""
	Array object that contains multiple earthquake objects
	"""

	def __init__(self,arrayName,refPoint,eqDF,staDf, coordsys='lonlat',phase='PKiKP'):
		self.name = arrayName
		self.refPoint = refPoint
		self.coordsys = coordsys
		self.events = [0]*len(eqDF)
		self.getGeometry(staDf,refPoint,coordsys=coordsys)
		for ind, row in eqDF.iterrows():
			self.events[ind] = Earthquake(self, row, phase=phase)

	def getGeometry(self,staDF,refPoint,coordsys='lonlat'):
		"""
		Return nested list containing the array geometry using for waveform shifting

		"""
		# row = eqDF.iloc[0]
		# stream = row.Stream
		# nstat = len(staDF)
		geometry = pd.DataFrame()
		# for i, tr in enumerate(stream):
		# 	if coordsys == 'lonlat':
		# 		sta = tr.stats.station
		# 		lat = tr.stats.sac.stla
		# 		lon = tr.stats.sac.stlo
		# 		dis_in_degree, az,baz = gpar.getdata.calc_Dist_Azi(refPoint[0],refPoint[1],lat,lon)
		# 		dis_in_km = dis_in_degree * deg2km
		# 		X = dis_in_km*np.sin(az*deg2rad)
		# 		Y = dis_in_km*np.cos(az*deg2rad)
		# 		newRow = {'STA':sta,'LAT':lat,'LON':lon,'X':X,'Y':Y}
		# 		geometry = geometry.append(newRow,ignore_index=True)
		# 	elif coordsys == 'xy':
		# 		sta = tr.stats.station
		# 		X = tr.stats.coordinates.x - refPoint[0]
		# 		Y = tr.stats.coordinates.y - refPoint[1]
		# 		newRow = {'STA':sta,'X':X,'Y':Y}
		# 		geometry =geometry.append(newRow,ignore_index=True)
		if coordsys == 'lonlat':
			dis_in_degree, az, baz = gpar.getdata.calc_Dist_Azi(refPoint[0], refPoint[1], staDF.LAT, staDF.LON)
			dis_in_km = dis_in_degree * deg2km
			X = dis_in_km * np.sin(az*deg2rad)
			Y = dis_in_km * np.cos(az*deg2rad)
			staDF['X'] = X
			staDF['Y'] = Y
			geometry = staDF
		elif coordsys == 'xy':
			X = staDF.X - refPoint[0]
			Y = staDF.Y - refPoint[1]
			staDF['X'] = X
			staDF['Y'] = Y
			geometry = staDF
		else:
			msg('Not a valid option, select from lonlat or xy')
			gpar.log(__name__, msg, level='error', pri=True)

		self.geometry = geometry
		
	def getTimeTable(self,sll_x=-15.0,sll_y=-15.0,sl_s=0.1,grdpts_x=301,grdpts_y=301,unit='deg'):
		"""
		Return timeshift table for given array geometry, modified from obsy
		
		"""
		geometry = self.geometry

		self.timeTable = getTimeTable(geometry,sll_x,sll_y,sl_s,grdpts_x,grdpts_y,unit)

	def beamforming(self,filt=[1,2,4,True],starttime=0,winlen=1800.0,
					stack='linear',unit='deg',write=True):
		"""
		Function to do beamforming for all earthquakes in the array
		"""
		for eq in self.events:
			eq.beamforming(geometry=self.geometry,arrayName=self.name,starttime=starttime,
						   winlen=winlen,filt=filt,unit=unit,
						   stack=stack, write=write)

	def slideBeam(self,filt=[1,2,4,True],grdpts_x=301,grdpts_y=301,sflag=2,stack='linear',
					sll_x=-15.0,sll_y=-15.0,sl_s=0.3,refine=True,
					starttime=400.0,endtime=1400.0, unit='deg',
					winlen=2.0,overlap=0.5,write=False, **kwargs):
		
		for eq in self.events:
			eq.slideBeam(geometry=self.geometry,timeTable=self.timeTable,arrayName=self.name,
						 grdpts_x=grdpts_x,grdpts_y=grdpts_y,
					filt=filt, sflag=sflag,stack=stack,
					sll_x=sll_x,sll_y=sll_y,sl_s=sl_s,refine=refine,
					starttime=starttime,endtime=endtime, unit=unit,
					winlen=winlen,overlap=overlap,write=write, **kwargs)

	def vespectrum(self,grdpts=401,
					filt=[1,2,4,True], sflag='log10',stack='linear',
					sl_s=0.1, vary='slowness',sll=-20.0,
					starttime=400.0,endtime=1400.0, unit='deg',
					**kwargs ):
		for eq in self.events:
			eq.vespectrum(geometry=self.geometry,arrayName=self.name,grdpts=grdpts,
					filt=filt, sflag=sflag,stack=stack,
					sl_s=sl_s, vary=vary,sll=sll,
					starttime=starttime,endtime=endtime, unit=unit,
					**kwargs)

	def write(self,fileName=None):
		"""
		Write instance into file (name is the same with array's name)
		"""
		if fileName == None:
			fileName = self.name +'.pkl'
		fileName = os.path.join(self.name, fileName)
		msg = 'writing array instance %s as %s' % (self.name, fileName)
		gpar.log(__name__,msg,level='info',pri=True)
		cPickle.dump(self,open(fileName,'wb'))

class Earthquake(object):
	"""
	Earthquake object
	"""

	def __init__(self, array, row, phase=['PKiKP']):
		"""
		Earthquake basic information including ray parameters for specific phase
		defualt is for PKiKP
		"""
		self.time = UTCDateTime(row.TIME)
		self.ID = row.DIR
		self.lat = row.LAT
		self.lon = row.LON
		self.dep = row.DEP
		self.Mw = row.Mw
		self.Del = row.Del
		self.azimuth = row.Az
		self.bakAzimuth = row.Baz
		self.pattern = row.BB
		self.rayParameter = row.Rayp
		# self.takeOffAngle = row.Angle
		self.phase = phase
		self.stream = row.Stream
		self.ntr = len(row.Stream)
		self.delta = row.Stream[0].stats.delta
		self._checkInputs()

	def _checkInputs(self):
		if not isinstance(self.time, UTCDateTime):
			self.time = UTCDateTime(self.time)
		if not isinstance(self.stream, obspy.core.stream.Stream):
			msg = ('Waveform data for %s is not stream, stop running' % self.ID)
			gpar.log(__name__,msg,level='error',e='ValueError',pri=True)

	def getArrival(self,phase=None, model='ak135'):
		"""
		Function to get theoritcal arrival times for the events

		Parameters:
			phase: str or list. Phase name for travel time calculating
			model: str. Model using in taup.
		"""

		model = TauPyModel(model)
		if phase != None:
			phase = phase
		else:
			phase = self.phase
		arrival = model.get_travel_times(source_depth_in_km=self.dep,distance_in_degree=self.Del,phase_list=phase)[0]
		self.arrival = self.time + arrival.time
		msg = ('Travel time for %s for earthquake %s in depth of %.2f in distance of %.2f is %s' % (phase, self.ID, self.dep, self.Del, self.arrival))
		gpar.log(__name__,msg,level='info',pri=True)

	def beamforming(self, geometry, arrayName, starttime=0.0, winlen=1800.0,
					filt=[1,2,4,True],stack='linear',unit='deg',write=True,**kwargs):
		"""
		Function to get beamforming for the phase in self.phase. Default is for PKiKP
		Only coding linear stacking now
		Parameters:
		-------------------
			geometry: DataFrame that contains geometry infromation for the array, as returns by getGeometry
			arrayName: str, name of array
			starttime: float, startting time for the trace, relative to the original time
			winle: window length in sec for beamforming
			filt: list, filter parameters for waveform filtering
			stack: str, wave to stack shifted wavefrom.
					linear: linear stacking
					psw: phase weight stacking
					root: root mean stack
			unit: str, units for the ray parameter.
				deg: s/degree
				km: s/km
				rad: s/radian
			write: boot, if True write a sac file for beamforming trace, store into the directory where the event waveforms are.
		"""
		# stime = time.time()
		msg = ('Calculating beamforming for earthquake %s' % self.ID)
		gpar.log(__name__,msg,level='info',pri=True)
		tsDF = getTimeShift(self.rayParameter,self.bakAzimuth,geometry,unit=unit)
		self.timeshift = tsDF
		stalist = tsDF.STA
		st = self.stream
		if len(st) < len(tsDF):
			for s in stalist:
				tmp_st = st.select(station=s)
				if len(tmp_st) == 0:
					msg = 'Station %s is missing for event %s in array %s, dropping station'%(s, self.ID, arrayName)
					gpar.log(__name__, msg, level='info', pri=True)
					tsDF = tsDF[~tsDF.STA == s]
		ntr = self.ntr
		delta = self.delta
		# st.detrend('demean')
		st.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
		st.detrend('demean')
		DT = tsDF.TimeShift - (tsDF.TimeShift/delta).astype(int)*delta
		lag = (tsDF.TimeShift/delta).astype(int)+1
		tsDF['DT'] = DT
		tsDF['LAG'] = lag
		npt = int(winlen/delta) + 1
		beamTr = obspy.core.trace.Trace()
		beamTr.stats.delta = self.delta
		if stack == 'psw':
			shiftTRdata = np.empty((ntr,npt),dtype=complex)
		else:
			shiftTRdata = np.empty((ntr,npt))
		for ind, tr in enumerate(st):
			sta = tr.stats.station
			data = tr.data
			if stack == 'root':
				order = kwargs['order']
			elif stack == 'psw':
				order = kwargs['order']
				data = scipy.signal.hilbert(data)
			tr_npt = tr.stats.npts
			tmp_tsDF = tsDF[tsDF.STA==sta].iloc[0]
			sind = int(int(starttime - tr.stats.sac.b)/delta + tmp_tsDF.LAG)

			if sind < 0:
				msg = ('Shift time is before start of records, padding %d points staion %s'%(sind,sta))
				gpar.log(__name__,msg,level='info',pri=True)
			if (sind+npt) >= tr_npt:
				msg = ('Shift time past end of record, padding %d points'%(sind+npt-tr_npt))
				gpar.log(__name__,msg,level='info',pri=True)

			if sind < 0 and (sind+npt) < tr_npt:
				dnpt = npt - np.abs(sind)
				data_use = data[0:dnpt]
				data_use = np.pad(data_use,(np.abs(sind),0),'edge')
				data_com = np.diff(data_use) * tmp_tsDF.DT/delta
				data_com = np.append(data_com,0)
				tmp_data = data_use + data_com
				if stack == 'root':
					shiftTRdata[ind,:] = np.sign(tmp_data)*tmp_data
				elif stack == 'linear' or stack == 'psw':
					shiftTRdata[ind,:] = tmp_data

			elif sind >=0 and sind+npt < tr_npt:
				data_use = data[sind:npt+sind]
				# print(data_use[0])
				data_com = np.diff(data_use) * tmp_tsDF.DT/delta
				data_com = np.append(data_com,0)
				tmp_data = data_use + data_com
				if stack == 'root':
					shiftTRdata[ind,:] = np.sign(tmp_data)*tmp_data
				elif stack == 'linear' or stack == 'psw':
					shiftTRdata[ind,:] = tmp_data
			elif sind >=0 and (sind+npt) >= tr_npt:
				data_use = data[sind:]
				pad_end = sind+npt - tr_npt
				data_use = np.pad(data_use,(0,pad_end),'edge')
				data_com = np.diff(data_use) * tmp_tsDF.DT/delta
				data_com = np.append(data_com,0)
				tmp_data = data_use + data_com
				if stack == 'root':
					shiftTRdata[ind,:] = np.sign(tmp_data)*np.power(np.abs(tmp_data), 1.0/order)
				elif stack == 'linear' or stack == 'psw':
					shiftTRdata[ind,:] = tmp_data
			elif sind < 0 and sind+npt >= tr_npt:
				pad_head = -sind
				pad_end = npt+sind-tr_npt
				data_use = data
				data_use = np.pad(data_use,(pad_head,pad_end),'edge')
				data_com = np.diff(data_use) * tmp_tsDF.DT/delta
				data_com = np.append(data_com,0)
				tmp_data = data_use + data_com
				if stack == 'root':
					shiftTRdata[ind,:] = np.sign(tmp_data)*tmp_data
				elif stack == 'linear' or stack == 'psw':
					shiftTRdata[ind,:] = tmp_data
		if stack == 'linear':
			beamdata = shiftTRdata.sum(axis=0) / ntr
		elif stack == 'root':
			beamdata = shiftTRdata.sum(axis=0) / ntr
			beamdata = np.sign(beamdata) * np.power(np.abs(beamdata), 1.0/order)
		elif stack == 'psw':
			amp = np.absolute(shiftTRdata)
			stack_amp = np.sum(np.real(shiftTRdata),axis=0)
			beamdata = stack_amp * np.power(np.absolute(np.sum(shiftTRdata/amp,axis=0))/ntr, order)/ntr
		beamTr.data = beamdata
		beamTr.stats.starttime = self.time
		beamTr.stats.channel = 'beam'
		sac = AttribDict({'b':starttime,'e':starttime + (npt-1)*delta,
						  'evla':self.lat,'evlo':self.lon,'evdp':self.dep,
						  'kcmpnm':"beam",'delta':delta,
						  'nzyear':self.time.year,'nzjday':self.time.julday,
						  'nzhour':self.time.hour,'nzmin':self.time.minute,
						  'nzsec':self.time.second,'nzmsec':self.time.microsecond/1000})
		beamTr.stats.sac = sac

		self.beam = beamTr
		if write:
			bpfilt = str(filt[0]) +'-'+str(filt[1])
			name = 'beam.' + self.ID + '.'+stack +'.'+bpfilt+'.sac'
			name = os.path.join('./',arrayName,'Data',self.ID,name)
			beamTr.write(name,format='SAC')
		# etime = time.time()
		# print(etime - stime)

	def slideBeam(self,geometry,timeTable,arrayName,grdpts_x=301,grdpts_y=301,
					filt=[1,2,4,True], sflag=1,stack='linear',
					sll_x=-15.0,sll_y=-15.0,sl_s=0.3,refine=True,
					starttime=400.0,endtime=1400.0, unit='deg',
					winlen=2.0,overlap=0.5,write=False, **kwargs):
		"""
		Function to get beamforming for the phase in self.phase. Default is for PKiKP
		Only coding linear stacking now
		Parameters:
		-------------------
			geometry: DataFrame that contains geometry infromation for the array, as returns by getGeometry
			timeTable: 3D numpy array that contain time shift for select slowness grids for all stations in the array, 
						created by getTimeTable
			arrayName: str, name of array
			sflag: int, options for calculating maximum value of the beamforming traces.
					1: return the maximum amplitude of the traces
					2: return the mean of the trace
					3: return the root-mean-sqaure of the trace
			filt: list, filter parameters for waveform filtering
			stack: str, wave to stack shifted wavefrom.
					linear: linear stacking
					psw: phase weight stacking
					root: root mean stack
			unit: str, units for the ray parameter.
				deg: s/degree
				km: s/km
				rad: s/radian
			grd_x: int, grid numbers for slowness in X direction
			grd_y: int, gid numbers for slowness in Y direction
			sll_x: float, starting point of X slowness
			sll_y: float, startting point of Y slowness
			sl_s: float, step size for slowness
			refine: boot, if True, will do the refine frid search near the best fitting slowness with the same grid steps.
			starttime: float, starttime of the time window to do slide beamforming, related to the starttime of the Streams
			endtime: float, endtime of the time window to do slide beamforming, related to the starttime of the Streams
			winlen: float,  window size in second for the beamforming segment
			overlap: float, from 0-1, the step in percent of winlen to move the beamforming time window
			write: boot, if True write a sac file for beamforming trace, store into the directory where the event waveforms are.

		Retrun:
			Stream store in slideSt of the Class
		"""
		msg = ('Calculating slide beamforming for earthquake %s' % self.ID)
		gpar.log(__name__,msg,level='info',pri=True)
		rel = slideBeam(self.stream, self.ntr, self.delta, 
						geometry,timeTable,arrayName,grdpts_x,grdpts_y,
					filt, sflag,stack,sll_x,sll_y,sl_s,refine,
					starttime,endtime, unit,
					winlen,overlap,**kwargs)
		st = obspy.core.stream.Stream()
		otime = self.time
		ndel = winlen*(1-overlap)
		bnpts = int((endtime-starttime)/ndel)
		label = ['Amplitude','Slowness','Back Azimuzh','coherence']
		# print(label)
		for i in range(4):
			tr = obspy.core.trace.Trace()
			tr.data = rel[i,:]
			tr.stats.starttime = otime+starttime
			tr.stats.delta = ndel
			tr.stats.channel = label[i]
			sac = AttribDict({'b':starttime,'e':starttime + (bnpts-1)*ndel,
						  'evla':self.lat,'evlo':self.lon,'evdp':self.dep,
						  'kcmpnm':"beam",'delta':ndel,
						  'nzyear':self.time.year,'nzjday':self.time.julday,
						  'nzhour':self.time.hour,'nzmin':self.time.minute,
						  'nzsec':self.time.second,'nzmsec':self.time.microsecond/1000})
			tr.stats.sac = sac
			st.append(tr)
		self.slideSt = st
		# print("done")
		if write:
			bpfilt = str(filt[0]) +'-'+str(filt[1])
			name = 'slide.' + self.ID + '.'+stack+'.'+bpfilt+'.sac'
			name = os.path.join('./',arrayName,'Data',self.ID,name)
			st.write(name,format='SAC')

	def vespectrum(self,geometry,arrayName,grdpts=401,
					filt=[1,2,4,True], sflag='log10',stack='linear',
					sl_s=0.1, vary='slowness',sll=-20.0,
					starttime=400.0,endtime=1400.0, unit='deg',
					**kwargs ):
		msg = ('Calculating vespetrum for earthquake %s' % self.ID)
		gpar.log(__name__,msg,level='info',pri=True)
		if vary == 'slowness':
			times, k, abspow = slantBeam(self.stream, self.ntr, self.delta,
									geometry, arrayName, grdpts,
									filt, sflag, stack, sl_s, vary, sll,starttime,
									endtime, unit, bakAzimuth=self.bakAzimuth)
		elif vary == 'theta':
			times, k, abspow = slantBeam(self.stream, self.ntr, self.delta,
									geometry,  arrayName, grdpts,
									filt, sflag, stack, sl_s, vary, sll,starttime,
									endtime, unit, rayParameter=self.rayParameter)
		self.slantTime = times
		self.slantType = vary
		self.slantK = k
		self.energy = abspow

	def plotves(self, show=True, saveName=False,**kwargs):

		extent=[np.min(self.slantTime),np.max(self.slantTime),np.min(self.slantK),np.max(self.slantK)]
		fig, ax = plt.subplots(figsize=(8,6))
		im = ax.imshow(self.energy,extent=extent, aspect='auto',**kwargs)
		if self.slantType == 'slowness':
			unit = 's/deg'
			a = u"\u00b0"
			title = 'Slant Stack at a Backazimuth of %.1f %sN'%(self.bakAzimuth,a)
		elif self.slantType == 'theta':
			unit = 'deg'
			title = 'Slant Stack at a slowness of %.2f s/deg'%(self.rayParameter)
		ax.set_xlabel('Time (s)')
		ax.set_ylabel(self.slantType)
		ax.set_title(title)
		if saveName:
			plt.savefig(saveName)
		if show:
			plt.show()


def getTimeShift(rayParameter,bakAzimuth,geometry,unit='deg'):

	"""
	Function to calculate timeshift table for giving ray parameter and arrary geometry

	Parameters:
		rayParameter: ray parameters or horizontal slowness to calculate the time shift.
		azimuth: 
		geometry: Dataframe containing the arrays geometry, does not consider the elevation
	"""
	if unit == 'deg':
		sx = rayParameter * np.sin(bakAzimuth * deg2rad)
		sx = sx/deg2km
		sy = rayParameter * np.cos(bakAzimuth * deg2rad)
		sy = sy/deg2km
	elif unit == 'km':
		sx = rayParameter * np.sin(bakAzimuth * deg2rad)
		sy = rayParameter * np.cos(bakAzimuth * deg2rad)
	elif unit == 'rad':
		sx = rayParameter * np.sin(bakAzimuth * deg2rad)
		sx = sx/6370.0
		sy = rayParameter * np.cos(bakAzimuth * deg2rad)
		sy = sy/6370.0
	else:
		msg = ('Not valid input, must be one of deg, km or rad')
		gapr.log(__name__,msg,level='error',pri=True,e=ValueError)


	tsx = geometry.X * sx
	tsy = geometry.Y * sy

	timeShift = -(tsx + tsy)
	d = {'STA':geometry.STA,'TimeShift':timeShift,'LAT':geometry.LAT,'LON':geometry.LON}
	timeDF = pd.DataFrame(d)

	return timeDF

def getTimeTable(geometry,sll_x=-15.0,sll_y=-15.0,sl_s=0.1,grdpts_x=301,grdpts_y=301,unit='deg'):
	"""
	Return timeshift table for given array geometry, modified from obsy
	
	"""
	sx = sll_x + np.arange(grdpts_x)*sl_s
	sy = sll_y + np.arange(grdpts_x)*sl_s

	if unit == 'deg':
		sx = sx/deg2km
		sy = sy/deg2km
	elif unit == 'rad':
		sx = rayParameter * np.sin(bakAzimuth * deg2rad)
		sx = sx/6370.0
		sy = rayParameter * np.cos(bakAzimuth * deg2rad)
		sy = sy/6370.0
	else:
		msg = ('Input not one of deg and rad, set unit as s/km')
		gapr.log(__name__,msg,level='warning',pri=True)


	mx = np.outer(geometry.X, sx)
	my = np.outer(geometry.Y, sy)

	timeTable = np.require(mx[:,:,np.newaxis].repeat(grdpts_y,axis=2) + 
				my[:,np.newaxis,:].repeat(grdpts_x,axis=1))

	# timeIndexTable = {}
	# for ind, row in geometry.iterrows():
	# 	sta = row.STA
	# 	timeIndexTable[sta] = timeTable[ind,:,:]

	return timeTable
def getSlantTime(geometry, grdpts=401, sl_s=0.1, sll_k=-20.0,vary='slowness',unit='deg',**kwargs):
	
	k = sll_k + np.arange(grdpts)*sl_s
	if vary == 'slowness':
		rayParameter = k
		bakAzimuth = kwargs['bakAzimuth']
	elif vary == 'theta':
		bakAzimuth = k
		rayParameter = kwargs['rayParameter']
	else:
		msg = ('Not valid input, must be one of slowness or theta\n')
		gapr.log(__name__,msg,level='error',pri=True,e=ValueError)
	if unit == 'deg':
		sx = rayParameter * np.sin(bakAzimuth * deg2rad)
		sx = sx/deg2km
		sy = rayParameter * np.cos(bakAzimuth * deg2rad)
		sy = sy/deg2km
	elif unit == 'km':
		sx = rayParameter * np.sin(bakAzimuth * deg2rad)
		sy = rayParameter * np.cos(bakAzimuth * deg2rad)
	elif unit == 'rad':
		sx = rayParameter * np.sin(bakAzimuth * deg2rad)
		sx = sx/6370.0
		sy = rayParameter * np.cos(bakAzimuth * deg2rad)
		sy = sy/6370.0
	else:
		msg = ('Not valid input, must be one of deg, km or rad')
		gapr.log(__name__,msg,level='error',pri=True,e=ValueError)

	tsx = np.outer(geometry.X, sx)
	tsy = np.outer(geometry.Y, sy)
	timeShift = (tsx + tsy)

	return timeShift

def slideBeam(stream, ntr, delta, geometry,timeTable,arrayName,grdpts_x=301,grdpts_y=301,
					filt=[1,2,4,True], sflag=1,stack='linear',
					sll_x=-15.0,sll_y=-15.0,sl_s=0.1, refine=True,
					starttime=400.0,endtime=1400.0, unit='deg',
					winlen=2.0,overlap=0.5,**kwargs):
	if sflag not in [1, 2, 3]:
		msg = 'Not available option, please choose from 1, 2 or 3\n1 for maximm amplitude\n2for mean\n3 for root-mean-sqaure\n'
		gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
	st = stream
	st.detrend('demean')
	st.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
	npts = st[0].stats.npts#int((endtime - starttime+2.0*padlen)/delta)+1
	winpts = int(winlen/delta) + 1
	tstep = winlen*(1-overlap)
	ndel = winlen*(1-overlap)
	bnpts = int((endtime-starttime)/ndel)
	hilbertTd = np.empty((ntr,npts),dtype=complex)
	stations = timeTable
	for ind, tr in enumerate(st):
		sta = tr.stats.station
		if len(tr.data) > npts:
			data = tr.data[0:npts]
		elif len(tr.data) < npts:
			dn = npts - len(tr.data)
			data = tr.data
			data = np.pad(data,(0,dn),'edge')
		else:
			data = tr.data 
		hilbertTd[ind,:] = scipy.signal.hilbert(data)
	if stack == 'linear':
		order = -1
		iflag = 1
	elif stack == 'psw':
		iflag = 2
		order = kwargs['order']
	elif stack == 'root':
		iflag = 3
		order = kwargs['order']
	else:
		msg = 'Not available stack method, please choose form linaer, psw or root\n'
		gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
	abspow = np.empty(bnpts)
	cohere = np.empty(bnpts)
	slow_h = np.empty(bnpts)
	bakaz = np.empty(bnpts)
	tmpabspow = np.empty((grdpts_x,grdpts_y),dtype=np.double)
	for ind in range(bnpts):
		begtime = starttime + tstep * ind
		errorcode = clibarray.slidebeam(ntr,grdpts_x,grdpts_y,
					sflag,iflag,npts,delta,
					begtime, winpts, order,
					stations,hilbertTd,tmpabspow)
		if errorcode != 0:
			msg = 'slidebeam stack for %dth segment exited with error %d\n' % (ind, errorcode)
			gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
		ix, iy = np.unravel_index(tmpabspow.argmax(),tmpabspow.shape)
		if refine:
			slow_x = sll_x + (ix-1) * sl_s
			slow_y = sll_y + (iy-1) * sl_s
			tsl_s = sl_s *2.0 /grdpts_x
			tmpTimeTable = getTimeTable(geometry,slow_x,slow_y,tsl_s,grdpts_x,grdpts_y,unit)
			errorcode = clibarray.slidebeam(ntr,grdpts_x,grdpts_y,
						sflag,iflag,npts,delta,
						begtime, winpts, order,
						tmpTimeTable,hilbertTd,tmpabspow)
			if errorcode != 0:
				msg = 'slidebeam stack for %dth segment exited with error %d\n' % (ind, errorcode)
				gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
			ix, iy = np.unravel_index(tmpabspow.argmax(),tmpabspow.shape)
			slow_x = slow_x + ix * tsl_s
			slow_y = slow_y + iy * tsl_s
			tmpTimeTable = tmpTimeTable[:,ix,iy]
		else:
			slow_x = sll_x + ix * sl_s
			slow_y = sll_y + iy * sl_s
			tmpTimeTable = stations[:,ix,iy]
		abspow[ind] = tmpabspow[ix, iy]
		slow = np.sqrt(slow_x**2 + slow_y**2)
		if slow < 1e-8:
			slow = 1e-8
		slow_h[ind] = slow
		azimuth = 180.0 * np.arctan2(slow_x, slow_y)/PI
		baz = azimuth % -360 +180
		if baz < 0:
			baz = baz +360.0
		bakaz[ind] = baz
			# get coherence for the maximum stack slowness
		
		cohere[ind] = getCoherence(hilbertTd,begtime,tmpTimeTable,winpts,delta)
		
	rel = np.empty((4,bnpts))
	# rel = np.empty((3,bnpts))
	rel[0,:] = np.log10(abspow)
	rel[1,:] = slow_h
	rel[2,:] = bakaz
	rel[3,:] = cohere

	return rel

def slantBeam(stream, ntr, delta, geometry,arrayName,grdpts=401,
					filt=[1,2,4,True], sflag='log',stack='linear',
					sl_s=0.1, vary='slowness',sll=-20.0,
					starttime=400.0,endtime=1400.0, unit='deg',
					**kwargs):
	st = stream
	st.detrend('demean')
	st.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
	npts = st[0].stats.npts
	winlen = endtime - starttime
	winpts = int(winlen/delta) + 1
	times = starttime + np.arange(winpts) * delta
	timeTable = getSlantTime(geometry,grdpts,sl_s,sll,vary,unit,**kwargs)
	hilbertTd = np.empty((ntr,npts),dtype=complex)
	for ind, tr in enumerate(st):
		sta = tr.stats.station
		if len(tr.data) > npts:
			data = tr.data[0:npts]
		elif len(tr.data) < npts:
			dn = npts - len(tr.data)
			data = tr.data
			data = np.pad(data,(0,dn),'edge')
		else:
			data = tr.data 
		hilbertTd[ind,:] = scipy.signal.hilbert(data)
	if stack == 'linear':
		order = -1
		iflag = 1
	elif stack == 'psw':
		iflag = 2
		order = kwargs['order']
	elif stack == 'root':
		iflag = 3
		order = kwargs['order']
	else:
		msg = 'Not available stack method, please choose from linear, psw or root\n'
		gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
	k = sll + np.arange(grdpts)*sl_s
	abspow = np.empty((grdpts,winpts))
	errorcode = clibarray.slant(ntr,grdpts,
				iflag,npts,delta,
				starttime, winpts, order,
				timeTable,hilbertTd,abspow)
	if errorcode != 0:
		msg = 'slidebeam stack for %dth segment exited with error %d\n' % (ind, errorcode)
		gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
	if sflag == 'beam':
		return [time, k, abspow]
	envel = np.abs(scipy.signal.hilbert(abspow))
	if sflag == 'log':
		envel = np.log(envel)
	elif sflag == 'log10':
		envel = np.log10(envel)
	elif sflag == 'sqrt':
		envel = np.sqrt(envel)
	else:
		msg = 'Not available option, please choose from beam, log, log10 or sqrt\n'
		gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
	return [times, k, envel]



def getCoherence(hilbertTd,sbeg,TimeTable,winpts,delta):
	"""
	Function to calculate coherence between two signals.
	Parameters:
		hilberTd: complex np.array. hilbert transform of signals.
		snpt: start index of the signals
		TimeTable: 1D time shift table for all signal traces
		winpts: total length in points of signals
		delta: float, delta of signals
	"""
	[nstat, npts] = hilbertTd.shape
	sinds = ((sbeg+TimeTable)/delta).astype(int) + 1
	dt = TimeTable - delta * (TimeTable/delta).astype(int)
	shiftTRdata = np.empty((nstat,winpts),dtype=complex)
	for ind in range(nstat):
		sind = sinds[ind]
		dx = dt[ind]
		if sind < 0 and (sind+winpts) < npts:
			dnpt = winpts - np.abs(sind)
			if dnpt < 0:
				msg = ("Time shift for segment in index %d for staion %d is too large to negative\n"%(sind, ind))
				gpar.log(__name__,msg,level='warning',pri=True)
				data_use = np.ones(winpts) * hilbertTd[ind,0]
				data_com = np.zeros(winpts)
			else:
				data_use = hilbertTd[ind, 0:dnpt]
				data_use = np.pad(data_use,(np.abs(sind),0),'edge')
				data_com = np.diff(data_use) * dx/delta
				data_com = np.append(data_com,0)
			shiftTRdata[ind,:] = data_use + data_com

		elif sind >=0 and (sind+winpts) < npts:
			data_use = hilbertTd[ind, sind:winpts+sind]
			data_com = np.diff(data_use) * dx/delta
			data_com = np.append(data_com,0)
			shiftTRdata[ind,:] = data_use + data_com
		elif sind >=0 and (sind+winpts) >= npts:
			if sind >= npts:
				msg = ("Time shift for segment in index %d for staion %d is too large, outside the end\n"%(sind, ind))
				gpar.log(__name__,msg,level='warning',pri=True)
				data_use = np.ones(winpts)*hilbertTd[ind,-1]
				data_com = np.zeros(winpts)
			else:
				data_use = hilbertTd[ind, sind:]
				pad_end = sind+winpts - npts
				data_use = np.pad(data_use,(0,pad_end),'edge')
				data_com = np.diff(data_use) * dx/delta
				data_com = np.append(data_com,0)
			shiftTRdata[ind,:] = data_use + data_com
		elif sind < 0 and sind+winpts >= npts:
			pad_head = -sind
			pad_end = winpts+sind-npts
			data_use = hilbertTd[ind, :]
			data_use = np.pad(data_use,(pad_head,pad_end),'edge')
			data_com = np.diff(data_use) * dx/delta
			data_com = np.append(data_com,0)
			shiftTRdata[ind,:] = data_use + data_com
	amp = np.absolute(shiftTRdata)
	tmp = np.sum(shiftTRdata/amp, axis=0)/nstat
	cohere = np.mean(np.absolute(tmp))


	return cohere 



