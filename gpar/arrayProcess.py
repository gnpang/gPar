from __future__ import print_function, absolute_import
from __future__ import with_statement, nested_scopes, division, generators

import os
import gc
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
from matplotlib.figure import Figure
import matplotlib.cm as cm
from itertools import zip_longest
try:
	import cPickle
except ImportError:
	import pickle as cPickle

from gpar.util.rolling_window import rolling_window

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

	def __init__(self,arrayName,refPoint,eqDF,staDf, 
				 coordsys='lonlat',beamphase='PKiKP', 
				 isDoublet=False,
				 phase_list=['P','PP','PcP','ScP','PKiKP','SP','ScS']
				 ,**kwargs):
		self.name = arrayName
		self.refPoint = refPoint
		self.coordsys = coordsys
		if not isDoublet:
			self.getGeometry(staDf,refPoint,coordsys=coordsys)
		if not isDoublet:
			self.events = [0]*len(eqDF)
			for ind, row in eqDF.iterrows():
				self.events[ind] = Earthquake(self, row, beamphase=beamphase,phase_list=phase_list)
		else:
			self.doublet = [0]*len(eqDF)
			for ind, row in eqDF.iterrows():
				self.doublet[ind] = Doublet(self, row, phase=[beamphase],**kwargs)

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
			dis_in_degree, baz, az = gpar.getdata.calc_Dist_Azi(staDF.LAT, staDF.LON, refPoint[0], refPoint[1])
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

	def beamforming(self,filts={'filt_1':[1,2,4,True],'filt_2':[2,4,4,True],'filt_3':[1,3,4,True]},
				    starttime=0,winlen=1800.0,
					stack='linear',unit='deg',write=True):
		"""
		Function to do beamforming for all earthquakes in the array
		"""
		for eq in self.events:
			eq.beamforming(geometry=self.geometry,arrayName=self.name,starttime=starttime,
						   winlen=winlen,filts=filts,unit=unit,
						   stack=stack, write=write)

	def slideBeam(self,filts={'filt_1':[1,2,4,True],'filt_2':[2,4,4,True],'filt_3':[1,3,4,True]},
				  grdpts_x=301,grdpts_y=301,sflag=2,stack='linear',
				  sll_x=-15.0,sll_y=-15.0,sl_s=0.1,refine=True,
				  starttime=400.0,endtime=1400.0, unit='deg',
				  winlen=2.0,overlap=0.5,write=False, **kwargs):
		
		for eq in self.events:
			eq.slideBeam(geometry=self.geometry,timeTable=self.timeTable,arrayName=self.name,
						 grdpts_x=grdpts_x,grdpts_y=grdpts_y,
						 filts=filts, 
						 sflag=sflag,stack=stack,
						 sll_x=sll_x,sll_y=sll_y,sl_s=sl_s,refine=refine,
						 starttime=starttime,endtime=endtime, unit=unit,
						 winlen=winlen,overlap=overlap,write=write, **kwargs)

	def vespectrum(self,grdpts=401,
					filts={'filt_1':[1,2,4,True],'filt_2':[2,4,4,True],'filt_3':[1,3,4,True]}, 
					sflag='log10',stack='linear',
					sl_s=0.1, vary='slowness',sll=-20.0,
					starttime=400.0,endtime=1400.0, unit='deg',
					**kwargs ):
		for eq in self.events:
			eq.vespectrum(geometry=self.geometry,arrayName=self.name,grdpts=grdpts,
					filts=filts, sflag=sflag,stack=stack,
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

	def __init__(self, array, row, beamphase='PKiKP',phase_list=['P','PP','PcP','ScP','PKiKP','SP','ScS']):
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
		self.beamphase = beamphase
		self.phase_list = phase_list
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

	def getArrival(self,phase_list=None, model='ak135'):
		"""
		Function to get theoritcal arrival times for the events

		Parameters:
			phase: str or list. Phase name for travel time calculating
			model: str. Model using in taup.
		"""

		model = TauPyModel(model)
		if phase_list == None:
			phase_list = self.phase_list
		arrivals = model.get_travel_times(source_depth_in_km=self.dep,distance_in_degree=self.Del,phase_list=phase_list)

		phases = {}
		for arr in arrivals:
			pha = arr.name
			times = {'UTC':self.time + arr.time,
					 'TT':arr.time,
					 'RP':arr.ray_param_sec_degree}
			phases[pha] = times

		self.arrivals = phases
		msg = ('Travel times for %s for earthquake %s in depth of %.2f in distance of %.2f' % (phase_list, self.ID, self.dep, self.Del))
		gpar.log(__name__,msg,level='info',pri=True)

	def beamforming(self, geometry, arrayName, starttime=0.0, winlen=1800.0,
					filts={'filt_1':[1,2,4,True],'filt_2':[2,4,4,True],'filt_3':[1,3,4,True]},
					stack='linear',unit='deg',write=True,**kwargs):
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
		tsDF = getTimeShift(self.rayParameter,self.bakAzimuth,geometry,unit=unit)
		self.timeshift = tsDF
		stalist = tsDF.STA
		st = self.stream.copy()
		if len(st) < len(tsDF):
			for s in stalist:
				tmp_st = st.select(station=s)
				if len(tmp_st) == 0:
					msg = 'Station %s is missing for event %s in array %s, dropping station'%(s, self.ID, arrayName)
					gpar.log(__name__, msg, level='info', pri=True)
					tsDF = tsDF[~(tsDF.STA == s)]
		ntr = self.ntr
		delta = self.delta
		st.detrend('demean')

		
		# st.detrend('demean')
		DT = tsDF.TimeShift - (tsDF.TimeShift/delta).astype(int)*delta
		lag = (tsDF.TimeShift/delta).astype(int)+1
		tsDF['DT'] = DT
		tsDF['LAG'] = lag
		npt = int(winlen/delta) + 1
		beamSt = obspy.core.stream.Stream()
		for name, filt in filts.items():
			msg = ('Calculating beamforming for earthquake %s in filter %s - %s' % (self.ID, name, filt))
			gpar.log(__name__,msg,level='info',pri=True)
			tmp_st = st.copy()
			tmp_st.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
			beamTr = obspy.core.trace.Trace()
			beamTr.stats.delta = self.delta
			if stack == 'psw':
				shiftTRdata = np.empty((ntr,npt),dtype=complex)
			else:
				shiftTRdata = np.empty((ntr,npt))
			for ind, tr in enumerate(tmp_st):
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
			bpfilt = str(filt[0]) +'-'+str(filt[1])
			beamTr.stats.network = 'beam'
			beamTr.stats.channel = bpfilt
			beamTr.stats.station = name
			sac = AttribDict({'b':starttime,'e':starttime + (npt-1)*delta,
							  'evla':self.lat,'evlo':self.lon,'evdp':self.dep,
							  'delta':delta,
							  'nzyear':self.time.year,'nzjday':self.time.julday,
							  'nzhour':self.time.hour,'nzmin':self.time.minute,
							  'nzsec':self.time.second,'nzmsec':self.time.microsecond/1000})
			beamTr.stats.sac = sac
			if write:
				name = 'beam.' + self.ID + '.'+stack +'.'+bpfilt+'.sac'
				name = os.path.join('./',arrayName,'Data',self.ID,name)
				beamTr.write(name,format='SAC')
			beamSt.append(beamTr)

		self.beam = beamSt
		
		# etime = time.time()
		# print(etime - stime)

	def slideBeam(self,geometry,timeTable,arrayName,grdpts_x=301,grdpts_y=301,
					filts={'filt_1':[1,2,4,True],'filt_2':[2,4,4,True],'filt_3':[1,3,4,True]}, 
					sflag=1,stack='linear',
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
			filts: dict, filters parameters for waveform filtering
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
		self.slideSt = {}
		for name, filt in filts.items():
			msg = ('Calculating slide beamforming for earthquake %s in array %s in filter %s - %s' % (self.ID, arrayName, name, filt))
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
			self.slideSt[name] = st
		# print("done")
			if write:
				bpfilt = str(filt[0]) +'-'+str(filt[1])
				name = 'slide.' + self.ID + '.'+stack+'.'+bpfilt+'.pkl'
				name = os.path.join('./',arrayName,'Data',self.ID,name)
				st.write(name,format='PICKLE')

	def vespectrum(self,geometry,arrayName,grdpts=401,
					filts={'filt_1':[1,2,4,True],'filt_2':[2,4,4,True],'filt_3':[1,3,4,True]},
					sflag='log10',stack='linear',
					sl_s=0.1, vary='slowness',sll=-20.0,
					starttime=400.0,endtime=1400.0, unit='deg',
					**kwargs ):
		msg = ('Calculating vespetrum for earthquake %s' % self.ID)
		gpar.log(__name__,msg,level='info',pri=True)
		if vary == 'slowness':
			times, k, abspow = slantBeam(self.stream, self.ntr, self.delta,
									geometry, arrayName, grdpts,
									filts, sflag, stack, sl_s, vary, sll,starttime,
									endtime, unit, bakAzimuth=self.bakAzimuth)
		elif vary == 'theta':
			times, k, abspow = slantBeam(self.stream, self.ntr, self.delta,
									geometry,  arrayName, grdpts,
									filts, sflag, stack, sl_s, vary, sll,starttime,
									endtime, unit, rayParameter=self.rayParameter)
		self.slantTime = times
		self.slantType = vary
		self.slantK = k
		self.energy = abspow

	def plotves(self, show=True, saveName=False,**kwargs):

		extent=[np.min(self.slantTime),np.max(self.slantTime),np.min(self.slantK),np.max(self.slantK)]
		fig, ax = plt.subplots(1, len(self.energy.keys()),figsize=(8,6),sharey='row')
		for ind, (name, abspow) in enumerate(self.energy.items()):
			im = ax[ind].imshow(abspow,extent=extent, aspect='auto',**kwargs)
			ax[ind].set_title(name)
			ax[ind].set_xlabel('Time (s)')
		if self.slantType == 'slowness':
			unit = 's/deg'
			a = u"\u00b0"
			title = 'Slant Stack at a Backazimuth of %.1f %sN'%(self.bakAzimuth,a)
		elif self.slantType == 'theta':
			unit = 'deg'
			title = 'Slant Stack at a slowness of %.2f s/deg'%(self.rayParameter)
		
		ax[0].set_ylabel(self.slantType)
		fig.suptitle(title)
		if saveName:
			plt.savefig(saveName)
		if show:
			plt.show()

class Doublet(object):
	"""
	Earthquake doublets object
	"""

	def __init__(self, array, row, 
				 resample=0.01, method='resample',
				 filt=[1.33, 2.67,3,True],
				 cstime=20.0, cetime=20.0,
				 winlen=5, step=0.05,
				 starttime=100.0, endtime=300.0,
				 domain='freq', fittype='cos',
				 phase=['PKIKP']):

		self.ID = row.DoubleID
		self.ev1 = {'TIME': UTCDateTime(row.TIME1), 'LAT': row.LAT1,
					'LON': row.LON1, 'DEP': row.DEP1,
					'MAG': row.M1}
		self.ev2 = {'TIME': UTCDateTime(row.TIME2), 'LAT': row.LAT2,
					'LON': row.LON2, 'DEP': row.DEP2,
					'MAG': row.M2}
		self.phase = phase
		self.delta = resample
		self.winlen = winlen
		self.step = step
		self.st1 = row.ST1
		self.st2 = row.ST2
		self._checkInput()
		self._resample(resample,method)
		if hasattr(row, 'TT1') and hasattr(row, 'TT2'):
			self.tt1 = row.TT1
			self.tt2 = row.TT2
			self.arr1 = self.ev1['TIME'] + row.TT1
			self.arr2 = self.ev2['TIME'] + row.TT2
		else:
			self.getArrival(array=array)
		self._alignWave(filt=filt,delta=resample,cstime=cstime,cetime=cetime,
						starttime=starttime,endtime=endtime,domain=domain,fittype=fittype)

	def _checkInput(self):
		if not isinstance(self.st1, obspy.core.stream.Stream):
			msg = ('Waveform data for %s is not stream, stop running' % self.ID)
			gpar.log(__name__,msg,level='error',pri=True)
		if not isinstance(self.st2, obspy.core.stream.Stream):
			msg = ('Waveform data for %s is not stream, stop running' % self.ID)
			gpar.log(__name__,msg,level='error',pri=True)
	def _resample(self, resample, method):
		self.st1.detrend('demean')
		self.st2.detrend('demean')
		rrate = 1.0/resample
		if method == 'resample':
			self.st1.resample(sampling_rate=rrate)
			self.st2.resample(sampling_rate=rrate)
		elif method == 'interpolate':
			self.st1.interpolate(sampling_rate=rrate)
			self.st2.interpolate(sampling_rate=rrate)
		else:
			msg = ('Not a valid option for waveform resampling, choose from resample and interpolate')
			gpar.log(__name__,msg,level='error',e='ValueError',pri=True)

	def getArrival(self,array, phase=None, model='ak135'):
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
		arr1 = model.get_travel_times_geo(source_depth_in_km=self.ev1['DEP'],source_latitude_in_deg=self.ev1['LAT'],
										  source_longitude_in_deg=self.ev1['LON'],phase_list=phase,
										  receiver_latitude_in_deg=array.refPoint[0], receiver_longitude_in_deg=array.refPoint[1])[0]
		self.arr1 = self.ev1['TIME'] + arr1.time
		self.tt1 = arr1.time
		arr2 = model.get_travel_times_geo(source_depth_in_km=self.ev2['DEP'],source_latitude_in_deg=self.ev2['LAT'],
										  source_longitude_in_deg=self.ev2['LON'],phase_list=phase,
										  receiver_latitude_in_deg=array.refPoint[0], receiver_longitude_in_deg=array.refPoint[1])[0]
		self.arr2 = self.ev2['TIME'] + arr2.time
		self.tt2 = arr2.time
		# msg = ('Travel time for %s for earthquake %s in depth of %.2f in distance of %.2f is %s' % (phase, self.ID, self.dep, self.Del, self.arrival))
		# gpar.log(__name__,msg,level='info',pri=True)

	def _alignWave(self,filt=[1.33, 2.67,3,True],delta=0.01,
				   cstime=20.0, cetime=20.0,
				   starttime=100.0, endtime=300.0,
				   domain='freq', fittype='cos'):
		sta1 = []
		sta2 = []
		for tr1, tr2 in zip_longest(self.st1, self.st2):
			if tr1 is not None:
				sta1.append(tr1.stats.network+'.'+tr1.stats.station+'..'+tr1.stats.channel)
			if tr2 is not None:
				sta2.append(tr2.stats.network+'.'+tr2.stats.station+'..'+tr2.stats.channel)
		if len(self.st1) != len(self.st2):
			msg = ('station records for two events are not equal, remove the additional station')
			gpar.log(__name__,msg,level='info',pri=True)
			
			sta = list(set(sta1) & set(sta2))
			st1 = obspy.Stream()
			st2 = obspy.Stream()
			for s in sta:
				_tr1 = self.st1.select(id=s).copy()[0]
				_tr2 = self.st2.select(id=s).copy()[0]
				st1.append(_tr1)
				st2.append(_tr2)
			st1.sort(keys=['station'])
			st2.sort(keys=['station'])
			st1.filter('bandpass', freqmin=filt[0], freqmax=filt[1], corners=filt[2], zerophase=filt[3])
			st2.filter('bandpass', freqmin=filt[0], freqmax=filt[1], corners=filt[2], zerophase=filt[3])
			sta.sort()
		else:
			st1 = self.st1.copy()
			st2 = self.st2.copy()
			st1.sort(keys=['station'])
			st2.sort(keys=['station'])
			st1.filter('bandpass', freqmin=filt[0], freqmax=filt[1], corners=filt[2], zerophase=filt[3])
			st2.filter('bandpass', freqmin=filt[0], freqmax=filt[1], corners=filt[2], zerophase=filt[3])
			sta1.sort()
			sta = sta1

		stime1 = self.arr1 - cstime
		etime1 = self.arr1 + cetime
		tmp_st1 = st1.copy()
		tmp_st1.trim(starttime=stime1, endtime=etime1)
		stime2 = self.arr2 - cstime
		etime2 = self.arr2 + cetime
		tmp_st2 = st2.copy()
		tmp_st2.trim(starttime=stime2, endtime=etime2)

		npts = int((cetime + cstime)/delta) + 1
		Mptd1 = np.zeros([len(st1), npts])
		Mptd2 = np.zeros([len(st2), npts])
		inds = range(len(st1))
		for ind, tr1, tr2 in zip_longest(inds, tmp_st1, tmp_st2):

			if tr1.stats.station != tr2.stats.station:
				msg = ('Orders of the traces are not right')
				gpar.log(__name__, msg, level='error',pri=True)
			data1 = tr1.data
			data2 = tr2.data

			if len(data1) > npts:
				data1 = data1[:npts]
			elif len(data1) < npts:
				data1 = np.pad(data1, (0,npts-len(data1)), 'edge')

			if len(data2) > npts:
				data2 = data2[:npts]
			elif len(data2) < npts:
				data2 = np.pad(data2, (0,npts-len(data2)), 'edge')

			Mptd1[ind,:] = data1
			Mptd2[ind,:] = data2

		taup, cc, _, _ = _getLag(Mptd1, Mptd2, delta, domain, fittype)
		col = ['STA','TS','CC']
		_df = pd.DataFrame(columns=col)
		_df['STA'] = sta
		_df['TS'] = taup
		_df['CC'] = cc
		self.align = _df
		# self.initTS = np.array(taup)
		# self.alignCC = np.array(cc)
		thiftBT = - starttime - np.array(taup)
		st2.trim(starttime = self.arr2-starttime, endtime=self.arr2+endtime)
		for ind, tr in enumerate(st1):
			stime = self.arr1 + thiftBT[ind]
			tr.trim(starttime=stime, endtime=stime+starttime+endtime)
		self.use_st1 = st1
		self.use_st2 = st2

	def codaInter(self, delta=0.01, 
				  winlen=5, step=0.05, 
				  starttime=100.0,
				  domain='freq',fittype='cos'):
		# rrate = 1.0/resample

		# if method == 'resample':
		# 	self.st1.resample(sampling_rate=rrate)
		# 	self.st2.resample(sampling_rate=rrate)
		# elif method == 'interpolate':
		# 	self.st1.interpolate(sampling_rate=rrate)
		# 	self.st2.interpolate(sampling_rate=rrate)
		# else:
		# 	msg = ('Not a valid option for waveform resampling, choose from resample and interpolate')
		# 	gpar.log(__name__,msg,level='error',e='ValueError',pri=True)

		# npts = int((endtime - starttime)/shift)
		# stalist = self.geometry.STA.tolist()
		if winlen is None:
			winlen = self.winlen
		if step is None:
			step = self.step
		if delta is None:
			delta = self.delta
		st1 = self.use_st1.copy()
		st2 = self.use_st2.copy()
		npts = int(winlen / delta)
		taup = []
		cc = []
		for win_st1, win_st2 in zip_longest(st1.slide(winlen, step), st2.slide(winlen, step)):
			_taup, _cc = codaInt(win_st1, win_st2, delta=delta, npts=npts,domain=domain,fittype=fittype)
			taup.append(_taup)
			cc.append(_cc)

		self.taup = taup
		self.cc = cc

		tpts = len(taup)
		ts = self.tt1 + np.arange(tpts) * step - starttime
		self.ts = ts

	def updateFilter(filt, starttime=100, endtime=300,
					cstime=20.0, cetime=20.0,
					winlen=None, step=None):
		self._alignWave(filt=filt,delta=self.delta,
						cstime=cstime,cetime=cetime,
						starttime=starttime,endtime=endtime)
		self.codaInter(delta=None, winlen=winlen, step=step,starttime=starttime)

	def plotCoda(self,tstart=20, tend=10, 
				 sta_id='CN.YKR9..SHZ',
				 stime=100, etime=300,
				 lim=[1100, 1450],
				 filt=[1.33, 2.67, 3, True]):
		# fig = plt.figure()

		# 
		plt.subplot(5,1,1)
		st1 = self.use_st1.copy()
		st2 = self.use_st2.copy()
		tr1 = st1.select(id=sta_id).copy()[0]
		tr2 = st2.select(id=sta_id).copy()[0]
		_ts = self.align[self.align.STA==sta_id].TS.iloc[0]
		tr1.trim(starttime=self.arr1-tstart-_ts, endtime=self.arr1+tend-_ts)
		tr2.trim(starttime=self.arr2-tstart, endtime=self.arr2+tend)
		data1 = tr1.data / np.max(np.absolute(tr1.data))
		data2 = tr2.data / np.max(np.absolute(tr2.data))
		# t1 = self.tt1 - tstart + np.arange(len(data1)) * st1[0].stats.delta 
		t2 = self.tt2 - tstart + np.arange(len(data2)) * tr2.stats.delta 
		plt.plot(t2, data1, 'b', t2, data2, 'r')
		# plt.xlabel('Time (s)')
		plt.ylabel('Normalized Amp')

		plt.subplot(5,1,2)
		tr1 = st1.select(id=sta_id).copy()[0]
		tr2 = st2.select(id=sta_id).copy()[0]
		tr1.trim(starttime=self.arr1-stime-_ts, endtime=self.arr1+etime-_ts)
		tr2.trim(starttime=self.arr2-stime, endtime=self.arr2+etime)
		t = self.ts.min() + np.arange(len(st1[0].data))*st1[0].stats.delta
		plt.plot(t, tr1.data)
		plt.ylabel('Event 1')
		plt.xlim(lim)
		plt.subplot(5,1,3)
		plt.plot(t,tr2.data)
		plt.xlim(lim)
		plt.ylabel('Event 2')
		plt.subplot(5, 1, 4)
		plt.plot(self.ts, self.cc)
		plt.ylim([0, 1])
		plt.xlim(lim)
		plt.ylabel('CCmax Stack')
		plt.subplot(5, 1, 5)
		plt.plot(self.ts, self.taup)
		plt.ylabel('Tau Stack')
		plt.ylim([-2,2])
		plt.xlim(lim)
		plt.xlabel('Time (s)')
		print('Showing figure')
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
	st = stream.copy()
	st.detrend('demean')
	st.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
	npts = st[0].stats.npts#int((endtime - starttime+2.0*padlen)/delta)+1
	winpts = int(winlen/delta) + 1
	tstep = winlen*(1-overlap)
	ndel = winlen*(1-overlap)
	bnpts = int((endtime-starttime)/ndel)
	hilbertTd = np.empty((ntr,npts),dtype=complex)
	# stations = timeTable
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
		gpar.log(__name__,msg,level='error',pri=True)
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
					timeTable,hilbertTd,tmpabspow)
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
			tmpTimeTable = timeTable[:,ix,iy]
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
					filts={'filt_1':[1,2,4,True],'filt_2':[2,4,4,True],'filt_3':[1,3,4,True]}, 
					sflag='log',stack='linear',
					sl_s=0.1, vary='slowness',sll=-20.0,
					starttime=400.0,endtime=1400.0, unit='deg',
					**kwargs):
	st = stream.copy()
	st.detrend('demean')
	# st.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
	npts = st[0].stats.npts
	winlen = endtime - starttime
	winpts = int(winlen/delta) + 1
	times = starttime + np.arange(winpts) * delta
	timeTable = getSlantTime(geometry,grdpts,sl_s,sll,vary,unit,**kwargs)
	k = sll + np.arange(grdpts)*sl_s
	envel = {}
	for name, filt in filts.items():
		tmp_st = st.copy()
		tmp_st.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
		hilbertTd = np.empty((ntr,npts),dtype=complex)
		for ind, tr in enumerate(tmp_st):
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
	
		abspow = np.empty((grdpts,winpts))
		errorcode = clibarray.slant(ntr,grdpts,
					iflag,npts,delta,
					starttime, winpts, order,
					timeTable,hilbertTd,abspow)
		if errorcode != 0:
			msg = 'slantbeam stack for %dth segment in filter %s-%s exited with error %d\n' % (ind, name, filt,errorcode)
			gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
		if sflag == 'beam':
			envel[name] = abspow
		else:
			env = np.abs(scipy.signal.hilbert(abspow))
			if sflag == 'log':
				envel[name] = np.log(env)
			elif sflag == 'log10':
				envel[name] = np.log10(env)
			elif sflag == 'sqrt':
				envel[name] = np.sqrt(env)
			else:
				msg = 'Not available option, please choose from beam, log, log10 or sqrt\n'
				gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
	return [times, k, envel]

def codaInt(st1, st2, delta, npts, domain='freq', fittype='cos'):

	if len(st1) != len(st2):
		msg = ('Stream 1 and stream 2 have diferent amount of traces')
		gpar.log(__name__, msg, level='error', pri=True)

	Mptd1 = np.zeros([len(st1), npts])
	Mptd2 = np.zeros([len(st2), npts])
	inds = range(len(st1))

	for ind, tr1, tr2 in zip_longest(inds, st1, st2):

		if tr1.stats.station != tr2.stats.station:
			msg = ('Orders of the traces are not right')
			gpar.log(__name__, msg, level='error',pri=True)
		data1 = tr1.data
		data2 = tr2.data

		if len(data1) > npts:
			data1 = data1[:npts]
		elif len(data1) < npts:
			data1 = np.pad(data1, (0,npts-len(data1)), 'edge')

		if len(data2) > npts:
			data2 = data2[:npts]
		elif len(data2) < npts:
			data2 = np.pad(data2, (0,npts-len(data2)), 'edge')

		Mptd1[ind,:] = data1
		Mptd2[ind,:] = data2

	_,_,taup, cc = _getLag(Mptd1, Mptd2, delta, domain,fittype)

	return [taup, cc]

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

def _getFreqDomain(mptd):
	reqlen = 2*mptd.shape[-1]
	reqlenbits = 2**reqlen.bit_length()
	mpfd = scipy.fftpack.fft(mptd, n =reqlenbits)

	return mpfd

def _corr(Mptd1, Mptd2, domain):


	t1, n = Mptd1.shape
	t2, m = Mptd2.shape

	if t1 != t2:
		msg = ('Amounts of traces between events are not equal, please check')
		gpar.log(__name__, msg, level='error', pri=True)


	# denorm = np.sqrt(n * np.sum(Mptd1**2, axis=1) - np.sum(Mptd1, axis=1))
	# denorm = denorm * np.sqrt(m * np.sum(Mptd2**2, axis=1) - np.sum(Mptd2, axis=1))
	# denorm = denorm.reshape(t1, 1)
	std1 = np.std(Mptd1, axis=1)
	std2 = np.std(Mptd2, axis=1)
	denorm = std1 * std2 * np.sqrt(n*m)
	denorm = denorm.reshape(t1, 1)
	# print(denorm)

	if domain == 'freq':
		nt1 = np.mean(Mptd1, axis=-1).reshape(t1,1)
		nt2 = np.mean(Mptd2, axis=-1).reshape(t2,1)
		Mpfd1 = _getFreqDomain(Mptd1 - nt1)
		Mpfd2 = _getFreqDomain(Mptd2 - nt2)

		c = np.real(scipy.fftpack.ifft(np.multiply(np.conjugate(Mpfd1), Mpfd2)))
		c = np.concatenate((c[:,-(n-1):], c[:,:m]),axis=1)
		# print(c[16,:])
		cor = c/denorm
	elif domain == 'time':
		c = np.convolve(Mptd1[..., ::-1], Mptd2)
		c = np.subtract(np.sqrt(n*m)*c, np.sum(Mptd1, axis=1)*np.sum(Mptd2, axis=1)) 
		cor = c/denorm
	else:
		msg = ('Not a valid option for correlation, choose from freq or time')
		gpar.log(__name__, msg, level='error', pri=True)

	return cor

def _getLag(Mptd1, Mptd2, delta, domain='freq', fittype='cos'):

	cc = _corr(Mptd1, Mptd2,domain=domain)
	ntr, ndt = Mptd1.shape
	_dump, n = cc.shape
	dt = (np.arange(n) - (ndt-1)) * delta
	_ind = np.arange(ntr) * n
	_index = np.argmax(cc, axis=-1) 
	_indexpre = _index - 1
	_indexnxt = _index + 1

	x = np.vstack([dt[_indexpre], dt[_index], dt[_indexnxt]]).transpose()

	_cind = _index + _ind
	_cindpre = _cind - 1
	_cindnxt = _cind + 1
	_yindex = np.unravel_index(_cind,cc.shape)
	_yindexpre = np.unravel_index(_cindpre,cc.shape)
	_yindexnxt = np.unravel_index(_cindnxt,cc.shape)
	y = np.vstack([cc[_yindexpre], cc[_yindex], cc[_yindexnxt]]).transpose()
	if fittype == 'cos':
		xmax, ymax = cosinfit(x, y, delta)
	elif fittype == 'parabola':
		xmax = []
		ymax = []
		for i in ntr:
			_x, _y = polyfit(x[i,:], y[i,:])
			xmax.append[_x]
			ymax.append[_y]
		xmax = np.array(xmax)
		ymax = np.array(ymax)
	else:
		msg = ('Not a valid option for fitting CC curve, selet from parabola and cos')
		gpar.log(__name__, msg, level='error', pri=True)

	return [xmax, ymax,np.mean(xmax), np.mean(ymax)]

def polyfit(x, y):
	p = np.polyfit(x,y,2)
	xmax = -p[1]/(2.0*p[0])
	ymax = p[2]-p[1]**2/(4.0*p[0])

	return [xmax, ymax]

def cosinfit(x, y, delta):
	w = np.arccos((y[:,0] + y[:,2])/(2*y[:,1]))/delta
	a = np.arctan((y[:,0]-y[:,2])/(2*y[:,1]*np.sin(w*delta)))
	t = -a/w
	xmax = x[:,1] + t
	O = a - w*x[:,1]
	ymax = y[:,1] / np.cos(a)
	# ymax = np.cos(w*xmax+O)

	return [xmax, ymax]

















