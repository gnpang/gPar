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
from obspy.signal.util import next_pow_2, util_geo_km
from obspy.signal.util import util_geo_km
from obspy.signal.headers import clibsignal
from obspy.signal.invsim import cosine_taper
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.cm as cm
from itertools import zip_longest
try:
	import cPickle
except ImportError:
	import pickle as cPickle

# from memory_profiler import profile

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
		self.getGeometry(staDf,refPoint,coordsys=coordsys)
		if not isDoublet:
			self.events = [0]*len(eqDF)
			for ind, row in eqDF.iterrows():
				self.events[ind] = Earthquake(self, row, beamphase=beamphase,phase_list=phase_list)
		else:
			self.doublet = [0]*len(eqDF)
			for ind, row in eqDF.iterrows():
				self.doublet[ind] = Doublet(self, row, tphase=beamphase,phase_list=phase_list,**kwargs)

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
			staDF['DEL'] = dis_in_degree
			staDF['DIS'] = dis_in_km
			RX = dis_in_degree * np.sin(az*deg2rad)
			RY = dis_in_degree * np.cos(az*deg2rad)
			X = dis_in_km * np.sin(az*deg2rad)
			Y = dis_in_km * np.cos(az*deg2rad)
			staDF['X'] = X
			staDF['Y'] = Y
			staDF['RX'] = RX
			staDF['RY'] = RY
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

	def calARF(self,tsx, tsy,
			  freq=1.0,sll_x=-15,sll_y=-15,
			  sl_s=0.1,grdpts_x=301,grdpts_y=301,
			  ):
		'''
		Function to calculate certain array response for certain frequency
		Parameters:
			tsx: float, target x slowness, s/deg
			tsy: float, target y slowness, s/deg
			freq: float, target frequency in Hz
			sll_x: float, minimum x slowness, s/deg
			sll_y: float, minimum y slowness, s/deg
			sl_s: float, increment in slowness, s/deg
			grdpts_x: int, total points in x slowness
			grdpts_y: ind, total points in y slowness
		'''

		sx = sll_x + np.arange(grdpts_x) * sl_s
		sy = sll_y + np.arange(grdpts_y) * sl_s
		delta_x = sx - tsx
		delta_y = sy - tsy
		mx = np.outer(self.geometry['RX'],delta_x)
		my = np.outer(self.geometry['RY'],delta_y)
		timeTable = np.require(mx[:,:,np.newaxis].repeat(grdpts_y,axis=2) +
							   my[:,np.newaxis,:].repeat(grdpts_x,axis=1))
		tcos = np.mean(np.cos(2.0*PI*freq*timeTable),axis=0)
		tsin = np.mean(np.sin(2.0*PI*freq*timeTable),axis=0)

		val = 10.0*np.log10(tcos**2 + tsin**2)

		arf = {'tsx':tsx, 'tsy':tsy,'freq':freq,
			   'sll_x':sll_x, 'sll_y':sll_y,'sl_s':sl_s,
			   'grdpts_x':grdpts_x,'grdpts_y':grdpts_y,
			   'arf':val}

		self.arf = arf

	def plotARF(self):
		arf = self.arf
		xmax = arf['sll_x'] + arf['sl_s'] * arf['grdpts_x']
		ymax = arf['sll_y'] + arf['sl_s'] * arf['grdpts_y']
		extent = [arf['sll_x'], xmax, arf['sll_y'], ymax]
		fig, ax = plt.subplots(figsize=(8,6))
		a = ax.imshow(arf['arf'], extent=extent, aspect='auto', cmap='Reds_r')
		ax.set_title('Array response for %s in frequency %s\ntargeting sx=%.2f, sy=%.2f'
					%(self.name, arf['freq'], arf['tsx'],arf['tsy']))
		fig.colorbar(a,ax=ax)
		ax.set_ylabel('Slowness Y')
		ax.set_xlabel('Slowness X')
		plt.show()

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
			eq.slideBeam(geometry=self.geometry,arrayName=self.name,
						 grdpts_x=grdpts_x,grdpts_y=grdpts_y,
						 filts=filts,
						 sflag=sflag,stack=stack,
						 sll_x=sll_x,sll_y=sll_y,sl_s=sl_s,refine=refine,
						 starttime=starttime,endtime=endtime, unit=unit,
						 winlen=winlen,overlap=overlap,write=write, **kwargs)

	def slideFK(self, winlen=2.0, overlap=0.5, grdpts_x=301.0, grdpts_y=301.0,
				sll_x=-15.0, sll_y=-15.0, sl_s=0.1, starttime=500, endtime=1000,
				method=0, prewhiten=0, freqmin=1, freqmax=2, write=False, **kwargs):
		for eq in self.events:
			eq.slideFK(timeTable=self.timeTable, arrayName=self.name,
					   winlen=winlen, overlap=overlap, 
					   sll_x=sll_x, sll_y=sll_y, sl_s=sl_s,
					   grdpts_x=grdpts_x, grdpts_y=grdpts_y,
					   freqmin=freqmin, freqmax=freqmax,
					   starttime=starttime, endtime=endtime,
					   prewhiten=prewhiten, method=method, write=write,
					   **kwargs)

	def vespectrum(self,grdpts=401,
					filts={'filt_1':[1,2,4,True],'filt_2':[2,4,4,True],'filt_3':[1,3,4,True]},
					stack='linear',
					sl_s=0.1, vary='slowness',sll=-20.0,
					starttime=400.0,endtime=1400.0, unit='deg',
					**kwargs ):
		for eq in self.events:
			eq.vespectrum(geometry=self.geometry,arrayName=self.name,grdpts=grdpts,
					filts=filts, stack=stack,
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
		# self.time = UTCDateTime(row.TIME)
		# self.ID = row.DIR
		# self.lat = row.LAT
		# self.lon = row.LON
		# self.dep = row.DEP
		# self.mw = row.Mw
		# self.dis = row.Del
		# self.az = row.Az
		# self.baz = row.Baz
		# self.bb = row.BB
		# self.rayp = row.Rayp
		# self.takeOffAngle = row.Angle
		# self._defOri()
		# self._updateOri(row)
		self.__dict__.update((k.lower(), v) for k,v in row.items())
		self.ID = row.DIR
		self.beamphase = beamphase
		self.phase_list = phase_list
		# self.stream = row.Stream
		self.ntr = len(row.Stream)
		self.delta = row.Stream[0].stats.delta
		self._checkInputs()

	# def _defOri(self):
	# 	self.time = -12345
	# 	self.ID = -12345
	# 	self.lat = -12345
	# 	self.lon = -12345
	# 	self.dep = -12345
	# 	self.mw = -12345
	# 	self.dis = -12345
	# 	self.az = -12345
	# 	self.baz = -12345
	# 	self.bb = -12345
	# 	self.rayp = -12345
	# 	self.takeOffAngle = -12345

	def _updateOri(self, row):
		self.__dict__.update((k, v) for k,v in row.items())

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

		if not hasattr(self, 'dep') or self.dis==-12345:
			msg = "Depth or distance for %s is not defined"%self.ID
			gpar.log(__name__, msg, level='error', pri=True)

		arrivals = model.get_travel_times(source_depth_in_km=self.dep,distance_in_degree=self.dis,phase_list=phase_list)

		phases = {}
		for arr in arrivals:
			pha = arr.name
			times = {'UTC':self.time + arr.time,
					 'TT':arr.time,
					 'RP':arr.ray_param_sec_degree}
			phases[pha] = times

		self.arrivals = phases
		msg = ('Travel times for %s for earthquake %s in depth of %.2f in distance of %.2f' % (phase_list, self.ID, self.dep, self.dis))
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
		if not hasattr(self, 'rayp') or not hasattr(self, 'baz'):
			msg = "Ray parameter or back azimuth is not defined for %s"%self.ID
			gpar.log(__name__, msg, level='error', pri=True)
		tsDF = getTimeShift(self.rayp,self.baz,geometry,unit=unit)
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
			beamTr = beamForming(tmp_st, tsDF, npt, starttime, stack=stack)
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

	def slideBeam(self,geometry,arrayName,grdpts_x=301,grdpts_y=301,
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
							geometry,arrayName,grdpts_x,grdpts_y,
						filt, sflag,stack,sll_x,sll_y,sl_s,refine,
						starttime,endtime, unit,
						winlen,overlap,**kwargs)
			st = obspy.core.stream.Stream()
			otime = self.time
			ndel = winlen*(1-overlap)
			bnpts = int((endtime-starttime)/ndel)
			label = ['Amplitude','Slowness','Back Azimuth','coherence']
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

	def slideFK(self, timeTable, arrayName, winlen=2.0, overlap=0.5, grdpts_x=301, grdpts_y=301,
				sll_x=-15.0, sll_y=-15.0, sl_s=0.1, starttime=500, endtime=1000,
				method=0, prewhiten=0, freqmin=1, freqmax=2, write=False, **kwargs):
		ntr = self.ntr
		st = self.stream.copy()
		st.detrend('demean')
		msg = "Calculating FK beamforming for %s in in frequency %s-%s"%(self.ID, freqmin, freqmax)
		gpar.log(__name__, msg, level='info', pri=True)

		rel = fkBeam(stream=st, ntr=ntr, time_shift_table=timeTable, 
					 winlen=winlen, overlap=overlap, 
					 sll_x=sll_x, sll_y=sll_y, sl_s=sl_s,
					 grdpts_x=grdpts_x, grdpts_y=grdpts_y,
					 freqmin=freqmin, freqmax=freqmax,
					 starttime=starttime, endtime=endtime,
					 prewhiten=prewhiten, method=method)
		fkst = obspy.core.stream.Stream()
		otime = self.time
		ndel = winlen*(1-overlap)
		bnpts = int((endtime-starttime)/ndel)
		label = ['Absamp', 'Slowness','Back Azimuth','Relamp']

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
			fkst.append(tr)

		self.fkSt = fkst

		if write:
			bpfilt = str(freqmin) + '-'+ str(freqmax)
			name = 'fk.' + self.ID +'.' +bpfilt+'.pkl'
			name = os.path.join('./',arrayName, 'Data', self.ID, name)
			st.write(name, format='PICKLE')

	#@profile
	def vespectrum(self,geometry,arrayName,grdpts=401,
					filts={'filt_1':[1,2,4,True],'filt_2':[2,4,4,True],'filt_3':[1,3,4,True]},
					stack='linear',
					sl_s=0.1, vary='slowness',sll=-20.0,
					starttime=400.0,endtime=1400.0, unit='deg',
					**kwargs ):
		msg = ('Calculating vespetrum for earthquake %s in array %s' % (self.ID, arrayName))
		gpar.log(__name__,msg,level='info',pri=True)
		self.slantType = vary
		self.slantK = sll + np.arange(grdpts)*sl_s
		winlen = endtime - starttime
		delta = self.delta
		winpts = int(winlen/delta) + 1
		self.slantTime = delta * np.arange(winpts) + starttime
		if vary == 'slowness':
			self.energy = slantBeam(self.stream, self.ntr, delta,
									geometry, arrayName, grdpts,
									filts, stack, sl_s, vary, sll,starttime,
									endtime, unit, bakAzimuth=self.baz)
		elif vary == 'theta':
			self.energy = slantBeam(self.stream, self.ntr, delta,
									geometry,  arrayName, grdpts,
									filts, stack, sl_s, vary, sll,starttime,
									endtime, unit, rayParameter=self.rayp)
		# self.slantTime = times

		# self.slantK = k
		# self.energy = abspow

	def plotves(self, show=True, saveName=False,**kwargs):

		extent=[np.min(self.slantTime),np.max(self.slantTime),np.min(self.slantK),np.max(self.slantK)]
		fig, ax = plt.subplots(1, len(self.energy.keys()),figsize=(8,6),sharey='row')
		for ind, row in self.energy.iterrows():
			im = ax[ind].imshow(np.log10(row.POWER),extent=extent, aspect='auto',**kwargs)
			name = self.ID + '.'+row.FILT + self.slantType
			ax[ind].set_title(name)
			ax[ind].set_xlabel('Time (s)')
		if self.slantType == 'slowness':
			unit = 's/deg'
			a = u"\u00b0"
			title = 'Slant Stack at a Backazimuth of %.1f %sN'%(self.baz,a)
		elif self.slantType == 'theta':
			unit = 'deg'
			title = 'Slant Stack at a slowness of %.2f s/deg'%(self.rayp)

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
				 rstime=20.0, retime=50.0,
				 domain='freq', fittype='cos',
				 tphase='PKIKP', rphase='PP',
				 phase_list=['PKiKP','PKIKP','PKP','PP'],
				 threshold=0.4,cut=10,tshift=5):

		self.ID = row.DoubleID
		msg = ("Building %s: EV1: %s; EV2: %s"%(self.ID,row.TIME1, row.TIME2))
		gpar.log(__name__, msg, level='info', pri=True)
		self.ev1 = {'TIME': UTCDateTime(row.TIME1), 'LAT': row.LAT1,
					'LON': row.LON1, 'DEP': row.DEP1,
					'MAG': row.M1}
		self.ev2 = {'TIME': UTCDateTime(row.TIME2), 'LAT': row.LAT2,
					'LON': row.LON2, 'DEP': row.DEP2,
					'MAG': row.M2}
		self.phase_list = phase_list
		self.tphase = tphase
		self.rphase = rphase
		self.delta = resample
		self.winlen = winlen
		self.step = step
		self.threshold = threshold
		self.cut = cut
		self.st1 = row.ST1
		self.st2 = row.ST2
		self.st1.detrend('demean')
		self.st2.detrend('demean')
		self._checkInput()
		self._getDel(array)
		
		self.getArrival(array=array)
		self._alignWave(filt=filt,delta=resample,cstime=cstime,cetime=cetime,
						rstime=rstime,retime=retime,
						domain=domain,fittype=fittype,
						threshold=threshold,cut=cut,tshift=tshift)

	def _checkInput(self):
		if not isinstance(self.st1, obspy.core.stream.Stream):
			msg = ('Waveform data for %s is not stream, stop running' % self.ID)
			gpar.log(__name__,msg,level='error',pri=True)
		if not isinstance(self.st2, obspy.core.stream.Stream):
			msg = ('Waveform data for %s is not stream, stop running' % self.ID)
			gpar.log(__name__,msg,level='error',pri=True)
	def _resample(self, st1, st2,resample, method, npts):
		rrate = 1.0/resample
		if method == 'resample':
			st1.resample(sampling_rate=rrate)
			st2.resample(sampling_rate=rrate)
		elif method == 'interpolate':
			st1.interpolate(sampling_rate=rrate)
			st2.interpolate(sampling_rate=rrate)
		else:
			msg = ('Not a valid option for waveform resampling, choose from resample and interpolate')
			gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
		for _tr1, _tr2 in zip_longest(st1, st2):
			data1 = _tr1.data
			data2 = _tr2.data
			if len(data1) > npts:
				_tr1.data = data1[0:npts]
			elif len(data1) < npts:
				data1 = np.pad(data1, (0,npts-len(data1)), 'edge')
				_tr1.data = data1
			if len(data2) > npts:
				_tr2.data = data2[0:npts]
			elif len(data2) < npts:
				data2 = np.pad(data2, (0,npts-len(data2)), 'edge')
				_tr2.data = data2

		return st1, st2

	def _getDel(self, array):
		olat = (self.ev1['LAT'] + self.ev2['LAT'])/2.0
		olon = (self.ev1['LON'] + self.ev2['LON'])/2.0
		dis, az, bakAzi = gpar.getdata.calc_Dist_Azi(olat, olon, array.refPoint[0], array.refPoint[1])
		self.dis = {'DEL': dis, 'AZ':az,'bakAzi':bakAzi}

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
			phase = self.phase_list
		arr1 = model.get_travel_times_geo(source_depth_in_km=self.ev1['DEP'],source_latitude_in_deg=self.ev1['LAT'],
										  source_longitude_in_deg=self.ev1['LON'],phase_list=phase,
										  receiver_latitude_in_deg=array.refPoint[0], receiver_longitude_in_deg=array.refPoint[1])
		pha1 = {}
		for a in arr1:
			pha = a.name
			times = {'UTC':self.ev1['TIME'] + a.time,
					 'TT':a.time,
					 'RP':a.ray_param_sec_degree}
			if pha in pha1:
				pha1[pha].append(times)
			else:
				pha1[pha] = [times]
		self.arr1 = pha1
		arr2 = model.get_travel_times_geo(source_depth_in_km=self.ev2['DEP'],source_latitude_in_deg=self.ev2['LAT'],
										  source_longitude_in_deg=self.ev2['LON'],phase_list=phase,
										  receiver_latitude_in_deg=array.refPoint[0], receiver_longitude_in_deg=array.refPoint[1])

		pha2 = {}
		for a in arr2:
			pha = a.name
			times = {'UTC':self.ev2['TIME'] + a.time,
					 'TT':a.time,
					 'RP':a.ray_param_sec_degree}

			if pha in pha2:
				pha2[pha].append(times)
			else:
				pha2[pha] = [times]
		self.arr2 = pha2
		# msg = ('Travel time for %s for earthquake %s in depth of %.2f in distance of %.2f is %s' % (phase, self.ID, self.dep, self.dis, self.arrival))

		# gpar.log(__name__,msg,level='info',pri=True)
	
	def beamForming(self,geometry, starttime=0.0, winlen=1800.0,
					filt=[1, 3, 4, True], stack='linear', unit='deg',
					cstime=20, cetime=20,method='resample', delta=0.01,
					channel='SHZ',retime=10,rstime=60,
					step=0.05, steplen=3,dv=0.02,nbtrial=401,
					stret_method='resample',
					**kwargs):
		# if not hasattr(self, 'rap1') or not hasattr(self, 'rap2'):
		# 	msg = "Ray parameters are not defined, calculating ray parameters"
		# 	gpar.log(__name__, msg, level='info', pri=True)
		# 	self.getArrival(array=kwargs['array'],phase=kwargs['beamphase'],model=kwargs['model'])
		tphase = self.tphase
		rphase = self.rphase
		tRayp = self.arr1[tphase][0]['RP']
		rRayp = self.arr1[rphase][0]['RP']
		self.cstime = cstime
		self.cetime = cetime 
		self.rstime = rstime 
		self.retime = retime

		tsDF = getTimeShift(tRayp, self.dis['bakAzi'], geometry, unit=unit)
		refDF = getTimeShift(rRayp, self.dis['bakAzi'], geometry, unit=unit)
		# tsDF1 = tsDF.copy()
		# tsDF2 = tsDF.copy()
		stalist = tsDF.STA
		refsta = refDF.STA
		# tsDF2 = getTimeShift(self.rayp2, self.dis['bakAzi'], geometry, unit=unit)
		# stalist2 = tsDF2.STA
		st1 = self.st1.copy()
		st1 = st1.select(channel=channel)
		st2 = self.st2.copy()
		st2 = st2.select(channel=channel)
		st1.filter('bandpass', freqmin=filt[0], freqmax=filt[1], corners=filt[2], zerophase=filt[3])
		st2.filter('bandpass', freqmin=filt[0], freqmax=filt[1], corners=filt[2], zerophase=filt[3])
		delta1 = st1[0].stats.delta
		delta2 = st2[0].stats.delta
		tsDF1 = getDT(st1, tsDF, stalist, delta1)
		tsDF2 = getDT(st2, tsDF, stalist, delta2)
		refDF1 = getDT(st1, refDF, refsta, delta1)
		refDF2 = getDT(st2, refDF, refsta, delta2)
		npt1 = int(winlen/delta1) + 1
		npt2 = int(winlen/delta2) + 1
		msg = ('Calculating beamforming for doublet %s - event 1 %s'%(self.ID, self.ev1['TIME']))
		gpar.log(__name__,msg, level='info', pri=True)
		beamTr1 = beamForming(st1, tsDF1, npt1, starttime, stack=stack)
		beamTr1.stats.starttime = self.ev1['TIME'] + starttime
		beamTr1.stats.channel = tphase
		beamRefTr1 = beamForming(st1, refDF1, npt1, starttime, stack=stack)
		beamRefTr1.stats.starttime = self.ev1['TIME'] + starttime
		beamRefTr1.stats.channel = rphase
		msg = ('Calculating beamforming for doublet %s - event 2 %s'%(self.ID, self.ev2['TIME']))
		gpar.log(__name__,msg, level='info', pri=True)
		beamTr2 = beamForming(st2, tsDF2, npt2, starttime, stack=stack)
		beamTr2.stats.starttime = self.ev2['TIME'] + starttime
		beamTr2.stats.channel = tphase
		beamRefTr2 = beamForming(st2, tsDF2, npt2, starttime, stack=stack)
		beamRefTr2.stats.starttime = self.ev2['TIME'] + starttime
		beamRefTr2.stats.channel = rphase
		self.beamSt1 = obspy.core.stream.Stream()
		self.beamSt1.append(beamTr1.copy())
		self.beamSt1.append(beamRefTr1.copy())
		self.beamSt2 = obspy.core.stream.Stream()
		self.beamSt2.append(beamTr2.copy())
		self.beamSt2.append(beamRefTr2.copy())
		stime1 = self.arr1[tphase][0]['UTC'] - cstime
		etime1 = self.arr1[tphase][0]['UTC'] + cetime
		rstime1 = self.arr1[rphase][0]['UTC'] - rstime
		retime1 = self.arr1[rphase][0]['UTC'] + retime
		stime2 = self.arr2[tphase][0]['UTC'] - cstime
		etime2 = self.arr2[tphase][0]['UTC'] + cetime
		rstime2 = self.arr2[rphase][0]['UTC'] - rstime
		retime2 = self.arr2[rphase][0]['UTC'] + retime
		tst1 = obspy.core.stream.Stream()
		tst2 = obspy.core.stream.Stream()
		rst1 = obspy.core.stream.Stream()
		rst2 = obspy.core.stream.Stream()
		beamTr1.trim(starttime=stime1, endtime=etime1)
		tst1.append(beamTr1)
		beamRefTr1.trim(starttime=rstime1, endtime=retime1)
		rst1.append(beamRefTr1)
		beamTr2.trim(starttime=stime2, endtime=etime2)
		tst2.append(beamTr2)
		beamRefTr2.trim(starttime=rstime2, endtime=retime2)
		rst2.append(beamRefTr2)
		# bst1 = obspy.core.stream.Stream()
		# bst2 = obspy.core.stream.Stream()
		# bst1.append(beamTr1);bst1.append(beamRefTr1)
		# bst2.append(beamTr2);bst2.append(beamRefTr2)

		taups = []
		npts = int((cetime + cstime)/delta) + 1
		tst1, tst2 = self._resample(tst1, tst2, delta, method, npts)
		Mptd1 = np.zeros([1, npts])
		Mptd2 = np.zeros([1, npts])
		Mptd1[0,:] = tst1[0].data
		Mptd2[0,:] = tst2[0].data
		taup, ccmax, dt, cc = _getLag(Mptd1, Mptd2, delta, domain='freq', fittype='cos', getcc=True)
		taups.append(taup[0])
		npts = int((retime + rstime)/delta) + 1
		rst1, rst2 = self._resample(rst1, rst2, delta, method, npts)
		Mptd1 = np.zeros([1, npts])
		Mptd2 = np.zeros([1, npts])
		Mptd1[0,:] = rst1[0].data
		Mptd2[0,:] = rst2[0].data
		taup, ccmax, dt, cc = _getLag(Mptd1, Mptd2, delta, domain='freq', fittype='cos', getcc=True)
		taups.append(taup[0])
		self.tshift = taups

		use_st1 = self.beamSt1.copy()
		use_st2 = self.beamSt2.copy()
		inds = len(use_st1)
		ttr1 = use_st1[0]
		ttr2 = use_st2[0]

		rtr1 = use_st1[1]
		rtr2 = use_st2[1]

		t_shift = -cstime - taups[0]
		r_shift = -rstime - taups[1]

		stime1 = self.arr1[tphase][0]['UTC'] + t_shift
		stime2 = self.arr2[tphase][0]['UTC'] - cstime 

		ttr1.trim(starttime=stime1, endtime=stime1 + cstime + cetime + steplen*2)
		ttr2.trim(starttime=stime2, endtime=stime2 + cstime + cetime + steplen*2)

		stime1 = self.arr1[rphase][0]['UTC'] + r_shift
		stime2 = self.arr2[rphase][0]['UTC'] - rstime

		rtr1.trim(starttime=stime1, endtime=stime1 + rstime + retime + steplen*2)
		rtr2.trim(starttime=stime2, endtime=stime2 + rstime + retime + steplen*2)
		
		tst1 = obspy.core.stream.Stream()
		tst2 = obspy.core.stream.Stream()
		rst1 = obspy.core.stream.Stream()
		rst2 = obspy.core.stream.Stream()
		
		tst1.append(ttr1)
		tst2.append(ttr2)
		npts = int((cetime + cstime + steplen*2)/delta) + 1
		tst1, tst2 = self._resample(tst1, tst2, delta, method, npts)

		rst1.append(rtr1)
		rst2.append(rtr2)
		npts = int((retime + rstime + steplen*2)/delta) + 1
		rst1, rst2 = self._resample(rst1, rst2, delta, method, npts)

		npts = int(steplen / delta)
		target_ts = []
		target_cc = []
		t_dvs = []
		for win_st1, win_st2 in zip_longest(tst1.slide(steplen, step), tst2.slide(steplen, step)):
			_taup, _cc = codaInt(win_st1, win_st2, delta=delta,
								npts=npts, domain='freq',
								fittype='cos')
			target_cc.append(_cc)
			target_ts.append(_taup)
			_dv = codaStr(tst1, tst2, delta, win_st1, win_st2, steplen, dv,nbtrial,stret_method)
			t_dvs.append(_dv)

		
		ref_ts = []
		ref_cc = []
		r_dvs = []

		for win_st1, win_st2 in zip_longest(rst1.slide(steplen, step), rst2.slide(steplen, step)):
			_taup, _cc = codaInt(win_st1, win_st2, delta=delta,
								npts=npts, domain='freq',
								fittype='cos')
			ref_cc.append(_cc)
			ref_ts.append(_taup)
			_dv = codaStr(rst1, rst2, delta, win_st1, win_st2, steplen, dv,nbtrial,stret_method)
			r_dvs.append(_dv)

		self.tcc=target_cc
		self.ttaup=target_ts
		self.rcc=ref_cc
		self.rtaup=ref_ts
		self.tdv = t_dvs
		self.rdv = r_dvs
	
	def plotBeam(self,delta=0.01, 
				 step=0.05, steplen=3, 
				 savefig=True):

		fig, ax = plt.subplots(4, 2, sharey='row', sharex='col',figsize=(6.4, 7.2),constrained_layout=True)
		

		ax[0,0].set_title(self.tphase)
		ax[0,1].set_title(self.rphase)
		ax[0,0].set_ylabel('Normal Amp')
		ax[1,0].set_ylabel('Taup')
		ax[2,0].set_ylabel('CC')
		# plt.subplot(3,2,1)
		taup = self.tshift 
		# cc = self.beamcc
		# npts = int((tstart[ind] + tend[ind])/delta) + 1
		# dt = (np.arange(len(cc)) - (npts -1)) * delta
		st1 = self.beamSt1.copy()
		st2 = self.beamSt2.copy()
		phase = [self.tphase, self.rphase]
		taups = [self.ttaup, self.rtaup]
		t_point = steplen/2
		twin = len(self.ttaup) * step + t_point
		rwin = len(self.rtaup) * step + t_point
		win_len = [twin, rwin]
		ccs = [self.tcc, self.rcc]
		dvs = [self.tdv, self.rdv]
		tstart = [self.cstime, self.rstime]
		tend = [self.rstime, self.retime]
		for ind in range(2):
			tr1 = st1[ind].copy()
			thiftBT = -tstart[ind]-taup[ind]
			stime = self.arr1[phase[ind]][0]['UTC'] + thiftBT
			tr1.trim(starttime=stime, endtime=stime+win_len[ind])
			tr2 = st2[ind].copy()
			stime2 = self.arr2[phase[ind]][0]['UTC']-tstart[ind]
			tr2.trim(starttime=stime2, endtime=stime2+win_len[ind])
			delta1 = tr1.stats.delta
			n_ind1 = -int((1+steplen*2)/delta1)
			delta2 = tr2.stats.delta
			n_ind2 = -int((1+steplen*2)/delta2)
			data1 = tr1.data/np.max(np.absolute(tr1.data[:n_ind1]))
			data2 = tr2.data/np.max(np.absolute(tr2.data[:n_ind2]))
			t1 = self.arr2[phase[ind]][0]['TT'] - tstart[ind] + np.arange(len(data1))*delta1
			t2 = self.arr2[phase[ind]][0]['TT'] - tstart[ind] + np.arange(len(data2))*delta2
			ax[0,ind].plot(t1, data1, 'b', linewidth=0.5)
			ax[0,ind].plot(t2, data2, 'r-.', linewidth=0.5)
			# ax[0,ind].axvline(x=self.arr2[phase[ind]]['TT'], c='k')
			ax[0,ind].set_xlim([np.min(t1), np.max(t1)])
			ax[0,ind].set_ylim([-1.1, 1.1])
			# ttaup = taups[0]
			ts = self.arr2[phase[ind]][0]['TT'] - tstart[ind] + np.arange(len(taups[ind])) * step + t_point
			ax[1,ind].plot(ts, taups[ind], linewidth=0.5)
			# ax[1,ind].axvline(x=self.arr2[phase[ind]]['TT'],c='k')
			ax[1,ind].set_ylim([-2, 2])
			# ax[1,0].set_ylabel('Taup')
			ax[2,ind].plot(ts, ccs[ind], linewidth=0.5)
			# ax[2,ind].axvline(x=self.arr2[phase[ind]]['TT'],c='k')
			ax[2,ind].set_ylim([0, 1])

			ax[3,ind].plot(ts, dvs[ind], linewidth=0.5)
			# ax[3,ind].axvline(x=self.arr2[phase[ind]]['TT'],c='k')
			ax[3,ind].set_ylim([-0.021, 0.021])
		# ax[2,0].set_ylabel('CC')

		# plt.title('Doublet %s: \n%s-%s\nDistance: %.2f\nMax CC %.3f - TimeShift %.4f'%(self.ID, self.ev1['TIME'], self.ev2['TIME'], self.dis['DEL'],self.ccmax, taup))
		# plt.subplot(3,1,2)
		# plt.plot(dt, cc)
		# plt.ylabel('CC')
		fig.suptitle('Doublet %s: \n%s-%s\nDistance: %.2f\n'%(self.ID, self.ev1['TIME'], self.ev2['TIME'], self.dis['DEL']))
		
		# plt.tight_layout()

		if savefig:
			savename = self.ID + '.' + 'beam.eps'
			plt.savefig(savename)
			plt.close()

	def slownessFK(self,geometry, grdpts_x=401, grdpts_y=401, 
				   sll_x=-0.02, sll_y=-0.02, sl_s=0.0001,
				   starttime=15, endtime=30, method=0, 
				   prewhiten=0, freqmin=1, freqmax=3,
				   unit='km', channel='SHZ',tphase='PKIKP'):
		st1 = self.st1.copy()
		st1 = st1.select(channel=channel)
		use_geometry_1 = pd.DataFrame()
		for tr in st1:
			sta = tr.stat.station
			trow = geometry[geometry.STA == sta].iloc[0]
			use_geometry_1 = use_geometry_1.append(trow, ignore_index=True)
		st2 = self.st2.copy()
		st2 = st2.select(channel=channel)
		use_geometry_2 = pd.DataFrame()
		for tr in st2:
			sta = tr.stat.station
			trow = geometry[geometry.STA == sta].iloc[0]
			use_geometry_2 = use_geometry_2.append(trow, ignore_index=True)
		self.sll_x = sll_x
		self.sll_y = sll_y
		self.sl_s = sl_s
		self.sphase = tphase
		self.fk_stime = starttime
		self.fk_etime = endtime 
		ntr1 = len(st1)
		ntr2 = len(st2)

		win = starttime + endtime
		arr1 = self.arr1[tphase][0]['UTC']
		stime = arr1 - starttime 
		etime = arr1 + endtime
		st1.trim(starttime=stime, endtime=etime)
		timeTable1 = getTimeTable(use_geometry_1, sll_x, sll_y, sl_s, grdpts_x, grdpts_x,unit=unit)
		relpow1, abspow1 = fkslowness(st1, ntr1, timeTable1, 
									  sll_x=sll_x, sll_y=sll_y,
									  sl_s=sl_s, grdpts_x=grdpts_x, grdpts_y=grdpts_y,
									  freqmin=freqmin, freqmax=freqmax,
									  starttime=0, endtime=win,
									  prewhiten=prewhiten, method=method)
		arr2 = self.arr2[tphase][0]['UTC']
		stime = arr2 - starttime
		etime = arr2 + endtime
		st2.trim(starttime=stime, endtime=etime)
		timeTable2 = getTimeTable(use_geometry_2, sll_x, sll_y, sl_s, grdpts_x, grdpts_x,unit=unit)
		relpow2, abspow2 = fkslowness(st2, ntr2, timeTable2, 
									  sll_x=sll_x, sll_y=sll_y,
									  sl_s=sl_s, grdpts_x=grdpts_x, grdpts_y=grdpts_y,
									  freqmin=freqmin, freqmax=freqmax,
									  starttime=0, endtime=win,
									  prewhiten=prewhiten, method=method)
		self.relpow1 = relpow1
		self.relpow2 = relpow2
		self.difpow = abspow1/abspow1.max() - abspow2/abspow2.max()
		
		self.abspow1 = abspow1
		self.abspow2 = abspow2

	def slantBeam(self, geometry, arrayName, grdpts=101, sll=-0.1, 
				  filts={'filt_1':[1,2,3,True],'filt_2':[1,3,3,True]},
				  stack='linear',sl_s=0.001,
				  unit='km', starttime=15, 
				  channel='SHZ', delta=0.01,
				  endtime=30, tphase='PKIKP',**kwargs):

		winlen = starttime + endtime 
		st1 = self.st1.copy()
		st2 = self.st2.copy()
		st1 = st1.select(channel=channel)
		st2 = st2.select(channel=channel)
		st1.resample(sampling_rate=1.0/delta)
		st2.resample(sampling_rate=1.0/delta)
		winpts = int(winlen/delta) + 1
		self.slantTime = delta * np.arange(winpts) - starttime
		self.vstime = starttime
		self.vetime = endtime
		baz = self.dis['bakAzi']
		arr1 = self.arr1[tphase][0]['UTC']
		time1 = self.ev1['TIME']
		stime = arr1 - starttime - time1
		etime = arr1 + endtime  - time1
		self.energy1 = slantBeam(st1, len(st1), delta,
								 geometry, arrayName, grdpts,
								 filts, stack, sl_s, 'slowness',
								 sll, stime, etime, unit, bakAzimuth=baz)

		arr2 = self.arr2[tphase][0]['UTC']
		time2 = self.ev2['TIME']
		stime = arr2 - starttime - time2
		etime = arr2 + endtime  - time2
		self.energy2 = slantBeam(st2, len(st2), delta,
								 geometry, arrayName, grdpts,
								 filts, stack, sl_s, 'slowness',
								 sll, stime, etime, unit, bakAzimuth=baz)

	def _alignWave(self,filt=[1, 3,4,True],delta=0.01,
				   cstime=20.0, cetime=50.0, method='resample',
				   rstime=20.0, retime=50.0,
				   domain='freq', fittype='cos',
				   threshold=0.4, cut=10,tshift=5):
		# msg = ('Aligning waveforms for doublet %s'%(self.ID))
		# gpar.log(__name__, msg, level='info', pri=True)
		sta1 = []
		sta2 = []
		tphase = self.tphase
		tmp_st1 = self.st1.copy()
		tmp_st2 = self.st2.copy()
		arr1 = self.arr1[tphase][0]['UTC']
		arr2 = self.arr2[tphase][0]['UTC']
		data = cutWaveForm(tmp_st1, tmp_st2, delta,
						   arr1, arr2, cstime, cetime,
						   filt,threshold,method,domain,
						   fittype,cut,tshift)
		if data == None:
			self._qual = False
		else:
			self.use_st1 = data[0]
			self.use_st2 = data[1]
			self.refTime = arr1 + data[2]
			self.align = data[3]
			self._qual=True
			rphase = self.rphase

			arr1 = self.arr1[rphase][0]['UTC']
			arr2 = self.arr2[rphase][0]['UTC']

			rdata = cutWaveForm(tmp_st1, tmp_st2,delta,
				arr1, arr2, rstime, retime,
				filt,threshold, method, domain,
				fittype, cut,tshift)

			if rdata != None:
				self.ref_st1 = rdata[0]
				self.ref_st2 = rdata[1]
				self.ref_time = arr1 + rdata[2]
				self.ref_align = rdata[3]
				self.ref_qual = True
			else:
				self.ref_qual = False
		
	def codaInter(self, delta=0.01,
				  winlen=5, step=0.05,
				  cstime=20.0,dv=0.02,
				  nbtrial=401,rstime=20.0,
				  domain='freq',fittype='cos',
				  stret_method='resample'):
		if not self._qual:
			msg = ('Waveforms for this doublet %s have quality issure, stop calculating'%self.ID)
			gpar.log(__name__, msg, level='warning', pri=True)
			return
		msg = ('Calculating Coda Interferometry for doublet %s'%self.ID)
		gpar.log(__name__, msg, level='info', pri=True)
		if winlen is None:
			winlen = self.winlen
		if step is None:
			step = self.step
		if delta is None:
			delta = self.delta
		st1 = self.use_st1.copy()
		st2 = self.use_st2.copy()
		#set starttime to be consistent in st1
		for tr in st1:
			tr.stats.starttime = self.refTime
		npts = int(winlen / delta)
		taup = []
		cc = []
		dvs = []
		# _i = 0
		for win_st1, win_st2 in zip_longest(st1.slide(winlen, step), st2.slide(winlen, step)):
			# print('running %d'%_i)
			# _i = _i + 1
			_taup, _cc = codaInt(win_st1, win_st2, delta=delta, 
								 npts=npts,domain=domain,
								 fittype=fittype)
			taup.append(_taup)
			cc.append(_cc)
			_dv = codaStr(st1, st2, delta, win_st1, win_st2, winlen,dv,nbtrial, stret_method)
			dvs.append(_dv)

		self.taup = taup
		self.cc = cc
		self.dv = dvs

		tpts = len(taup)
		ts = self.arr2[self.tphase][0]['TT'] + np.arange(tpts) * step - cstime
		self.ts = ts

		if not self.ref_qual:
			msg = ('Waveforms for this doublet %s have quality issure, stop calculating'%self.ID)
			gpar.log(__name__, msg, level='warning', pri=True)
			self.ref_ts = None
			self.ref_cc = None
			self.ref_dv = None
			return
		
		st1 = self.ref_st1.copy()
		st2 = self.ref_st2.copy()
		for tr in st1:
			tr.stats.starttime = self.ref_time
		ref_taup = []
		ref_cc = []
		ref_dv = []

		for win_st1, win_st2 in zip_longest(st1.slide(winlen, step), st2.slide(winlen, step)):
			_taup, _cc = codaInt(win_st1, win_st2, delta=delta,
								 npts=npts, domain=domain,
								 fittype=fittype)
			ref_taup.append(_taup)
			ref_cc.append(_cc)
			_dv = codaStr(st1, st2, delta, win_st1, win_st2, winlen,dv,nbtrial,stret_method)
			ref_dv.append(_dv)

		self.ref_taup = ref_taup
		self.ref_cc = ref_cc
		self.ref_dv = ref_dv

		ts = self.arr2[self.rphase][0]['TT'] + np.arange(len(ref_taup))*step - rstime
		self.ref_ts = ts

	def updateFilter(self, filt, rstime=100, retime=300,
					cstime=20.0, cetime=20.0, tshift=5,
					winlen=None, step=None,dv=0.02, nbtrial=401):
		self._alignWave(filt=filt,delta=self.delta,
						cstime=cstime,cetime=cetime,
						rstime=rstime,retime=retime,
						threshold=self.threshold,
						cut=self.cut,tshift=tshift)
		self.codaInter(delta=None, winlen=winlen, step=step,cstime=cstime,
					   rstime=rstime,dv=dv,nbtrial=nbtrial)

	def plotCoda(self,cstime=20, cetime=10,
				 sta_id='CN.YKR9..SHZ',
				 ref_sta_id='CN.YKR9..SHZ',
				 rstime=20, retime=50,aphase='PKIKP',
				 filt=[1, 3, 3, True],
				 savefig=True, show=True):
		# fig = plt.figure()

		fig, ax = plt.subplots(4,2,figsize=(6.4, 7.2), sharey='row', constrained_layout=True)
		# ax.subplot(5,1,1)
		ax[0,0].set_title(self.tphase)
		ax[0,1].set_title(self.rphase)
		TTs = [self.arr2[self.tphase][0]['TT'], self.arr2[self.rphase][0]['TT']]
		# tlim = [TTs[0]-stime-, TTs[0]+etime-10]
		tlim = [np.min(self.ts), np.max(self.ts)]
		st1 = self.use_st1.copy()
		st2 = self.use_st2.copy()
		tr1 = st1.select(id=sta_id).copy()[0]
		tr2 = st2.select(id=sta_id).copy()[0]
		_ts = self.align[self.align.STA==sta_id].TS.iloc[0]
		tr1.trim(starttime=self.arr1[self.tphase][0]['UTC']-cstime-_ts, endtime=self.arr1[self.tphase][0]['UTC']+cetime-_ts)
		tr2.trim(starttime=self.arr2[self.tphase][0]['UTC']-cstime, endtime=self.arr2[self.tphase][0]['UTC']+cetime)
		data1 = tr1.data / np.max(np.absolute(tr1.data))
		data2 = tr2.data / np.max(np.absolute(tr2.data))
		t1 = TTs[0] - cstime + np.arange(len(data1)) * tr1.stats.delta
		t2 = TTs[0] - cstime + np.arange(len(data2)) * tr2.stats.delta
		ax[0,0].plot(t1, data1, 'b', linewidth=0.5)
		ax[0,0].plot(t2, data2, 'r-.', linewidth=0.5)
		ax[0,0].axvline(TTs[0], c='r',linewidth=0.5)
		if aphase != None and self.arr2.get(aphase) != None:
			ax[0,0].axvline(self.arr2[aphase][0]['TT'], c='g', linewidth=0.5)
		# plt.xlabel('Time (s)')
		ax[0,0].set_ylabel('Amp')
		ax[0,0].set_ylim([-1,1])
		ax[0,0].set_xlim(tlim)
		# ax[0].set_title('Doublet %s: \n%s-%s\nDistance: %.2f'%(self.ID, self.ev1['TIME'], self.ev2['TIME'], self.dis['DEL']))
		# plt.subplot(5, 1, 4)
		ax[1,0].plot(self.ts, self.cc,linewidth=0.5)
		ax[1,0].axvline(TTs[0], c='r',linewidth=0.5, linestyle='-.')
		# ax[0,1].axvline(TTs[1], c='b',linewidth=0.5, linestyle='-.')
		if aphase != None and self.arr2.get(aphase) != None:
			ax[1,0].axvline(self.arr2[aphase][0]['TT'], c='g', linewidth=0.5)
		ax[1,0].set_ylim([0, 1])
		ax[1,0].set_xlim(tlim)
		ax[1,0].set_ylabel('CC')
		# plt.subplot(5, 1, 5)
		ax[2,0].plot(self.ts, self.taup,linewidth=0.5)
		ax[2,0].axvline(TTs[0], c='r',linewidth=0.5, linestyle='-.')
		# ax[0,2].axvline(TTs[1], c='b',linewidth=0.5, linestyle='-.')
		if aphase != None and self.arr2.get(aphase) != None:
			ax[2,0].axvline(self.arr2[aphase][0]['TT'], c='g', linewidth=0.5)
		ax[2,0].set_ylabel('Tau')
		ax[2,0].set_ylim([-1,1])
		ax[2,0].set_xlim(tlim)
		ax[3,0].set_xlabel('Time (s)')
		ax[3,0].plot(self.ts, self.dv,linewidth=0.5)
		ax[3,0].set_xlim(tlim)
		ax[3,0].set_ylim([-0.02,0.02])
		ax[3,0].set_ylabel('dv/v')

		if self.ref_qual:
			rlim = [np.min(self.ref_ts), np.max(self.ref_ts)]
			st1 = self.ref_st1.copy()
			st2 = self.ref_st2.copy()
			tr1 = st1.select(id=ref_sta_id).copy()[0]
			tr2 = st2.select(id=ref_sta_id).copy()[0]
			_ts = self.ref_align[self.ref_align.STA==ref_sta_id].TS.iloc[0]
			tr1.trim(starttime=self.arr1[self.rphase][0]['UTC']-rstime-_ts, endtime=self.arr1[self.rphase][0]['UTC']+retime-_ts)
			tr2.trim(starttime=self.arr2[self.rphase][0]['UTC']-rstime, endtime=self.arr2[self.rphase][0]['UTC']+retime)
			data1 = tr1.data / np.max(np.absolute(tr1.data))
			data2 = tr2.data / np.max(np.absolute(tr2.data))
			t1 = TTs[1] - rstime + np.arange(len(data1)) * tr1.stats.delta
			t2 = TTs[1] - rstime + np.arange(len(data2)) * tr2.stats.delta
			ax[0,1].plot(t1, data1, 'b', linewidth=0.5)
			ax[0,1].plot(t2, data2, 'r-.', linewidth=0.5)
			ax[0,1].axvline(TTs[1], c='r', linewidth=0.5)
			ax[0,1].set_ylim([-1,1])
			ax[0,1].set_xlim(rlim)
			ax[1,1].plot(self.ref_ts, self.ref_cc, linewidth=0.5)
			ax[1,1].axvline(TTs[1],c='r',linewidth=0.5,linestyle='-.')
			ax[1,1].set_ylim([0,1])
			ax[1,1].set_xlim(rlim)
			ax[2,1].plot(self.ref_ts, self.ref_taup, linewidth=0.5)
			ax[2,1].axvline(TTs[1],c='r',linewidth=0.5,linestyle='-.')
			ax[2,1].set_ylim([-1,1])
			ax[2,1].set_xlim(rlim)
			ax[3,1].plot(self.ref_ts, self.ref_dv, linewidth=0.5)
			ax[3,1].axvline(TTs[1],c='r',linewidth=0.5,linestyle='-.')
			ax[3,1].set_ylim([-0.02,0.02])
			ax[3,1].set_xlim(rlim)
			ax[3,1].set_xlabel('Time (s)')

		fig.suptitle('Doublet %s: \n%s-%s\nDistance: %.2f'%(self.ID, self.ev1['TIME'], self.ev2['TIME'], self.dis['DEL']))

		if savefig:
			savename = self.ID + '.' + sta_id + '.eps'
			plt.savefig(savename)
			plt.close()
		if show:
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
		gpar.log(__name__,msg,level='error',pri=True,e=ValueError)


	tsx = geometry.X * sx
	tsy = geometry.Y * sy

	timeShift = -(tsx + tsy)
	d = {'STA':geometry.STA,'TimeShift':timeShift,'LAT':geometry.LAT,'LON':geometry.LON}
	timeDF = pd.DataFrame(d)

	return timeDF

def getTimeTable(geometry,sll_x=-15.0,sll_y=-15.0,sl_s=0.1,grdpts_x=301,grdpts_y=301,unit='deg'):
	"""
	Return timeshift table for given array geometry, modified from obspy

	"""
	sx = sll_x + np.arange(grdpts_x)*sl_s
	sy = sll_y + np.arange(grdpts_y)*sl_s

	if unit == 'deg':
		sx = sx/deg2km
		sy = sy/deg2km
	elif unit == 'rad':
		sx = sx/6370.0
		sy = sy/6370.0
	else:
		msg = ('Input not one of deg and rad, set unit as s/km')
		gpar.log(__name__,msg,level='warning',pri=True)


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
		gpar.log(__name__,msg,level='error',pri=True,e=ValueError)
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
		gpar.log(__name__,msg,level='error',pri=True,e=ValueError)

	tsx = np.outer(geometry.X, sx)
	tsy = np.outer(geometry.Y, sy)
	timeShift = (tsx + tsy)

	return timeShift

# def getSteer(ntr, deltaf, nsamp, time_shift_table, grdpts_x=301, grdpts_y=301, sl_s=0.1, freqmin=1.0, freqmax=2.0):
# 	'''
# 	Function to calculate steer matrix for FK
# 	Parameters:
# 		ntr: number of traces
# 		deltaf: delta of frequency
# 		time_shift_table: a numpy time shift table made by getTimeTable
# 		grdpts_x: number of points in x slowness
# 		grdpts_y: number of points in y slowness
# 		sl_s: interval of slowness
# 		filt: filter information
# 	'''

# 	nfft = next_pow_2(nsamp)
# 	nlow = int(freqmin/deltaf + 0.5)
# 	nhigh = int(freqmax/deltaf + 0.5)
# 	nlow = max(1, nlow)
# 	nhigh = min(nfft//2-1, nhigh) #avoid using nyquist
# 	nf = nhigh - nlow + 1
# 	steer = np.empty((nf, grdpts_x, grdpts_y, nstat), dtype=np.complex128)
# 	clibsignal.calcSteer(nstat, grdpts_x, grdpts_y, nf, nlow, deltaf, time_shift_table, steer)

# 	return steer
def get_spoint(stream, stime, etime):
	spoint = np.empty(len(stream), dtype=np.int32, order="C")
	epoint = np.empty(len(stream), dtype=np.int32, order="C")
	for i, tr in enumerate(stream):
		length = tr.stats.endtime - tr.stats.starttime
		if length < etime:
			msg = "Stream is shorter than %s sec" % (etime)
			gpar.log(__name__, msg, level='error', pri=True)
		spoint[i] = int(stime * tr.stats.sampling_rate + 0.5)
		epoint[i] = int(etime * tr.stats.sampling_rate + 0.5)

	return spoint, epoint

def beamForming(stream, tsDF, npt, starttime, stack='linear'):
	beamTr = obspy.core.trace.Trace()
	delta = stream[0].stats.delta
	beamTr.stats.delta = delta
	ntr = len(stream)
	if ntr != len(tsDF):
		msg = 'Station timeshift table is not matching with stream'
		gpar.log(__name__, msg, level='error', pri=True)
	if stack == 'psw':
		shiftTRdata = np.empty((ntr,npt),dtype=complex)
	else:
		shiftTRdata = np.empty((ntr,npt))
	for ind, tr in enumerate(stream):
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
	return beamTr 


def fkBeam(stream, ntr, time_shift_table,
		   winlen=2.0, overlap=0.5,
		   sll_x=-15, sll_y=-15, sl_s=0.1,
		   grdpts_x=301.0, grdpts_y=301.0, 
		   freqmin=1.0, freqmax=2.0, 
		   starttime=900, endtime=1300,
		   prewhiten=0, method=0, **kwargs):
	'''
	Function for seismic array fk/capon beamforming, calling obspy fk analysis function
	'''
	time_shift_table = time_shift_table.astype(np.float32)
	eotr = True
	res = []
	spoint, _epoint = get_spoint(stream, starttime, endtime)
	fs = stream[0].stats.sampling_rate
	nsamp = int(winlen * fs)
	ndel = winlen * (1-overlap)
	nfft = next_pow_2(nsamp)
	deltaf = fs/float(nfft)
	nlow = int(freqmin/deltaf + 0.5)
	nhigh = int(freqmax/deltaf + 0.5)
	nlow = max(1, nlow)
	nhigh = min(nfft//2-1, nhigh) #avoid using nyquist
	nf = nhigh - nlow + 1
	steer = np.empty((nf, grdpts_x, grdpts_y, ntr), dtype=np.complex128)
	clibsignal.calcSteer(ntr, grdpts_x, grdpts_y, nf, nlow, deltaf, time_shift_table, steer)
	nstep = int(nsamp * 1-(overlap))
	_r = np.empty((nf, ntr, ntr), dtype=np.complex128)
	ft = np.empty((ntr, nf), dtype=np.complex128)
	offset = 0
	tap = cosine_taper(nsamp, p=0.22)
	abspow_map = np.empty((grdpts_x, grdpts_y), dtype=np.float64)
	relpow_map = np.empty((grdpts_x, grdpts_y), dtype=np.float64)
	bnpts = int((endtime - starttime)/ndel)
	abspow = np.empty(bnpts)
	relpow = np.empty(bnpts)
	slow_h = np.empty(bnpts)
	bakaz = np.empty(bnpts)

	for ind in range(bnpts):
		try:
			for i, tr in enumerate(stream):
				dat = tr.data[spoint[i]+offset: spoint[i]+offset+nsamp]
				dat = (dat - dat.mean()) * tap
				ft[i, :] = np.fft.rfft(dat, nfft)[nlow:nlow+nf]
		except:
			msg = 'Problem in extat data and do the DFFT'
			gpar.log(__name__, msg, level='error', pri=True)
		ft = np.ascontiguousarray(ft, np.complex128)
		relpow_map.fill(0.)
		abspow_map.fill(0.)

			# Computing the covariances of the signal at different receivers
		dpow = 0.0
		for i in range(ntr):
			for j in range(i, ntr):
				_r[:, i, j] = ft[i,:] * ft[j,:].conj()
				if method == 1:
					_r[:, i, j] /= np.abs(_r[:, i, j].sum())
				if i != j:
					_r[:, j, i] = _r[:, i, j].conjugate()
				else:
					dpow += np.abs(_r[:, i, j].sum())
		dpow *= ntr
		if method == 1:
			for n in range(nf):
				_r[n, :, :] = np.linalg.pinv(_r[n, :, :], rcond=1e-6)

		errcode = clibsignal.generalizedBeamformer(relpow_map, abspow_map, 
													   steer, _r, ntr, prewhiten,
            										   grdpts_x, grdpts_y, nf, dpow, method)
		if errcode != 0:
			msg = 'generalizedBeamforming exited with error %d'%(errcode)
			gpar.log(__name__, msg, level='error', pri=True)
		ix, iy = np.unravel_index(relpow_map.argmax(), relpow_map.shape)
		relpow_tmp, abspow_tmp = relpow_map[ix, iy], abspow_map[ix, iy]
		slow_x = sll_x + ix * sl_s
		slow_y = sll_y + iy * sl_s
		slow = np.sqrt(slow_x**2 + slow_y**2)
		if slow < 1e-8:
			slow = 1e-8
		slow_h[ind] = slow
		azimuth = 180.0 * np.arctan2(slow_x, slow_y)/PI
		baz = azimuth % -360 +180
		if baz < 0:
			baz = baz +360.0
		bakaz[ind] = baz
		abspow[ind] = abspow_tmp
		relpow[ind] = relpow_tmp
		offset += nstep

	rel = np.empty((4,bnpts))
	rel[0,:] = np.log10(abspow)
	rel[1,:] = slow_h
	rel[2,:] = bakaz
	rel[3,:] = relpow

	return rel

def fkslowness(stream, ntr, time_shift_table,
			   sll_x=-15, sll_y=-15, sl_s=0.1,
			   grdpts_x=301, grdpts_y=301,
			   freqmin=1, freqmax=3.0,
			   starttime=1150, endtime=1200,
			   prewhiten=0, method=0, **kwargs):
	time_shift_table = time_shift_table.astype(np.float32)
	res = []
	spoint, _epoint = get_spoint(stream, starttime, endtime)
	fs = stream[0].stats.sampling_rate
	nsamp = int((endtime - starttime) * fs)
	nfft = next_pow_2(nsamp)
	deltaf = fs/float(nfft)
	nlow = int(freqmin/deltaf + 0.5)
	nhigh = int(freqmax/deltaf + 0.5)
	nlow = max(1, nlow)
	nhigh = min(nfft//2 - 1, nhigh)
	nf = nhigh - nlow + 1
	steer = np.empty((nf, grdpts_x, grdpts_y, ntr), dtype=np.complex128)
	clibsignal.calcSteer(ntr, grdpts_x, grdpts_y, nf, nlow, deltaf, time_shift_table, steer)
	_r = np.empty((nf, ntr, ntr), dtype=np.complex128)
	ft = np.empty((ntr, nf), dtype=np.complex128)
	tap = cosine_taper(nsamp, p=0.22)
	abspow_map = np.empty((grdpts_x, grdpts_y), dtype=np.float64)
	relpow_map = np.empty((grdpts_x, grdpts_y), dtype=np.float64)

	try:
		for i, tr in enumerate(stream):
			dat = tr.data[spoint[i]: spoint[i]+nsamp]
			dat = (dat - dat.mean()) * tap 
			ft[i, :] = np.fft.rfft(dat,nfft)[nlow:nlow+nf]
	except:
		msg = 'Problem in existed data and can not do the DFFT'
		gpar.log(__name__, msg, level='error', pri=True)

	ft = np.ascontiguousarray(ft, np.complex128)
	relpow_map.fill(0.)
	abspow_map.fill(0.)

	dpow = 0.0

	for i in range(ntr):
		for j in range(i, ntr):
			_r[:, i, j] = ft[i,:] * ft[j,:].conj()
			if method == 1:
				_r[:, i, j] /= np.abs(_r[:, i, j].sum())
			if i != j:
				_r[:, j, i] = _r[:, i, j].conjugate()
			else:
				dpow += np.abs(_r[:, i, j].sum())
	dpow *= ntr
	if method == 1:
		for n in range(nf):
			_r[n, :, :] = np.linalg.pinv(_r[n, :, :], rcond=1e-6)
	errcode = clibsignal.generalizedBeamformer(relpow_map, abspow_map, 
													   steer, _r, ntr, prewhiten,
            										   grdpts_x, grdpts_y, nf, dpow, method)

	if errcode != 0:
		msg = 'generalizedBeamforming exited with error %d'%(errcode)
		gpar.log(__name__, msg, level='error', pri=True)

	return relpow_map, abspow_map


def slideBeam(stream, ntr, delta, geometry,arrayName,grdpts_x=301,grdpts_y=301,
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
	use_geometry = pd.DataFrame()
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
		tmprow = geometry[geometry.STA == sta].iloc[0]
		use_geometry = use_geometry.append(tmprow, ignore_index=True)
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
	timeTable = getTimeTable(use_geometry, sll_x, sll_y, sl_s, grdpts_x, grdpts_y,unit)
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


#@profile
def slantBeam(stream, ntr, delta, geometry,arrayName,grdpts=401,
					filts={'filt_1':[1,2,4,True],'filt_2':[2,4,4,True],'filt_3':[1,3,4,True]},
					stack='linear',sl_s=0.1, vary='slowness',sll=-20.0,
					starttime=400.0,endtime=1400.0, unit='deg',
					**kwargs):
	st = stream.copy()
	st.detrend('demean')
	# st.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
	npts = st[0].stats.npts
	winlen = endtime - starttime
	winpts = int(winlen/delta) + 1
	#times = starttime + np.arange(winpts) * delta
	#k = sll + np.arange(grdpts)*sl_s
	envel = pd.DataFrame(columns=['FILT','POWER'])
	for name, filt in filts.items():
		tmp_st = st.copy()
		tmp_st.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
		hilbertTd = np.empty((ntr,npts),dtype=complex)
		use_geometry = pd.DataFrame()
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
			tmprow = geometry[geometry.STA == sta].iloc[0]
			use_geometry = use_geometry.append(tmprow, ignore_index=True)
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

		timeTable = getSlantTime(use_geometry,grdpts,sl_s,sll,vary,unit,**kwargs)
		abspow = np.empty((grdpts,winpts))
		errorcode = clibarray.slant(ntr,grdpts,
					iflag,npts,delta,
					starttime, winpts, order,
					timeTable,hilbertTd,abspow)
		if errorcode != 0:
			msg = 'slantbeam stack for %dth segment in filter %s-%s exited with error %d\n' % (ind, name, filt,errorcode)
			gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
		#envel[name] = abspow
		envel = envel.append({'FILT': name, 'POWER': abspow}, ignore_index=True)
		# if sflag == 'beam':
		# 	envel[name] = abspow
		# else:
		# 	env = np.abs(scipy.signal.hilbert(abspow))
		# 	if sflag == 'log':
		# 		envel[name] = np.log(env)
		# 	elif sflag == 'log10':
		# 		envel[name] = np.log10(env)
		# 	elif sflag == 'sqrt':
		# 		envel[name] = np.sqrt(env)
		# 	else:
		# 		msg = 'Not available option, please choose from beam, log, log10 or sqrt\n'
		# 		gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
	return envel

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

def codaStr(st1, st2, delta, win_st1, win_st2, winlen,dv,nbtrial,method):

	dvs = []
	for tr1, tr2, win_tr1, win_tr2 in zip_longest(st1, st2, win_st1, win_st2):

		time1 = np.arange(tr1.stats.npts)

		t0_x = win_tr1.stats.starttime - tr1.stats.starttime
		t0_y = win_tr2.stats.starttime - tr2.stats.starttime

		s = stretching(tr1.data, tr2.data, 
					   time1, time1, delta, 
					   t0_x, t0_y, winlen,
					   dv_range=dv,nbtrial=nbtrial)

		dvs.append(s)

	dv_mean = np.mean(dvs)

	return dv_mean

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

def _getLag(Mptd1, Mptd2, delta, domain='freq', fittype='cos', getcc=False):

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

	if getcc:
		return [xmax, ymax, dt, cc]
	else:
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

def getDT(st, DF, stalist, delta):

	tsDF = DF.copy()
	if len(st) < len(tsDF):
		for s in stalist:
			tmp_st = st.select(station=s)
			if len(tmp_st) == 0:
				msg = 'Station %s is missing, dropping station'%(s)
				gpar.log(__name__, msg, level='info', pri=True)
				tsDF = tsDF[~(tsDF.STA == s)]
	DT = tsDF.TimeShift - (tsDF.TimeShift/delta).astype(int)*delta
	lag = (tsDF.TimeShift/delta).astype(int) + 1
	tsDF['DT'] = DT
	tsDF['LAG'] = lag

	return tsDF

def cutWaveForm(st1, st2,delta,
				arr1, arr2, cstime, 
				cetime,filt,threshold,
				method, domain, fittype,
				cut,tshift):

	tmp_st1 = st1.copy()
	tmp_st2 = st2.copy()
	stime1 = arr1 - cstime
	stime2 = arr2 - cstime
	etime1 = arr1 + cetime
	etime2 = arr2 + cetime

	tmp_st1.trim(starttime=stime1, endtime=etime1)
	if len(tmp_st1) == 0:
		msg = ('Earthquake does not have waveform in %s period'%(arr1))
		gpar.log(__name__, msg, level='warning', pri=True)
		return None

	tmp_st2.trim(starttime=stime2, endtime=etime2)
	if len(tmp_st2) == 0:
		msg = ('Earthquake does not have waveform in %s period'%(arr2))
		gpar.log(__name__, msg, level='warning', pri=True)
		return None

	sta_id1 = []
	sta_id2 = []

	for tr1, tr2 in zip_longest(tmp_st1, tmp_st2):
		if tr1 != None:
			sta_id1.append(tr1.get_id())
		if tr2 != None:
			sta_id2.append(tr2.get_id())

	sta_id = list(set(sta_id1) & set(sta_id2))
	sta_id.sort()
	nst1 = obspy.Stream()
	nst2 = obspy.Stream()

	for s in sta_id:
		_tr1 = st1.select(id=s).copy()[0]
		_tr2 = st2.select(id=s).copy()[0]
		nst1.append(_tr1)
		nst2.append(_tr2)

	nst1.filter('bandpass', freqmin=filt[0], 
				freqmax=filt[1], corners=filt[2], 
				zerophase=filt[3])
	nst2.filter('bandpass', freqmin=filt[0], 
				freqmax=filt[1], corners=filt[2], 
				zerophase=filt[3])

	tmp_st1 = nst1.copy().trim(starttime=stime1, endtime=etime1)
	tmp_st2 = nst2.copy().trim(starttime=stime2, endtime=etime2)
	if tmp_st1 == None or tmp_st2 == None:
		msg = ('Data is empty for doublet %s-%s'%(arr1,arr2))
		gpar.log(__name__, msg, level='warning',pri=True)
		return None

	npts = int((cetime+cstime)/delta) + 1

	tmp_st1, tmp_st2 = _resample(tmp_st1, tmp_st2, delta, method, npts)

	Mptd1 = np.zeros([len(nst1), npts])
	Mptd2 = np.zeros([len(nst2), npts])
	inds = range(len(sta_id))

	for ind, tr1, tr2, in zip_longest(inds, tmp_st1, tmp_st2):
		if tr1.stats.station != tr2.stats.station:
			msg = ('Orders of the traces are not right for Doublet %s-%s'%(arr1, arr2))
			gpar.log(__name__, msg, level='warning',pri=True)
			return None
		data1 = tr1.data
		data2 = tr2.data
		Mptd1[ind, :] = data1
		Mptd2[ind, :] = data2

	taup, cc,_,_ = _getLag(Mptd1, Mptd2, delta, domain, fittype)
	col = ['STA', 'TS', 'CC']
	df = pd.DataFrame(columns=col)
	df['STA'] = sta_id
	df['TS'] = taup
	df['CC'] = cc
	df = df[df.CC >= threshold]
	df = df[np.abs(df.TS)<=tshift]
	if len(df) < cut:
		msg = ('Correlation for doublet %s-%s is too bad, maybe due to SNR, dropping'%(arr1,arr2))
		gpar.log(__name__, msg, level='info', pri=True)
		return None

	df.sort_values("STA", inplace=True)
	df.reset_index(inplace=True)

	refTime = np.min(-cstime - df.TS)
	use_st1 = obspy.Stream()
	use_st2 = obspy.Stream()

	for ind, row in df.iterrows():
		_id = row.STA 
		_tr1 = nst1.select(id=_id)[0].copy()
		_tr2 = nst2.select(id=_id)[0].copy()
		_tr2.trim(starttime=arr2-cstime, endtime=arr2+cetime)
		use_st2.append(_tr2)
		thiftBT = -cstime - row.TS
		stime = arr1 + thiftBT
		_tr1.trim(starttime=stime, endtime=stime+cstime+cetime)
		use_st1.append(_tr1)

	use_st1, use_st2 = _resample(use_st1, use_st2, delta, method, npts)
	use_st1.sort(keys=['station'])
	use_st2.sort(keys=['station'])
	return [use_st1, use_st2, refTime, df]

def _resample(st1, st2,resample, method, npts):
	rrate = 1.0/resample
	if method == 'resample':
		st1.resample(sampling_rate=rrate)
		st2.resample(sampling_rate=rrate)
	elif method == 'interpolate':
		st1.interpolate(sampling_rate=rrate)
		st2.interpolate(sampling_rate=rrate)
	else:
		msg = ('Not a valid option for waveform resampling, choose from resample and interpolate')
		gpar.log(__name__,msg,level='error',e='ValueError',pri=True)
	for _tr1, _tr2 in zip_longest(st1, st2):
		data1 = _tr1.data
		data2 = _tr2.data
		if len(data1) > npts:
			_tr1.data = data1[0:npts]
		elif len(data1) < npts:
			data1 = np.pad(data1, (0,npts-len(data1)), 'edge')
			_tr1.data = data1
		if len(data2) > npts:
			_tr2.data = data2[0:npts]
		elif len(data2) < npts:
			data2 = np.pad(data2, (0,npts-len(data2)), 'edge')
			_tr2.data = data2

	return st1, st2




def stretching(x,y,t_x,t_y,delta, t0_x,t0_y, win, dv_range=0.02,nbtrial=401):

	# st=np.arange(0.5,4,0.1)
	st = 1 + np.linspace(-np.abs(dv_range),np.abs(dv_range),nbtrial)
	n=len(y)
	sind = int((t0_x - t_x[0])/delta) + 1
	npts = int(win/delta) + 1
	xi = x[sind: sind+npts]
	xi = xi[np.newaxis,:].repeat(len(st), axis=0)

	ks = np.arange(npts)

	ti = t0_y + ks * delta

	ts = np.outer(st, ti)

	dt = (ts - t_y[0]) / delta
	i1 = dt.astype(int)

	i2 = i1 + 1
	i1 = np.where(i1>=n,n-1,i1)
	i2 = np.where(i2>=n,n-1,i2)

	i1 = np.where(i1<0, 0, i1)
	i2 = np.where(i2<0, 0, i2)

	y1 = y[i1]
	y2 = y[i2]

	ytmp = (y2 - y1) * (dt - i1) + y1

	_, cc, _, _ = _getLag(xi, ytmp, delta, 'freq', 'cos')

	max_i = np.argmax(cc)

	s = st[max_i]
	dv = s - 1

	return dv

def stretching_interp(ref, cur, delta, t0_x, t0_y, win,dv_range,nbtrial):

	st = 1 + np.linspace(-np.abs(dv_range),np.abs(dv_range),nbtrial)
	tvec = np.arange(t0_y, t0_y+win+delta, delta)
	cof = np.zeros(st.shape,dtype=np.float32)

	for ii in range(len(st)):
		nt = tvec * st[ii]
		s = np.interp(x=tvec, xp=nt, fp=cur)
		waveform_ref = ref
		waveform_cur = s
		cof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0,1]

	cdp = np.corrcoef(cur, ref)[0,1]

	imax = np.nanargmax(cof)

	if imax >= len(st) - 2:
		imax = imax - 2

	if imax <= 2:
		imax = imax + 2

	dtfiner = np.linspace(st[imax-2], st[imax+2], 100)
	ncof = np.zeros(dtfiner.shape, dtype=np.float32)

	for ii in range(len(dtfiner)):
		nt = tvec * dtfiner[ii]
		s = np.interp(x=tvec, xp=nt, fp=cur)
		waveform_ref = ref
		waveform_cur = s
		ncof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0,1]

	cc = np.max(ncof)
	dv = dtfiner[np.argmax(ncof)] - 1

	return dv, cc, cdp
