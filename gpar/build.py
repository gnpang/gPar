from __future__ import print_function, absolute_import
from __future__ import with_statement, nested_scopes, division, generators

import os
import time
import numpy as np 
import pandas as pd 
from obspy.core import UTCDateTime
import obspy

import gpar
from gpar.util import util
import scipy

def createEqList(arrayList='array.list',
				ndkfile='gcmt_1996_2017.ndk',
				formats='ndk',
				mindis=50.0, maxdis=75, mag=None,
				model='ak135',
				phase=['PKiKP'],
				savefile='array.list.e'):
	
	ardf = util.readList(arrayList, list_type='array', sep='\s+')
	msg = ('reading ndk file %s'%ndkfile)
	gpar.log(__name__, msg, level='info', pri=True)
	# ndk = obspy.read_events(ndkfile)
	if formats == 'ndk':
		ndk = util.readNDK(ndkfile)
	elif formats == 'csv':
		ndk = pd.read_csv(ndkfile, delimiter='\s+')
	else:
		msg = ('Not a right format for %s\nchoose from ndk or csv'%ndkfile)
		gpar.log(__name__, msg, level='error', pri=True)
	eqnum = ['']*len(ardf)
	for ind, row in ardf.iterrows():
		name = row.NAME
		if not os.path.isdir(name):
			os.mkdir(name)
		msg = ('Generating Eq list for array %s'%name)
		gpar.log(__name__, msg, level='info', pri=True)
		evedf = gpar.getdata.makeEventList(ndk=ndk, array=name,
						Olat=row.LAT, Olon=row.LON, 
						mindis=mindis, maxdis=maxdis, minmag=mag,
						model=model, phase=phase,
						)
		# evedf = evedf[evedf.time>= UTCDateTime(starttime)]
		n = len(evedf)
		eqnum[ind] = n

	ardf['EQNUM'] = eqnum

	ardf.to_csv(savefile, sep=str('\t'), index=False)

	return ardf


def createArray(arrayList='array.list',
				savecarray=True,
				fileName='array.pkl',
				model='ak135',
				phase=['PKiKP'],
				coordsys='lonlat',
				calTime=True,
				save=True,
				saveName=False,
				**kwargs):

	"""
	Function to create the Array instance 
	"""

	arraydf = util.readList(arrayList, list_type='array', sep='\s+')

	arDF = pd.DataFrame(columns=['NAME','ARRAY'])
	for ind, row in arraydf.iterrows():
		# eqlist = os.path.join(row.NAME,'eq.'+row.NAME+'.list')
		# if not os.path.exists(eqlist):
		# 	msg = ('Earthquake list for array %s is not exists, building from ndk file' % row.NAME)
		# 	gpar.log(__name__,msg,level='warning',pri=True)
		# 	if 'ndkfile' not in kwargs.keys():
		# 		msg = ('input the ndkfile and related parameters (see getdata.makeEventList) for building earthquake list')
		# 		gpar.log(__name__, msg, level='error', pri=True)
		# 	if not os.path.isfile(ndkfile):
		# 		msg = ('ndk file does not exist')
		# 		gpar.log(__name__,msg, level='error', pri=True)

		# 	ndkfile = kwargs['ndkfile']
		# 	eqdf = gpar.getdata.makeEventList(ndkfile=ndkfile,array=row.NAME,Olat=row.LAT,Olon=row.LON,
		# 									  model=model,phase=phase)
		# else:
		# 	eqdf = util.readList(eqlist,list_type='event',sep='\s+')
		refpoint = [row.LAT, row.LON]
		fet = gpar.getdata.quickFetch(row.NAME)
		# streams = [''] * len(eqdf)
		# for num, eve in eqdf.iterrows():
		# 	st = fet.getStream(eve.DIR)
		# 	streams[num] = st
		# eqdf['Stream'] = streams
		eqdf, stadf = fet.getEqData(row, phase=phase)
		if eqdf is None:
			msg = 'Earthquake list for array %s is not existed, skipping' %(row.NAME)
			gpar.log(__name__, msg, level='warning', pri=True)
			continue
		array = gpar.arrayProcess.Array(row.NAME,refpoint,eqdf, stadf, coordsys,phase)

		if calTime:
			msg = ('Calculate time shift table for sliding window slowness beaforming for array %s'%row.NAME)
			gpar.log(__name__,msg,level='info',pri=True)

			req_pa = set(['sll_x','sll_y','sl_s','grdpts_x','grdpts_y','unit'])

			if not req_pa.issubset(kwargs.keys()):
				msg = ('Required parameters %s for time shift table are missing\nusing default parameters sll_x=-15, sll_y=-15, sl_s=0.1,grdpts_x=301, grdpts_y=301, unit="deg"'%req_pa)
				gpar.log(__name__,msg,level='info',pri=True)
				array.getTimeTable()
			else:
				sll_x=kwargs['sll_x']
				sll_y=kwargs['sll_y']
				sl_s=kwargs['sl_s']
				grdpts_x = kwargs['grdpts_x']
				grdpts_y = kwargs['grdpts_y']
				unit = kwargs['unit']

				array.getTimeTable(sll_x=sll_x,sll_y=sll_y,sl_s=sl_s,grdpts_x=grdpts_x,grdpts_y=grdpts_y,unit=unit)
		
		if save:
			array.write()

		newRow = {'NAME':row.NAME,'ARRAY':array}

		arDF = arDF.append(newRow,ignore_index=True)

	if saveName:
		arDF.to_pickle(saveName)

	return arDF


def upDateArray(array, beamtype='beam', filt=[1,2,4,True],
				starttime=0,endtime=1800,unit='deg', stack='linear',
				save=True,write=True,**kwargs):
	"""
	Function to undate array processing to the array instance
	array: str or instance of gpar.arrayProcess.Array or DataFrame.
		If it is DataFrame, array name is needed as input of arrayName
	beamtype: str. Array processing that need to be done, avaliable now:
		'beam': normal beamforming
		'slide': sliding window beamforming.
		'vespectrum': vespectrum for seleceted array. If choose, vary for slowness or back azimuth needs to be choose either. Default vary ="slowness" 
	"""

	if isinstance(array, gpar.arrayProcess.Array):
		ar = array
	elif isinstance(array, str):
		ar = util.loadArray(array)
	elif isinstance(array, pd.DataFrame):
		tmp = array[array.NAME == kwargs['arrayName']].iloc[0]
		ar = tmp.ARRAY
	else:
		msg = ('Not a valid array type, must be a path, Array instance or a DataFrame that has Array instance\nif is DataFrame, arrayName is needed')
		gpar.log(__name__,msg,level='error',pri=True)

	if beamtype == 'beam':
		winlen = endtime - starttime
		ar.beamforming(filt=filt,starttime=starttime,winlen=winlen,
						stack=stack, unit=unit,write=write)
		if save:
			fileName = ar.name + '.beam.pkl'
			ar.write(fileName)
	elif beamtype == 'slide':
		req_pa = set(['sll_x','sll_y','sl_s',
					  'grdpts_x','grdpts_y',
					  'winlen','overlap','sflag',
					  'refine'])
		if not req_pa.issubset(kwargs.keys()):
			msg = ('Required parameters %s for sliding beamforming are missing'%req_pa)
			gpar.log(__name__,msg,level='error',pri=True)
		sll_x = kwargs['sll_x']
		sll_y = kwargs['sll_y']
		sl_s = kwargs['sl_s']
		grdpts_x = kwargs['grdpts_x']
		grdpts_y = kwargs['grdpts_y']
		if not hasattr(ar, 'timeTable'):
			msg = ('Shift time table is not calculated before for array %s, now calculating the shift time table first'%(ar.name))
			gpar.log(__name__,msg,level='info',pri=True)
			ar.getTimeTable(sll_x=sll_x,sll_y=sll_y,sl_s=sl_s,grdpts_x=grdpts_x,grdpts_y=grdpts_y,unit=unit)
		ar.slideBeam(filt=filt, stack=stack,
					 starttime=starttime, endtime=endtime,
					 write=write,**kwargs)
		if save:
			fileName = ar.name + '.slide.pkl'
			ar.write(fileName)
	elif beamtype == 'vespectrum':

		req_pa = set(['sl_s','grdpts','sflag','vary','sll'])
		if not req_pa.issubset(kwargs.keys()):
			msg = ('Required parameters %s for vespectrum are missing'%req_pa)
			gpar.log(__name__,msg,level='error',pri=True)
		ar.vespectrum(filt=filt,stack=stack,
					  starttime=starttime,endtime=endtime,
					  unit=unit,**kwargs)
		if save:
			fileName = ar.name + '.veps.pkl'
			ar.write(fileName)
	else:
		msg = ('Not a valid array processing, choose from beam, slide or vespectrum')
		gpar.log(__name__,msg,level='error',pri=True)

	return ar

