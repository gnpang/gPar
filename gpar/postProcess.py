"""
GUI code modified based on https://github.com/miili/StreamPick

Need updates:
Need to reformat the strip dataframe, add in stripping information when previous stripping is loaded
"""

import os
import pickle
import pandas as pd
import numpy as np 

# GUI import
import PyQt4
from PyQt4 import QtGui
import sys
import signal
import scipy
import gpar
from itertools import cycle

#figure plot import
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.transforms import offset_copy

from mpl_toolkits.basemap import Basemap 
# from mpl_toolkits.axesgrid1 import make_axes_locatable

#obspy import
from obspy.taup import TauPyModel 
import obspy

signal.signal(signal.SIGINT, signal.SIG_DFL)


#class for event first evaluation
class glanceEQ(QtGui.QMainWindow):

	def __init__(self, array=None, parent=None, ap=None):

		if ap is None:
			self.qApp = QtGui.QApplication(sys.argv)
		else:
			self.qApp = ap
		self.KeepGoing = False

		if isinstance(array, str):
			ar = gpar.util.loadArray(array)
		elif isinstance(array, gpar.arrayPropocess.Array):
			ar = array
		else:
			msg = 'Define Array instance = gpar.arrayPropocess.Array() or a path to a pickle file'
			raise ValueError(msg)
		self.array = ar
		self.eve_type = ['A','B','C','D']
		self._shortcuts = {'eve_next': 'n',
						   'eve_prev': 'p',
						   'trim_apply': 'w',
						   'gain_up': 'u',
						   'gain_down': 'd',
						   'strip': 's',
						   'A':'a',
						   'B':'b',
						   'C':'c',
						   'D':'d'}

		self._plt_drag = None

		# init events in the array
		self._events = ar.events #defines list self._events
		self.savefile = None
		self._initEqList()
		self._stripDF = pd.DataFrame()
		self._badDF = pd.DataFrame()
		self._btype = 'beam'
		self._method = 'all'
		self.trinWin = [{'name':'N200-C200','noise':200.0,'coda':200.0,
						 'stime':400.0,'etime':1800,'model':'ak135'}]
		self._current_win = None
		self._current_strip = False
		self._eventCycle = cycle(self._eqlist)
		self._eventInfo(self._eventCycle.next())
		QtGui.QMainWindow.__init__(self)
		self.setupUI()


	def setupUI(self):
		self.main_widget = QtGui.QWidget(self)
		self._initMenu()
		self._createStatusBar()
		self._initPlots()

		l = QtGui.QVBoxLayout(self.main_widget)
		l.addLayout(self.btnbar)
		l.addLayout(self.btnbar2)
		l.addWidget(self.canvas)

		self.setCentralWidget(self.main_widget)
		self.setGeometry(300, 300, 1200, 800)
		self.setWindowTitle('Array Analysis')
		self.show()
	def _killLayout():
		pass

	def _initEqList(self):
		self._eqlist = []

		for _eve in self._events:
			self._eqlist.append(_eve.ID)
		self._eqlist.sort()

	def _initPlots(self):
		self.fig = Figure(facecolor='.86',dpi=100, frameon=True)
		self.canvas = FigureCanvas(self.fig)
		self.canvas.setFocusPolicy(PyQt4.QtCore.Qt.StrongFocus)
		self._drawFig()

		# connect the events
		self.fig.canvas.mpl_connect('scroll_event', self._pltOnScroll)
		self.fig.canvas.mpl_connect('motion_notify_event', self._pltOnDrag)
		self.fig.canvas.mpl_connect('button_release_event', self._pltOnButtonRelease)

	def _initMenu(self):
		# Next and Prev Earthquake
		nxt = QtGui.QPushButton('Next >>',
			shortcut=self._shortcuts['eve_next'], parent=self.main_widget)
		nxt.clicked.connect(self._pltNextEvent)
		nxt.setToolTip('shortcut <b>n</d>')
		nxt.setMaximumWidth(150)
		prv = QtGui.QPushButton('Prev >>',
			shortcut=self._shortcuts['eve_prev'], parent=self.main_widget)
		prv.clicked.connect(self._pltPrevEvent)
		prv.setToolTip('shortcut <b>p</d>')
		prv.setMaximumWidth(150)

		# Earthquake drop-down
		self.evecb = QtGui.QComboBox(self)
		for eve in self._eqlist:
			self.evecb.addItem(eve)
		self.evecb.activated.connect(self._pltEvent)
		self.evecb.setMaximumWidth(1000)
		self.evecb.setMinimumWidth(80)

		# coda strip button
		self.codabtn = QtGui.QPushButton('Strip',
			shortcut=self._shortcuts['strip'],parent=self.main_widget)
		self.codabtn.setToolTip('shortcut <b>s</b>')
		self.codabtn.clicked.connect(self._appStrip)

		self.codacb = QtGui.QComboBox(self)
		for med in ['all', 'coda','twoline']:
			self.codacb.addItem(med)
		self.codacb.activated.connect(self._selectMethod)
		self.codacb.setMaximumWidth(100)
		self.codacb.setMinimumWidth(80)
		self.wincb = QtGui.QComboBox(self)
		self.wincb.activated.connect(self._changeStrip)
		self._updateWinow()
		
		# edit/delete coda selected window
		winEdit = QtGui.QPushButton('Coda Window')
		winEdit.resize(winEdit.sizeHint())
		winEdit.clicked.connect(self._editTimeWindow)

		winDelt = QtGui.QPushButton('Delete')
		winDelt.resize(winDelt.sizeHint())
		winDelt.clicked.connect(self._deleteWin)

		# Coda level
		_radbtn = []
		for _o in self.eve_type:
			_radbtn.append(QtGui.QRadioButton(_o.upper(), shortcut=self._shortcuts[_o.upper()]))
			_radbtn[-1].setToolTip('Level: '+_o)
		self.levelGrp = QtGui.QButtonGroup()
		self.levelGrp.setExclusive(True)
		levelbtn = QtGui.QHBoxLayout()

		for _i, _btn in enumerate(_radbtn):
			self.levelGrp.addButton(_btn, _i)
			levelbtn.addWidget(_btn)

		# plot slide beam figure button
		self.sbcb = QtGui.QComboBox(self)
		for btype in ['beam', 'slide', 'vespetrum','strip']:
			self.sbcb.addItem(btype)
		self.sbcb.activated.connect(self._updatePlot)
		self.codacb.setMaximumWidth(100)
		self.codacb.setMinimumWidth(80)
		# Arrange buttons
		vline = QtGui.QFrame()
		vline.setFrameStyle(QtGui.QFrame.VLine | QtGui.QFrame.Raised)
		self.btnbar = QtGui.QHBoxLayout()
		self.btnbar.addWidget(prv)
		self.btnbar.addWidget(nxt)
		self.btnbar.addWidget(QtGui.QLabel('Event'))
		self.btnbar.addWidget(self.evecb)
		##
		self.btnbar.addWidget(vline)
		self.btnbar.addWidget(self.codabtn)
		self.btnbar.addWidget(self.codacb)
		self.btnbar.addWidget(self.wincb)
		self.btnbar.addWidget(winEdit)
		self.btnbar.addWidget(winDelt)
		self.btnbar.addStretch(1)

		self.btnbar2 = QtGui.QHBoxLayout()
		self.btnbar2.addWidget(QtGui.QLabel('Level: '))
		self.btnbar2.addLayout(levelbtn)
		

		self.btnbar2.addWidget(vline)
		self.btnbar2.addWidget(QtGui.QLabel('TYPE'))
		self.btnbar2.addWidget(self.sbcb)
		self.btnbar2.addStretch(3)

		#Menubar
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(QtGui.QIcon.fromTheme('document-save'),
						   'Save', self._saveFile)
		fileMenu.addAction(QtGui.QIcon().fromTheme('document-save'),
						   'Save as', self._saveFileFormat)
		fileMenu.addAction(QtGui.QIcon().fromTheme('document-open'),
						   'Load Pickle File', self._openFile)
		fileMenu.addSeparator()
		fileMenu.addAction(QtGui.QIcon().fromTheme('document-save'),
						   'Save Plot', self._savePlot)
		fileMenu.addSeparator()
		quit = QtGui.QAction("Quit", self)
		fileMenu.addAction(quit)
		fileMenu.triggered[QtGui.QAction].connect(self.closeArray)

	def _hardExist(self):
		self.deleteLater()

	def _createStatusBar(self):
		"""
		Creates the status bar
		"""
		sb =QtGui.QStatusBar()
		sb.setFixedHeight(18)
		self.setStatusBar(sb)
		self.statusBar().showMessage('Ready')

	def _selectMethod(self, index):
		self._method = self.codacb.currentText()
		self.sbcb.setCurrentIndex(3)
		self._updatePlot()

	def _changeStrip(self,index):

		if index == len(self.trinWin):
			return self._newTrim()
		else:
			return self._appStrip()

	def _newTrim(self):
		"""
		Creat new strip window
		"""

		newWin = self.defWindow(self)

		if newWin.exec_():
			self.trinWin.append(newWin.getValues())
			self._updateWinow()
			self.wincb.setCurrentIndex(len(self.trinWin)-1)
			self._appStrip()

	def _editTimeWindow(self):
		"""
		Edit existing coda selection window
		"""

		_i = self.wincb.currentIndex()
		this_window = self.trinWin[_i]
		editWindow = self.defWindow(self, this_window)
		if editWindow.exec_():
			self.trinWin[_i] = editWindow.getValues()
			self.updateWindow()
			self.wincb.setCurrentIndex(_i)
			self._appStrip()

	def _deleteWin(self):
		"""
		Delete window
		"""

		pass
		_i = self.wincb.currentIndex()

	def _updateWinow(self):
		self.wincb.clear()
		self.wincb.setCurrentIndex(-1)
		for _i, _f in enumerate(self.trinWin):
			self.wincb.addItem('Noise %.2f sec - Coda %.2f sec' %(_f['noise'], _f['coda']))
		self.wincb.addItem('Create new Window')


	def _appStrip(self, button=True, draw=True):
		"""
		Apply coda strip
		"""
		_method = self.codacb.currentText()
		_j = self.wincb.currentIndex()
		self._eventInfo(self._current_id)
		self._current_strip = True
		codaStrip(self._current_event,beamtype=self._btype,method=_method,
				 	  siglen=self.trinWin[_j]['coda'], noise=self.trinWin[_j]['noise'],phase='PKiKP', 
			  		  model=self.trinWin[_j]['model'], stime=self.trinWin[_j]['stime'], etime=self.trinWin[_j]['etime'],)
		self._btype = 'strip'
		self.sbcb.setCurrentIndex(3)
		self._setCodaStrip()
		self._updatePlot()
				

	def _pltEvent(self):
		"""
		Plot event from DropDown Menu
		"""
		_i = self.evecb.currentIndex()
		while self._eventCycle.next() != self._eqlist[_i]:
			pass
		self._eventInfo(self._eqlist[_i])
		self._drawFig()

	def _pltPrevEvent(self):
		"""
		Plot previous events
		"""
		_j = self.evecb.currentIndex()
		for _i in range(len(self._eqlist) - 1):
			prevEvent = self._eventCycle.next()
		self._eventInfo(prevEvent)
		self.evecb.setCurrentIndex(_j-1)
		if self._btype == 'strip':
			self._btype = 'beam'
			self.sbcb.setCurrentIndex(0)
		self._drawFig()

	def _pltNextEvent(self):
		self._eventInfo(self._eventCycle.next())
		_i = self.evecb.currentIndex()
		self.evecb.setCurrentIndex(_i+1)
		if self._btype == 'strip':
			self._btype = 'beam'
			self.sbcb.setCurrentIndex(0)
		self._drawFig()

	def _eventInfo(self, eqid):
		"""
		Copies the array process result from the current Earthquake object
		"""

		for eve in self._events:
			if eve.ID == eqid:
				event = eve
		self._current_event = event
		self._current_id = eqid
		if not hasattr(event, 'beam'):
			return
		self._current_beam = event.beam
		self._current_ID = event.ID
		self._current_dis = event.Del
		self._current_p = event.rayParameter
		self._current_bb = event.pattern
		self._current_bakAz = event.bakAzimuth
		if hasattr(event, 'slideSt'):
			self._current_slide = event.slideSt
		if hasattr(event, 'energy'):
			self._current_energy = event.energy
			self._current_time = event.slantTime
			self._current_K = event.slantK
			self._current_type = event.slantType
	
	def _setCodaStrip(self):

		if not self._current_strip:
			return

		event = self._current_event
		_i = self.wincb.currentIndex()
		win = self.trinWin[_i]
		if len(self._stripDF) != 0:
			existDF = self._stripDF[(self._stripDF.ID == self._current_event.ID) & (self._stripDF.winName == win['name'])]
			if len(existDF) !=0:
				choice = QtGui.QMessageBox.question(self, 'Replace stripping',
							"Do you want to replace existed stripping?",
							QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
				if choice == QtGui.QMessageBox.Yes:
					index = existDF.index
					self._stripDF.drop(index,axis=0,inplace=True)
					self._stripDF.reset_index()
				else:
					return
		level = self.eve_type[self.levelGrp.checkedId()]
		ID = event.ID
		lat = event.lat
		lon = event.lon
		dep = event.dep
		mw = event.Mw
		dis = event.Del
		bb = event.pattern
		bakAzi = event.bakAzimuth

		if level =='D':
			newRow = {'ID': ID, 'lat':lat,
					  'lon':lon,'dep':dep,
					  'Mw':mw,'Del':dis,
					  'BB':bb,'bakAzi':bakAzi,'Level':'D'}
			self._badDF = self._badDF.append(newRow, ignore_index=True)
		else:
			if self._method == 'all':
				newRow = {'ID': ID, 'lat':lat,
						  'lon':lon,'dep':dep,
						  'Mw':mw,'Del':dis,
						  'BB':bb,'bakAzi':bakAzi,
						  'winName':win['name'], 'win':win,
						  'Level':level,
						  'codaResTr':event.codaResTr,
						  'codaTr':event.codaTr,
						  'crms':event.codaMod,
						  'twoResTr':event.twoResTr,
						  'twoTr':event.twoTr,
						  'trms':event.twoMod}
				self._stripDF = self._stripDF.append(newRow, ignore_index=True)
			elif self._method == 'coda':
				newRow = {'ID': ID, 'lat':lat,
						  'lon':lon,'dep':dep,
						  'Mw':mw,'Del':dis,
						  'winName':win['name'], 'win':win,
						  'BB':bb,'bakAzi':bakAzi,
						  'Level':level,
						  'codaResTr':event.codaResTr,
						  'codaTr':event.codaTr,
						  'crms':event.codaMod,
						  }
				self._stripDF = self._stripDF.append(newRow, ignore_index=True)
			elif self._method == 'twoline':
				newRow = {'ID': ID, 'lat':lat,
						  'lon':lon,'dep':dep,
						  'Mw':mw,'Del':dis,
						  'BB':bb,'bakAzi':bakAzi,
						  'Level':level,
						  'winName':win['name'], 'win':win,
						  'twoResTr':event.twoResTr,
						  'twoTr':event.twoTr,
						  'trms':event.twoMod}
				self._stripDF = self._stripDF.append(newRow, ignore_index=True)


	def _drawFig(self):
		self.fig.clear()
		if self._btype == 'beam' or self._btype == 'slide':
			if self._btype == 'beam':
				self._current_st = obspy.core.stream.Stream(traces=[self._current_beam])
			elif self._btype == 'slide':
				self._current_st = self._current_slide
			num_plots = len(self._current_st)
			for _i, tr in enumerate(self._current_st):
				ax = self.fig.add_subplot(num_plots, 1, _i+1)
				if hasattr(tr.stats, 'channel'):
					label = tr.stats.channel
				else:
					label=None
				time = np.arange(tr.stats.npts) * tr.stats.delta + tr.stats.sac.b
				ax.plot(time, tr.data, 'k', label=label)
				if not hasattr(self._current_event, 'arrival'):
					self._current_event.getArrival()
				arrival = self._current_event.arrival - self._current_event.time
				ax.vlines(arrival, ax.get_ylim()[0],ax.get_ylim()[1],'r')
				ax.legend()
				if _i == 0:
					ax.set_xlabel('Seconds')
				self.fig.suptitle('%s - %s'%(self._current_event.ID, self._btype))
		elif self._btype == 'vespetrum':
			ax = self.fig.add_subplot(1, 1, 1)
			extent=[np.min(self._current_time),np.max(self._current_time),np.min(self._current_K),np.max(self._current_K)]
			ax.imshow(self._current_energy, extent=extent, aspect='auto')
			if self._current_type == 'slowness':
				a = u"\u00b0"
				title = 'Slant Stack at a Backazimuth of %.1f %sN'%(self._current_event.bakAzimuth,a)
			elif self._current_type == 'theta':
				title = 'Slant Stack at a slowness of %.2f s/deg'%(self._current_event.rayParameter)
			self.fig.suptitle(title)
		elif self._btype == 'strip':
			existDF = self._stripDF[(self._stripDF.ID == self._current_event.ID)]
			if len(existDF) == 0:
				choice = QtGui.QMessageBox.question(self, 'Stripping?',
						"Haven't stripping yet, want to do it?",
						QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
				if choice == QtGui.QMessageBox.Yes:
					self._appStrip()
				else:
					self._btype = 'beam'
					self.sbcb.setCurrentIndex(3)
					self._updatePlot()
			else:
				trinwin = existDF.win.iloc[0]
				stime = trinwin['stime']
				etime = trinwin['etime']
				delta = self._current_beam.stats.delta
				npts = int((etime - stime)/delta) + 1
				time = np.linspace(stime, etime, npts)
				data = np.abs(scipy.signal.hilbert(self._current_beam.data))
				sind = int(stime / delta)
				eind = int(etime / delta)
				data = data[sind:sind+npts]
				if self._method == 'all':
					ax1 = self.fig.add_subplot(2,2,1)
					ax1.plot(time, np.log10(data), 'k')
					codaTr = existDF.codaTr.iloc[0]
					data_coda = codaTr.data
					ax1.plot(time, np.log10(data_coda), 'r')
					ax1.set_xlim([stime, etime])
					ax1.set_ylim([-1, 5])
					ax1.set_xlabel('Seconds')
					ax1.set_ylabel('log10(Amp)')

					ax2 = self.fig.add_subplot(2,2,2)
					ax2.plot(time, np.log10(data), 'k')
					twoTr = existDF.twoTr.iloc[0]
					data_two = twoTr.data
					data_time = np.arange(twoTr.stats.npts) * delta + (twoTr.stats.starttime - self._current_beam.stats.starttime)
					ax2.plot(data_time,data_two,'r')
					ax2.set_xlabel('Seconds')
					ax2.set_xlim([stime, etime])
					ax2.set_ylim([-1, 5])
					ax2.set_ylabel('log10(Amp)')

					ax3 = self.fig.add_subplot(2,2,3)
					cRes = existDF.codaResTr.iloc[0]
					timeR = np.arange(cRes.stats.npts)*cRes.stats.delta - trinwin['noise']
					ax3.plot(timeR, cRes.data)
					ax3.set_xlim([-trinwin['noise'], trinwin['noise']+trinwin['coda']])
					ax3.hlines(0,ax3.get_xlim()[0],ax3.get_xlim()[1])
					ax3.set_xlabel('Seconds')

					ax4 = self.fig.add_subplot(2,2,4)
					tRes = existDF.twoResTr.iloc[0]
					ax4.plot(timeR, tRes.data)
					ax4.set_xlim([-trinwin['noise'], trinwin['noise']+trinwin['coda']])
					ax4.hlines(0,ax3.get_xlim()[0],ax3.get_xlim()[1])
					ax4.set_xlabel('Seconds')
				elif self._method == 'coda':
					ax1 = self.fig.add_subplot(2,1,1)
					ax1.plot(time, np.log10(data), 'k')
					data_coda = self._current_event.codaTr.data
					ax1.plot(time, np.log10(data_coda), 'r')
					ax1.set_xlim([stime, etime])
					ax1.set_ylim([-1,5])
					ax1.set_xlabel('Seconds')
					ax1.set_ylabel('log10(Amp)')
					ax3 = self.fig.add_subplot(2,1,2)
					cRes = self._current_event.codaResTr
					timeR = np.arange(cRes.stats.npts)*cRes.stats.delta - trinwin['noise']
					ax3.plot(timeR, cRes.data)
					ax3.set_xlim([-trinwin['noise'], trinwin['noise']+trinwin['coda']])
					ax3.hlines(0,ax3.get_xlim()[0],ax3.get_xlim()[1])
					ax3.set_xlabel('Seconds')
				elif self._method == 'twoline':
					ax1 = self.fig.add_subplot(2,1,1)
					ax1.plot(time, np.log10(data), 'k')
					twoTr = self._current_event.twoTr
					data_two = twoTr.data
					data_time = np.arange(twoTr.stats.npts) * delta + (twoTr.stats.starttime - self._current_beam.stats.starttime)
					ax1.plot(data_time,data_two,'r')
					ax1.set_xlabel('Seconds')
					ax1.set_xlim([stime, etime])
					ax1.set_ylim([-1,5])
					ax1.set_ylabel('log10(Amp)')

					ax4 = self.fig.add_subplot(2,1,2)
					tRes = self._current_event.twoResTr
					timeR = np.arange(tRes.stats.npts)*tRes.stats.delta - trinwin['noise']
					ax4.plot(timeR, tRes.data)
					ax4.set_xlim([-trinwin['noise'], trinwin['noise']+trinwin['coda']])
					ax4.hlines(0,ax4.get_xlim()[0],ax4.get_xlim()[1])
					ax4.set_xlabel('Seconds')
				self.fig.suptitle('Coda Strip for %s using %s method in win %s'%(self._current_event.ID, self._method, trinwin['name']))

		
		self._canvasDraw()

	def _updatePlot(self):

		self._btype = self.sbcb.currentText()
		self._drawFig()

	def _canvasDraw(self):
		"""
		Redraws the canvas and re-set mouse focus
		"""
		# if isinstance(st, obspy.core.stream.Stream):
		# 	delta = st[0].stats.delta
		# elif isinstance(st, obspy.core.trace.Trace):
		# 	delta = st.stats.delta

		for _i, _ax in enumerate(self.fig.get_axes()):
			_ax.set_xticklabels(_ax.get_xticks())
		self.fig.canvas.draw()
		self.canvas.setFocus()

	def _pltOnScroll(self, event):
		"""
		Scrolls/Redraws the plot along x axis
		"""

		if event.inaxes is None:
			return

		if event.key == 'control':
			axes = [event.inaxes]
		else:
			axes = self.fig.get_axes()

		for _ax in axes:
			left = _ax.get_xlim()[0]
			right = _ax.get_xlim()[1]
			extent_x = right - left
			dxzoom = .2 * extent_x
			aspect_left = (event.xdata - _ax.get_xlim()[0]) / extent_x
			aspect_right = (_ax.get_xlim()[1] - event.xdata) / extent_x

			up = _ax.get_ylim()[1]
			down = _ax.get_ylim()[0]
			extent_y = up - down
			dyzoom = 0.5 * extent_y
			aspect_down = (0 - _ax.get_ylim()[0]) / extent_y
			aspect_up = _ax.get_ylim()[1] / extent_y

			if event.button == 'up':
				left += dxzoom * aspect_left
				right -= dxzoom * aspect_right
				down += dyzoom * aspect_down
				up -= dyzoom * aspect_up
			elif event.button == 'down':
				left -= dxzoom * aspect_left
				right += dxzoom * aspect_right
				down -= dyzoom * aspect_down
				up += dyzoom * aspect_up
			else:
				return
			_ax.set_xlim([left, right])
			_ax.set_ylim([down, up])

		self._canvasDraw()

	def _pltOnDrag(self, event):
		"""
		Drags/redraws the plot upon drag
		"""
		if event.inaxes is None:
			return

		if event.key == 'control':
			axes = [event.inaxes]
		else:
			axes = self.fig.get_axes()

		if event.button == 1:
			if self._plt_drag is None:
				self._plt_drag = event.xdata
				return
			for _ax in axes:
				_ax.set_xlim([_ax.get_xlim()[0] + 
							  (self._plt_drag - event.xdata), _ax.get_xlim()[1] + (self._plt_drag - event.xdata)])
		else:
			return
		self._canvasDraw()

	def _pltOnButtonRelease(self, event):
		"""
		On Button Release Reset drag variable
		"""

		self._plt_drag = None

	# def _pltOnButtonPress(self, event):

	# 	"""
	# 	This function is using for zoom in relative phase region
	# 	"""

	def _saveFile(self):
		if self.savefile is None:
			return self._saveFileFormat()
		savefile = str(self.savefile)
		if os.path.splitext(savefile)[1].lower() == '.pkl':
			self._savePickle(savefile)
		elif os.path.splitext(savefile)[1].lower() == '.csv':
			self._saveCSV(savefile)
	def _saveFileFormat(self):
		files_types = "Pickle (*.pkl);; CSV (*.csv)"
		self.savefile = QtGui.QFileDialog.getSaveFileName(self,
										'Save as', os.getcwd(), files_types)
		self.savefile = str(self.savefile)
		if os.path.splitext(self.savefile)[1].lower() == '.pkl':
			self._savePickle(self.savefile)
		elif os.path.splitext(self.savefile)[1].lower() == '.csv':
			self._saveCSV(self.savefile)

	def _savePickle(self, filename):
		self._stripDF.to_pickle(filename)

	def _saveCSV(self, filename):
		_stripDF = self._stripDF
		_stripDF.drop(['codaTr','twoTr','twoResTr','codaResTr'])
		_stripDF.to_csv(filename,index=False,sep=',')

	def _openFile(self):
		filename = QtGui.QFileDialog.getOpenFileName(self,'Load Pickle File',
													os.getcwd(), 'Pickl Format (*.pkl)', '20')
		if filename:
			self._stripDF = pd.read_pickle(str(filename))
			self.savefile = str(filename)

	def _savePlot(self):
		# path = os.getcwd()
		# path = os.path.join(path,self.array.name,self._current_event.ID)
		file_types = "Image Format (*.png *.pdf *.ps *.eps);; ALL (*)"
		filename = QtGui.QFileDialog.getSaveFileName(self, 'Save Plot',
													 os.getcwd(), file_types)
		if not filename:
			return
		filename = str(filename)
		formats = os.path.splitext(filename)[1][1:].lower()
		if formats not in ['png', 'pdf', 'ps', 'eps']:
			formats = 'png'
			filename += '.' +formats
		self.fig.savefig(filename=filename, format=formats, dpi=72)
	def closeArray(self,event):

		if len(self._stripDF) > 0 and self.savefile is None:
			ask = QtGui.QMessageBox.question(self, 'Save stripping?',
				'Do you want to save your coda data?',
				QtGui.QMessageBox.Save |
				QtGui.QMessageBox.Discard |
				QtGui.QMessageBox.Cancel, QtGui.QMessageBox.Save)
			if ask == QtGui.QMessageBox.Save:
				self._saveFileFormat()
			elif ask == QtGui.QMessageBox.Cancel:
				event.ignore()


	class defWindow(QtGui.QDialog):
		def __init__(self, parent=None, windowvalue=None):
			"""
			Coda strip window dialog
			"""

			QtGui.QDialog.__init__(self, parent)
			self.setWindowTitle('Create new coda strip window')
			self.noisewin = QtGui.QDoubleSpinBox(decimals=1, maximum=400, minimum=20, singleStep=10, value=10)
			self.codawin = QtGui.QDoubleSpinBox(decimals=1, maximum=400, minimum=20, singleStep=10, value=10)
			self.stime = QtGui.QDoubleSpinBox(decimals=1, maximum=600, minimum=0, singleStep=50, value=50)
			self.etime = QtGui.QDoubleSpinBox(decimals=1, maximum=2400, minimum=1600, singleStep=50, value=50)

			self.winName = QtGui.QLineEdit('Window Name')
			self.winName.selectAll()

			self.model = QtGui.QLineEdit('ak135')
			self.model.selectAll()

			grid = QtGui.QGridLayout()
			grid.addWidget(QtGui.QLabel('Window Name'), 0, 0)
			grid.addWidget(self.winName, 0, 1)
			grid.addWidget(QtGui.QLabel('Noise Win.'), 1, 0)
			grid.addWidget(self.noisewin, 1, 1)
			grid.addWidget(QtGui.QLabel('Coda Win.'), 2, 0)
			grid.addWidget(self.codawin, 2, 1)
			grid.addWidget(QtGui.QLabel('Start Time.'), 3, 0)
			grid.addWidget(self.stime, 3, 1)
			grid.addWidget(QtGui.QLabel('Start Time.'), 4, 0)
			grid.addWidget(self.etime, 4, 1)
			grid.addWidget(QtGui.QLabel('Model.'), 5, 0)
			grid.addWidget(self.model, 5, 1)
			grid.setVerticalSpacing(10)

			btnbox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok |
											QtGui.QDialogButtonBox.Cancel)

			btnbox.accepted.connect(self.accept)
			btnbox.rejected.connect(self.reject)

			layout = QtGui.QVBoxLayout()
			layout.addWidget(QtGui.QLabel('Define noise window and coda window for stripping'))
			layout.addLayout(grid)
			layout.addWidget(btnbox)

			if windowvalue is not None:
				self.winName.setText(windowvalue['name'])
				self.noisewin.setValue(windowvalue['noise'])
				self.codawin.setValue(windowvalue['coda'])
			self.setLayout(layout)
			self.setSizeGripEnabled(False)

		def getValues(self):
			"""
			Return window dialog values as a dictionary
			"""

			return dict(name=str(self.winName.text()),
						noise=float(self.noisewin.cleanText()),
						coda=float(self.coda.cleanText()))	

# class for event stacking in arrays
class stackArray(QtGui.QMainWindow):

	def __init__(self, arraylist=None, parent=None, ap=None):
		if ap is None:
			self.qApp = QtGui.QApplication(sys.argv)
		else:
			self.qApp = ap
		if isinstance(arraylist, str):
			arlist = pd.read_csv(arraylist, delimiter='\s+')
		elif isinstance(arraylist, pd.DataFrame):
			arlist = arraylist
		else:
			msg = 'Define array list in DataFrame or a path to a csv file'
			raise ValueError(msg)

		self._shortcuts = {'arr_next': 'n',
						   'arr_prev': 'p',
						   'stack': 's'}

		self._list = arlist
		self._initArrayList()
		self._arrayCycle = cycle(self._namelist)
		self.dis = []
		self._arrayInfo(self._arrayCycle.next())
		QtGui.QMainWindow.__init__(self)
		self.setupUI()

	def setupUI(self):
		self.main_widget = QtGui.QWidget(self)
		self._initMenu()
		self._createStatusBar()
		self._initPlots()

		l = QtGui.QVBoxLayout(self.main_widget)

		l.addLayout(self.btnbar)
		l.addWidget(self.canvas)
		self.setCentralWidget(self.main_widget)
		self.setGeometry(300, 300, 1200, 800)
		self.setWindowTitle('Array Stack')
		self.show()

	def _killLayout():
		pass
	def _initPlots(self):
		self.fig = Figure(dpi=100)
		self.canvas = FigureCanvas(self.fig)
		self.canvas.setFocusPolicy(PyQt4.QtCore.Qt.StrongFocus)
		self._drawFig()

		# self.fig.canvas.mpl_connect()

	def _initMenu(self):
		# Next and Prev array
		nxt = QtGui.QPushButton('Next >>',
			shortcut=self._shortcuts['arr_next'], parent=self.main_widget)
		nxt.clicked.connect(self._pltNextArray)
		nxt.setToolTip('shortcut <b>n</d>')
		nxt.setMaximumWidth(150)
		prv = QtGui.QPushButton('Prev >>',
			shortcut=self._shortcuts['arr_prev'], parent=self.main_widget)
		prv.clicked.connect(self._pltPrevArray)
		prv.setToolTip('shortcut <b>p</d>')
		prv.setMaximumWidth(150)

		# Array drop-down
		self.arcb = QtGui.QComboBox(self)
		for arr in self._namelist:
			self.arcb.addItem(arr)
		self.arcb.activated.connect(self._pltArray)
		self.arcb.setMaximumWidth(1000)
		self.arcb.setMinimumWidth(80)

		# Select region

		# Stacking earthquakes in array
		self.stbtn = QtGui.QPushButton('Stack',
					shortcut=self._shortcuts['stack'], parent=self.main_widget)
		self.stbtn.setToolTip('shortcut <b>s</b>')
		self.stbtn.clicked.connect(self._appStack)

		self.discb = QtGui.QComboBox(self)
		self.discb.activated.connect(self._changeStack)
		self._updateWindow()

		disEdit = QtGui.QPushButton('Edit')
		disEdit.resize(disEdit.sizeHint())
		disEdit.clicked.connect(self._editDis)

		disDelt = QtGui.QPushButton('Delete')
		disDelt.resize(disEdit.sizeHint())
		disDelt.clicked.connect(self._deleteDis)
		# Select distance
		self.btnbar = QtGui.QHBoxLayout()
		self.btnbar.addWidget(prv)
		self.btnbar.addWidget(nxt)
		self.btnbar.addWidget(QtGui.QLabel('Array'))
		self.btnbar.addWidget(self.arcb)

		vline = QtGui.QFrame()
		vline.setFrameStyle(QtGui.QFrame.VLine | QtGui.QFrame.Raised)
		self.btnbar.addWidget(vline)
		self.btnbar.addWidget(self.stbtn)
		self.btnbar.addWidget(self.discb)
		self.btnbar.addWidget(disEdit)
		self.btnbar.addWidget(disDelt)
		self.btnbar.addStretch(1)


	def _initArrayList(self):
		self._arlist = pd.DataFrame()
		for _ind, _ar in self._list.iterrows():
			name = _ar.NAME
			tmp = os.path.join(name,_ar.FILE)
			tmp_df = pd.read_pickle(tmp)
			newRow = {'NAME':name,'DF':tmp_df,
					  'LAT':_ar.LAT, 'LON':_ar.LON}
			self._arlist = self._arlist.append(newRow, ignore_index=True)
		self._namelist = self._list.NAME.tolist()

	def _arrayInfo(self, name):
		ardf = self._arlist[self._arlist.NAME == name].iloc[0]
		array = {'name':ardf.NAME,'lat':ardf.LAT,'lon':ardf.LON}
		self._current_array = array
		self._current_array_df = ardf.DF

	def _createStatusBar(self):
		sb = QtGui.QStatusBar()
		sb.setFixedHeight(18)
		self.setStatusBar(sb)
		self.statusBar().showMessage('Ready')

	def _drawFig(self):
		self.fig.clear()
		self.fig.add_subplot(1, 1, 1)
		_ax = self.fig.get_axes()

		m = Basemap(projection='cyl', lon_0=-180.0, lat_0=0.0,
					area_thresh=10000,ax=_ax[0])
		x0, y0 = m(self._current_array['lon'], self._current_array['lat'])
		x, y = m(self._current_array_df.lon.tolist(), self._current_array_df.lat.tolist())
		m.drawcoastlines(ax=_ax[0])
		m.drawmapboundary(ax=_ax[0])
		m.fillcontinents(color='lightgray',lake_color='white',ax=_ax[0])
		parallels = np.arange(-90.0, 90.0, 60)
		m.drawparallels(parallels,ax=_ax[0])
		meridians = np.arange(-180.0, 180.0, 60.0)
		m.drawmeridians(meridians,ax=_ax[0])
		m.scatter(x0, y0, marker='*',c='y',s=100,alpha=0.7,ax=_ax[0],zorder=10)
		m.scatter(x, y, marker='o', c='lightblue', s=50, alpha=0.7,ax=_ax[0],zorder=10)

		self.fig.suptitle('Earthquakes in array %s'%self._current_array['name'])
		self._canvasDraw()

	def _canvasDraw(self):

		for _i, _ax in enumerate(self.fig.get_axes()):
			_ax.set_xticklabels(_ax.get_xticks())
		self.fig.canvas.draw()
		self.canvas.setFocus()

	def _pltArray(self):
		_i = self.arcb.currentIndex()
		while self._arrayCycle.next() != self._namelist[_i]:
			pass
		self._arrayInfo(self._namelist[_i])
		self._drawFig()

	def _pltPrevArray(self):
		_j = self.arcb.currentIndex()
		for _i in range(len(self._namelist) - 1):
			prevarray = self._arrayCycle.next()
		self._arrayInfo(prevarray)
		self.arcb.setCurrentIndex(_j-1)
		self._drawFig()

	def _pltNextArray(self):
		_i = self.arcb.currentIndex()
		self._arrayInfo(self._arrayCycle.next())
		self.arcb.setCurrentIndex(_i+1)
		self._drawFig()

	def _appStack(self):
		_i = self.discb.currentIndex()
		self._arrayInfo(self._current_array['name'])
		savefile = None
		if self.dis[_i]['write']:
			savefile = self.dis[_i]['name'] + '.sac'
		win = self._current_array_df['win'].iloc[0]
		window = [win['noise'], win['coda']]
		self.stackTr, self.stdTr = stackTR(self._current_array_df,
										   sacname=savefile,win=window,
										   step=self.dis[_i]['step'],
										   overlap=self.dis[_i]['overlap'],
										   write=self.dis[_i]['write'])

		self._drawStack(self)

	def _drawStack(self):
		self.fig.clear()
		self.fig.add_subplot(121)
		_ax = self.fig.get_axes()[0]
		m = Basemap(projection='cyl', lon_0=-180.0, lat_0=0.0,
					area_thresh=10000,ax=_ax[0])
		x0, y0 = m(self._current_array['lon'], self._current_array['lat'])
		x, y = m(self._current_array_df.lon.tolist(), self._current_array_df.lat.tolist())
		m.drawcoastlines(ax=_ax[0])
		m.drawmapboundary(ax=_ax[0])
		m.fillcontinents(color='lightgray',lake_color='white',ax=_ax[0])
		parallels = np.arange(-90.0, 90.0, 60)
		m.drawparallels(parallels,ax=_ax[0])
		meridians = np.arange(-180.0, 180.0, 60.0)
		m.drawmeridians(meridians,ax=_ax[0])
		m.scatter(x0, y0, marker='*',c='y',s=100,alpha=0.7,ax=_ax[0],zorder=10)
		m.scatter(x, y, marker='o', c='lightblue', s=50, alpha=0.7,ax=_ax[0],zorder=10)

		self.fig.add_subplot(122)
		_ax = self.fig.get_axes()[1]
		_ax.set_xlabel('Time (s)')
		delta = self.stackTr.stats.delta
		npts = self.stackTr.stats.npts
		time = np.arange(npts)*delta + self.stackTr.stats.sac.b
		_ax.plot(time, self.stackTr.data,'darkred')
		_ax.errorbar(time, self.stackTr.data, yerr=2*self.stdTr.data,
					 marker='.',mew=0.1, ecolor='red', linewidth=0.2, markersize=0.2,
					 capsize=0.1, alpha=0.5)
		self._canvasDraw()


	def _changeStack(self,index):
		if index == len(self.dis):
			return self._newDis()
		else:
			return self._appStack()

	def _newDis(self):
		newDis = self.defDisStep(self)

		if newDis.exec_():
			self.dis.append(newDis.getValues())
			self._updateWindow()
			self.discb.setCurrentIndex(len(self.dis)-1)
			self._appStack()

	def _editDis(self):
		_i = self.discb.currentIndex()
		this_window = self.dis[_i]
		editWindow = self.defWindow(self, this_window)
		if editWindow.exec_():
			self.dis[_i] = editWindow.getValues()
			self.updateWindow()
			self.discb.setCurrentIndex(_i)
			self._appStack()

	def _deleteDis(self):
		pass
		_i = self.discb.currentIndex()

	def _updateWindow(self):
		self.discb.clear()
		self.discb.setCurrentIndex(-1)
		for _i, _f in enumerate(self.dis):
			self.discb.addItem('Step %.2f deg - overlap %.2f ' %(_f['step'], _f['overlap']))
		self.discb.addItem('Create new distance stack')

	class defDisStep(QtGui.QDialog):
		def __init__(self, parent=None, stepvalue=None):

			QtGui.QDialog.__init__(self, parent)
			self.setWindowTitle('Create new stacking distance step')
			self.step = QtGui.QDoubleSpinBox(decimals=1, maximum=20, minimum=0.1, singleStep=0.1, value=0.1)
			self.overlap = QtGui.QDoubleSpinBox(decimals=2, maximum=1.0, minimum=0.0, singleStep=0.1, value=0.1)
			self.Name = QtGui.QLineEdit('Name')
			self.Name.selectAll()
			self.saveTr = ['True', 'False']

			grid = QtGui.QGridLayout()
			grid.addWidget(QtGui.QLabel('Name'), 0, 0)
			grid.addWidget(self.NAme, 0, 1)
			grid.addWidget(QtGui.QLabel('Step'), 1, 0)
			grid.addWidget(self.step, 1, 1)
			grid.addWidget(QtGui.QLabel('Overlap'), 2, 0)
			grid.addWidget(self.overlap, 2, 1)

			_savebtn = [QtGui.QRadioButton("Yes"), QtGui.QRadioButton("No")]
			self.saveGrp = QtGui.QButtonGroup()
			self.saveGrp.setExclusive(True)
			sbtn = QtGui.QHBoxLayout()

			for _i, _btn in enumerate(_savebtn):
				self.saveGrp.addButton(_btn, _i)
				sbtn.addWidget(_btn)
			grid.addWidget(QtGui.QLabel('Save Stack'), 3, 0)
			grid.addWidget(sbtn, 3, 1)

			btnbox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok |
											QtGui.QDialogButtonBox.Cancel)

			btnbox.accepted.connect(self.accept)
			btnbox.rejected.connect(self.reject)

			layout = QtGui.QVBoxLayout()
			layout.addWidget(QtGui.QLabel('Define distance steps and overlap for stacking'))
			layout.addLayout(grid)
			layout.addWidget(btnbox)

			if stepvalue is not None:
				self.Name.setText(stepvalue['name'])
				self.step.setValue(stepvalue['step'])
				self.overlap.setValue(stepvalue['overlap'])
			self.setLayout(layout)
			self.setSizeGripEnabled(False)

		def getValues(self):

			savefile = self.saveTr(self.saveGrp.checkedId())

			return dict(name=str(self.Name),
						step=float(self.step.cleanText()),
						overlap=float(self.overlap.cleanText()),
						write=bool(savefile))



def codaStrip(eve, beamtype='beam', method='all',
			  siglen=200, noise=200,phase='PKiKP', 
			  model='ak135', stime=400.0, etime=1800.0,
			  write=False):
	"""
	Function to remove background coda noise for events
	"""
	print 'doing coda strip'
	
	if not hasattr(eve, 'arrival'):
		eve.getArrival(phase=phase,model=model)
		
	if beamtype == 'beam':
		tr = eve.beam
	elif beamtype == 'slide':
		tr = eve.slideSt
	else:
		msg = ('Not a valid option for codastrip')
		gpar.log(__name__,msg,level='error',pri=True)

	delta = tr.stats.delta
	starttime = tr.stats.starttime
	tt1 = eve.arrival - noise - starttime
	tt2 = eve.arrival + siglen - starttime
	tari = eve.arrival - starttime
	data = np.abs(scipy.signal.hilbert(tr.data))
	sig_pts = int(siglen/delta) + 1
	noi_pts = int(noise/delta) + 1
	noi_ind1 = int(tt1/delta)
	sig_ind = int((eve.arrival - starttime)/delta)
	noi_ind2 = int(tt2/delta)
	time_before = tt1 + np.arange(int(noise/delta)+1) * delta
	data_before = data[noi_ind1: noi_ind1 + noi_pts]
	
	data_sig = data[sig_ind:sig_ind + sig_pts]

	time_after = tt2 + np.arange(int(noise/delta)+1) * delta
	data_after = data[noi_ind2: noi_ind2+noi_pts]
	sind = int(stime/delta)
	npts = int((etime - stime)/delta) + 1
	time = np.linspace(stime, etime, npts)
	obs_data = data[sind: sind+npts] 
	ind = int((tari - stime)/delta)
	if method == 'all':
		#fitting coda model
		coda_par = codaFit(np.append(time_before,time_after),np.append(data_before,data_after))
		#getting predict noise signal in linear scale
		coda_data = np.exp(coda_par[0][0] - coda_par[1][0]*np.log(time) - coda_par[2][0]*time)
		#getting residual signal after removing the predict noise
		coda_res = obs_data - coda_data
		res = np.mean(coda_res[ind:ind+sig_pts])
		#store coda model information
		codamod = {'RES':res,'lnA':coda_par[0][0],'B':coda_par[1][0],'C':coda_par[2][0]}
		eve.codaMod = codamod
		codaTr = obspy.core.trace.Trace()
		codaTr.stats.delta = delta
		codaTr.stats.npts = npts
		codaTr.stats.starttime = starttime + stime
		codaTr.data = coda_data
		eve.codaTr = codaTr
		resTr = obspy.core.trace.Trace()
		resTr.stats.delta = delta
		resTr.stats.starttime = eve.arrival - noise
		res_ind = int((tt1 - stime)/delta)
		pts = int((noise*2+siglen)/delta)+1
		resTr.data = coda_res[res_ind: res_ind+pts]
		resTr.stats.npts = pts
		eve.codaResTr = resTr
		#fittint twoline model
		twoline_par_before = twoLineFit(time_before, data_before)
		twoline_par_after = twoLineFit(time_after, data_after)
		y1 = twoline_par_before[0][0] + twoline_par_before[1][0] * tari
		y2 = twoline_par_after[0][0] + twoline_par_after[1][0] * tt2
		k = (y2 - y1)/(tt2 - tari)
		b = y2 - k * tt2
		t1 = np.linspace(tt1,tari, int(noise/delta)+1)
		d1 = twoline_par_before[0][0] + twoline_par_before[1][0] * t1
		t2 = np.linspace(tari+delta, tt2, int(siglen/delta))
		d2 = k * t2 + b
		t3 = np.linspace(tt2+delta,tt2+noise, int(noise/delta))
		d3 = twoline_par_after[0][0] + twoline_par_after[1][0] * t3
		two_data = np.append(d1,d2)
		two_data = np.append(two_data,d3)
		two_res = obs_data[res_ind: res_ind+pts] - two_data
		res = np.mean(two_res[int(noise)/delta:int(noise)/delta+sig_pts])
		twomod = {'kn1':twoline_par_before[1][0],'bn1':twoline_par_before[0][0],
				  'kn2':twoline_par_after[1][0],'bn2':twoline_par_after[0][0],'RES':res}
		eve.twoMod = twomod
		twoTr = obspy.core.trace.Trace()
		twoTr.stats.delta = delta
		twoTr.stats.starttime = starttime + tt1
		twoTr.data = two_data
		eve.twoTr = twoTr
		resTr = obspy.core.trace.Trace()
		resTr.stats.delta = delta
		resTr.stats.starttime = eve.arrival - noise
		res_ind = int((tt1 - stime)/delta)
		pts = int((noise*2+siglen)/delta)+1
		resTr.data = two_res
		resTr.stats.npts = pts
		eve.twoResTr = resTr	
	elif method == 'coda':
		#fitting coda model
		coda_par = codaFit(np.append(time_before,time_after),np.append(data_before,data_after))
		coda_data = np.exp(coda_par[0][0] - coda_par[1][0]*np.log(time) - coda_par[2][0]*time)
		coda_res = obs_data - coda_data
		res = np.mean(coda_res[ind:ind+sig_pts])
		codamod = {'RES':res,'lnA':coda_par[0][0],'B':coda_par[1][0],'C':coda_par[2][0]}
		eve.codaMod = codamod
		codaTr = obspy.core.trace.Trace()
		codaTr.stats.delta = delta
		codaTr.stats.npts = npts
		codaTr.stats.starttime = starttime + stime
		codaTr.data = coda_data
		eve.codaTr = codaTr
		resTr = obspy.core.trace.Trace()
		resTr.stats.delta = delta
		resTr.stats.starttime = eve.arrival - noise
		res_ind = int((tt1 - stime)/delta)
		pts = int((noise*2+siglen)/delta)+1
		resTr.data = coda_res[res_ind: res_ind+pts]
		resTr.stats.npts = pts
		eve.codaResTr = resTr
	elif method == 'twoline':
		#fittint twoline model
		twoline_par_before = twoLineFit(time_before, data_before)
		twoline_par_after = twoLineFit(time_after, data_after)
		res_ind = int((tt1 - stime)/delta)
		pts = int((noise*2+siglen)/delta)+1
		y1 = twoline_par_before[0][0] + twoline_par_before[1][0] * tari
		y2 = twoline_par_after[0][0] + twoline_par_after[1][0] * tt2
		k = (y2 - y1)/(tt2 - tari)
		b = y2 - k * tt2
		t1 = np.linspace(tt1,tari, int(noise/delta)+1)
		d1 = twoline_par_before[0][0] + twoline_par_before[1][0] * t1
		t2 = np.linspace(tari+delta, tt2, int(siglen/delta))
		d2 = k * t2 + b
		t3 = np.linspace(tt2+delta,tt2+noise, int(noise/delta))
		d3 = twoline_par_after[0][0] + twoline_par_after[1][0] * t3
		two_data = np.append(d1,d2)
		two_data = np.append(two_data,d3)
		two_res = obs_data[res_ind: res_ind+pts] - two_data
		res = np.mean(two_res[int(noise)/delta:int(noise)/delta+sig_pts])
		twomod = {'kn1':twoline_par_before[1][0],'bn1':twoline_par_before[0][0],
				  'kn2':twoline_par_after[1][0],'bn2':twoline_par_after[0][0],'RES':res}
		eve.twoMod = twomod
		twoTr = obspy.core.trace.Trace()
		twoTr.stats.delta = delta
		twoTr.stats.starttime = starttime + tt1
		twoTr.data = two_data
		eve.twoTr = twoTr
		resTr = obspy.core.trace.Trace()
		resTr.stats.delta = delta
		resTr.stats.starttime = eve.arrival - noise
		res_ind = int((tt1 - stime)/delta)
		pts = int((noise*2+siglen)/delta)+1
		resTr.data = two_res
		resTr.stats.npts = pts
		eve.twoResTr = resTr
	return eve




def codaFit(time, amp):
	"""
	Function to remove background coda noise by fitting coda model s=A*B^-t*exp(-Ct) in log scale
	"""
	amp = np.matrix(np.log(amp))
	n = len(time)
	amp = amp.reshape(n,1)
	A = np.zeros((n,3))
	a = np.ones(time.shape)
	b = -np.log(time)
	c = -time

	A[:,0] = a
	A[:,1] = b
	A[:,2] = c
	A = np.matrix(A)
	G = A.transpose() * A
	d = A.transpose() * amp
	m = np.linalg.inv(G) * d

	return np.asarray(m)

def twoLineFit(time, amp):
	"""
	Function to do the two line method fitting to get the coda
	"""

	amp = np.matrix(np.log10(amp))
	n = len(time)
	amp = amp.reshape(n,1)

	A = np.zeros((n,2))
	A[:,0] = np.ones(time.shape)
	A[:,1] = time

	A = np.matrix(A)
	G = A.transpose() * A
	d = A.transpose() * amp
	m = np.linalg.inv(G) * d

	return np.asarray(m)

def moving_ave(x, window=5):
	"""
	Function to smooth data
	"""

	weigths = np.ones(window)/window
	y = np.convolve(x,weigths,mode='same')

	return y

def norm(st,wins=[0.0,200.0]):
	btime = st.stats.sac.b
	delta = st.stats.sac.delta
	startIndex = int((wins[0] - btime)/delta)
	endIndex = int((wins[1]-btime)/delta)

	peak = np.max(np.absolute(st.data[startIndex:endIndex]))
	data = st.data/peak
	st.data = data

	return st

def stack(df,win):

	data = df.DATA.values
	std_data = np.std(data,axis=0)/np.sqrt(len(data))
	#stack_data = np.sum(data,axis=0)/len(df)
	stack_data = np.mean(data)
	peak = np.max(np.absolute(stack_data[win[0]:win[1]]))
	stack_data = stack_data/peak

	return stack_data, std_data

def misfit(data1,data2):

	l2 = np.sum((data1-data2)**2)

	return l2

def stackTR(obsdf, sacname=None,win=[200.0,200.0],
		   step=5.0,overlap=0.0, write=False):
	DATA = [''] * len(obsdf)
	for ind, row in obsdf.iterrows():
		DATA[ind] = row.codaTr.data
	obsdf['DATA'] = DATA
	n = len(obsdf)
	t = obsdf.iloc[0]
	stats = t.codaResTr.stats
	npt = stats.npts
	delta = stats.delta
	time = np.arange(npt) * delta - win[0]
	sind = int((win[0])/delta)
	eind = int((win[1])/delta)

	stack_obs,std_obs = stack(obsdf,[sind,sind + eind])

	tr_obs = Trace()
	tr_obs.stats.sac = AttribDict()
	tr_obs.stats.delta = delta
	tr_obs.stats.npts = npt
	tr_obs.stats.station = n
	tr_obs.data = stack_obs
	tr_obs.stats.sac.delta = delta
	tr_obs.stats.sac.b = b
	tr_obs.stats.sac.e = e
	tr_obs.stats.sac.npts = npt

	tr_std = tr_obs.copy()

	tr_std.data = std_obs

	if write != None:
		if sacname is None:
			sacname = 'stack.sac'
		tr_obs.write(sacname,format='SAC')

	return tr_obs, tr_std
