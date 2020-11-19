"""
GUI code modified based on https://github.com/miili/StreamPick

For earthquake PKiKP coda quality evaluation and stack

"""

import os
import pickle
import pandas as pd
import numpy as np

# GUI import
import PyQt5
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import signal
import scipy
import gpar
from gpar.util import util
from itertools import cycle

#figure plot import
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.transforms import offset_copy
from matplotlib.widgets import RectangleSelector
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid
# from mpl_toolkits.axesgrid1 import make_axes_locatable

#obspy import
from obspy.taup import TauPyModel
import obspy
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy.core import read
from obspy.core import AttribDict

signal.signal(signal.SIGINT, signal.SIG_DFL)

# color = list(mcolors.cnames.values())
color = ['red', 'blue', 'green','yellow','cyan','magenta','purple']

#class for event first evaluation
class glanceEQ(QtWidgets.QMainWindow):

	def __init__(self, array=None, parent=None, ap=None):

		if ap is None:
			self.qApp = QtWidgets.QApplication(sys.argv)
		else:
			self.qApp = ap
		self.KeepGoing = False

		if isinstance(array, str):
			ar = util.loadArray(array)
		elif isinstance(array, gpar.arrayProcess.Array):
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
						 'stime':400.0,'etime':1800,
						 'smooth':4.0,'model':'ak135'}]
		self._current_win = None
		self._current_strip = False
		self._eventCycle = cycle(self._eqlist)
		self._eventInfo(next(self._eventCycle))
		QMainWindow.__init__(self)
		self.setupUI()


	def setupUI(self):
		self.main_widget = QtWidgets.QWidget(self)
		self._initMenu()
		self._createStatusBar()
		self._initPlots()

		l = QVBoxLayout(self.main_widget)
		l.addLayout(self.btnbar)
		l.addLayout(self.btnbar2)
		l.addWidget(self.canvas)

		self.setCentralWidget(self.main_widget)
		self.setGeometry(300, 300, 1200, 800)
		self.setWindowTitle('Array Analysis: %s'%self.array.name)
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
		self.canvas.setFocusPolicy(PyQt5.QtCore.Qt.StrongFocus)
		self._drawFig()

		# connect the events
		self.fig.canvas.mpl_connect('scroll_event', self._pltOnScroll)
		self.fig.canvas.mpl_connect('motion_notify_event', self._pltOnDrag)
		self.fig.canvas.mpl_connect('button_release_event', self._pltOnButtonRelease)

	def _initMenu(self):
		# Next and Prev Earthquake
		nxt = QtWidgets.QPushButton('Next >>',
			shortcut=self._shortcuts['eve_next'], parent=self.main_widget)
		nxt.clicked.connect(self._pltNextEvent)
		nxt.setToolTip('shortcut <b>n</d>')
		nxt.setMaximumWidth(150)
		prv = QPushButton('Prev >>',
			shortcut=self._shortcuts['eve_prev'], parent=self.main_widget)
		prv.clicked.connect(self._pltPrevEvent)
		prv.setToolTip('shortcut <b>p</d>')
		prv.setMaximumWidth(150)

		# Earthquake drop-down
		self.evecb = QComboBox(self)
		for eve in self._eqlist:
			self.evecb.addItem(eve)
		self.evecb.activated.connect(self._pltEvent)
		self.evecb.setMaximumWidth(1000)
		self.evecb.setMinimumWidth(80)

		# coda strip button
		self.codabtn = QtWidgets.QPushButton('Strip',
			shortcut=self._shortcuts['strip'],parent=self.main_widget)
		self.codabtn.setToolTip('shortcut <b>s</b>')
		self.codabtn.clicked.connect(self._appStrip)

		self.codacb = QComboBox(self)
		for med in ['all', 'coda','twoline']:
			self.codacb.addItem(med)
		self.codacb.activated.connect(self._selectMethod)
		self.codacb.setMaximumWidth(100)
		self.codacb.setMinimumWidth(80)
		self.wincb = QComboBox(self)
		self.wincb.activated.connect(self._changeStrip)
		self._updateWindow()

		# edit/delete coda selected window
		winEdit = QtWidgets.QPushButton('Coda Window')
		winEdit.resize(winEdit.sizeHint())
		winEdit.clicked.connect(self._editTimeWindow)

		winDelt = QtWidgets.QPushButton('Delete')
		winDelt.resize(winDelt.sizeHint())
		winDelt.clicked.connect(self._deleteWin)

		# Coda level
		_radbtn = []
		for _o in self.eve_type:
			_radbtn.append(QRadioButton(_o.upper(), shortcut=self._shortcuts[_o.upper()]))
			_radbtn[-1].setToolTip('Level: '+_o)
		self.levelGrp = QButtonGroup()
		self.levelGrp.setExclusive(True)
		levelbtn = QHBoxLayout()

		for _i, _btn in enumerate(_radbtn):
			self.levelGrp.addButton(_btn, _i)
			levelbtn.addWidget(_btn)

		# plot slide beam figure button
		self.sbcb = QComboBox(self)
		for btype in ['beam', 'slide', 'vespetrum','strip']:
			self.sbcb.addItem(btype)
		self.sbcb.activated.connect(self._updatePlot)

		self.vepcb = QComboBox(self)
		for scale in ['log10', 'log','sqrt','beam']:
			self.vepcb.addItem(scale)
		self.vepcb.activated.connect(self._updatePlot )
		self.vepcb.setEnabled(False)

		self.codacb.setMaximumWidth(100)
		self.codacb.setMinimumWidth(80)
		self.ampmin = QDoubleSpinBox(decimals=1, maximum=5, minimum=-2, singleStep=.5, value=1)
		self.ampmax = QDoubleSpinBox(decimals=1, maximum=5, minimum=-2, singleStep=.5, value=3)
		self.ampmin.valueChanged.connect(self._updatePlot)
		self.ampmax.valueChanged.connect(self._updatePlot)
		self.ampmin.setEnabled(False)
		self.ampmax.setEnabled(False)
		# self._initAmp()
		self.sbcb.activated.connect(self._activeAmp)

		self.ttbtn = QtWidgets.QPushButton('Phases', parent=self.main_widget)
		self.ttbtn.setCheckable(True)
		self.ttbtn.clicked.connect(self._updatePlot)
		# Arrange buttons
		vline = QFrame()
		vline.setFrameStyle(QFrame.VLine | QFrame.Raised)
		self.btnbar = QHBoxLayout()
		self.btnbar.addWidget(prv)
		self.btnbar.addWidget(nxt)
		self.btnbar.addWidget(QLabel('Event'))
		self.btnbar.addWidget(self.evecb)
		##
		self.btnbar.addWidget(vline)
		self.btnbar.addWidget(self.codabtn)
		self.btnbar.addWidget(self.codacb)
		self.btnbar.addWidget(self.wincb)
		self.btnbar.addWidget(winEdit)
		self.btnbar.addWidget(winDelt)
		self.btnbar.addStretch(1)

		self.btnbar2 = QHBoxLayout()
		self.btnbar2.addWidget(QLabel('Level: '))
		self.btnbar2.addLayout(levelbtn)


		self.btnbar2.addWidget(vline)
		self.btnbar2.addWidget(QLabel('TYPE'))
		self.btnbar2.addWidget(self.sbcb)
		self.btnbar2.addWidget(vline)
		self.btnbar2.addWidget(QLabel('Scale'))
		self.btnbar2.addWidget(self.vepcb)
		self.btnbar2.addWidget(QLabel('AMP'))
		self.btnbar2.addWidget(self.ampmin)
		self.btnbar2.addWidget(self.ampmax)
		self.btnbar2.addWidget(vline)
		self.btnbar2.addWidget(self.ttbtn)
		self.btnbar2.addStretch(1)

		#Menubar
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(QtGui.QIcon().fromTheme('document-save'),
						   'Save', self._saveFile)
		fileMenu.addAction(QtGui.QIcon().fromTheme('document-save'),
						   'Save as', self._saveFileFormat)
		fileMenu.addSeparator()
		fileMenu.addAction(QIcon().fromTheme('document-open'),
						   'Load array', self._openArray)
		fileMenu.addAction(QtGui.QIcon().fromTheme('document-open'),
						   'Load Strip Pickle File', self._openFile)
		fileMenu.addSeparator()
		fileMenu.addAction(QtGui.QIcon().fromTheme('document-save'),
						   'Save Plot', self._savePlot)
		fileMenu.addSeparator()
		quit = QAction(QIcon().fromTheme('application-exit')," &Exit", self)
		fileMenu.addAction(quit)
		fileMenu.triggered[QAction].connect(self.closeArray)

	def _hardExist(self):
		self.deleteLater()


	def _activeAmp(self):
		if self.sbcb.currentText() == 'vespetrum':
			self.ampmin.setEnabled(True)
			self.ampmax.setEnabled(True)
			self.vepcb.setEnabled(True)
			if self.vepcb.currentText() == 'beam':
				self.ampmax.setMaximum(100000)
				# self.ampmax.setValue(1000)
				self.ampmax.setSingleStep(500)
				# self.ampmin.setValue(10)
				self.ampmin.setMaximum(100000)
				self.ampmin.setSingleStep(500)
			elif self.vepcb.currentText() == 'sqrt':
				self.ampmax.setMaximum(300)
				# self.ampmax.setValue(30)
				self.ampmax.setSingleStep(5)
				# self.ampmin.setValue(3)
				self.ampmin.setMaximum(300)
				self.ampmin.setSingleStep(5)
			elif self.vepcb.currentText() == 'log':
				self.ampmax.setMaximum(12)
				# # self.ampmax.setValue(7)
				self.ampmax.setSingleStep(1)
				# # self.ampmin.setValue(2)
				self.ampmin.setMaximum(12)
				self.ampmin.setSingleStep(1)
			elif self.vepcb.currentText() == 'log10':
				self.ampmax.setSingleStep(0.5)
				self.ampmin.setSingleStep(0.5)
				self.ampmax.setMaximum(5)
				self.ampmin.setMaximum(5)
		else:
			self.ampmin.setEnabled(False)
			self.ampmax.setEnabled(False)
			self.vepcb.setEnabled(False)

	def _createStatusBar(self):
		"""
		Creates the status bar
		"""
		sb =QStatusBar()
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
			self._updateWindow()
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
			self._updateWindow()
			self.wincb.setCurrentIndex(_i)
			self._appStrip()

	def _deleteWin(self):
		"""
		Delete window
		"""

		pass
		_i = self.wincb.currentIndex()

	def _updateWindow(self):
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
		spts = int(self.trinWin[_j]['smooth'] / self._current_delta )
		codaStrip(self._current_event, method=_method, window=spts,
				 	  siglen=self.trinWin[_j]['coda'], noise=self.trinWin[_j]['noise'],beamphase=self.beamphase,
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
		while next(self._eventCycle) != self._eqlist[_i]:
			pass
		self._eventInfo(self._eqlist[_i])
		self._current_strip = False
		_id = self._current_event.ID
		if len(self._stripDF) != 0:
			existDF = self._stripDF[self._stripDF.ID == _id]
		else:
			existDF = pd.DataFrame()
		if len(existDF) != 0:
			level = existDF.Level.iloc[0]
			ind = self.eve_type.index(level)
			self.levelGrp.button(ind).setChecked(True)
			self._current_strip=True
		else:
			if len(self._badDF) != 0:
				_badDF = self._badDF[self._badDF.ID == _id]
				if len(_badDF) != 0:
					self.levelGrp.button(3).setChecked(True)
					self._current_strip = True
		self._drawFig()

	def _pltPrevEvent(self):
		"""
		Plot previous events
		"""
		_j = self.evecb.currentIndex()
		for _i in range(len(self._eqlist) - 1):
			prevEvent = next(self._eventCycle)
		self._eventInfo(prevEvent)
		self._current_strip = False
		_id = self._current_event.ID
		if len(self._stripDF) != 0:
			existDF = self._stripDF[self._stripDF.ID == _id]
		else:
			existDF = pd.DataFrame()
		if len(existDF) != 0:
			level = existDF.Level.iloc[0]
			ind = self.eve_type.index(level)
			self.levelGrp.button(ind).setChecked(True)
			self._current_strip = True
		else:
			if len(self._badDF) != 0:
				_badDF = self._badDF[self._badDF.ID == _id]
				if len(_badDF) != 0:
					self.levelGrp.button(3).setChecked(True)
					self._current_strip = True
		if _j == 0:
			_n = len(self.evecb) - 1
			self.evecb.setCurrentIndex(_n)
		else:
			self.evecb.setCurrentIndex(_j-1)
		if self._btype == 'strip':
			self._btype = 'beam'
			self.sbcb.setCurrentIndex(0)
		self._drawFig()

	def _pltNextEvent(self):
		_id = self._current_event.ID
		level = self.eve_type[self.levelGrp.checkedId()]
		if level == 'D':
			self._current_strip = True
			self._setCodaStrip()
		else:
			# if len(self._stripDF) != 0:
			# 	existDF = self._stripDF[(self._stripDF.ID == _id)]
			# else:
			# 	existDF = pd.DataFrame()
			# if len(existDF) == 0:
			if not self._current_strip:
				choice = QMessageBox.question(self, 'Stripping?',
							"Haven't stripping yet, want to do it?",
							QMessageBox.Yes | QMessageBox.No)
				if choice is QMessageBox.Yes:
					self._current_strip = True
					self._appStrip()
					return
		self._eventInfo(next(self._eventCycle))
		self._current_strip = False
		_id = self._current_event.ID
		if len(self._stripDF) != 0:
			existDF = self._stripDF[self._stripDF.ID == _id]
		else:
			existDF = pd.DataFrame()
		if len(existDF) != 0:
			level = existDF.Level.iloc[0]
			ind = self.eve_type.index(level)
			self.levelGrp.button(ind).setChecked(True)
			self._current_strip = True
		else:
			if len(self._badDF) != 0:
				_badDF = self._badDF[self._badDF.ID == _id]
				if len(_badDF) != 0:
					self.levelGrp.button(3).setChecked(True)
					self._current_strip = True
		_i = self.evecb.currentIndex()
		if _i == len(self.evecb) - 1:
			self.evecb.setCurrentIndex(0)
		else:
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
		self.beamphase = event.beamphase
		self._current_id = eqid
		if not hasattr(event, 'beam'):
			return
		self._current_beam = event.beam
		filts = {}
		for tr in self._current_beam:
			filts[tr.stats.station] = tr.stats.channel
		self._current_filts = filts
		self._current_ID = event.ID
		self._current_dis = event.dis
		self._current_p = event.rayp
		self._current_bb = event.bb
		self._current_bakAz = event.baz
		self._current_delta = event.delta
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
		else:
			existDF = pd.DataFrame()
		if len(self._badDF) !=0:
			_badDF = self._badDF[self._badDF.ID == self._current_event.ID]
		else:
			_badDF = pd.DataFrame()
		if len(existDF) !=0:
			choice = QMessageBox.question(self, 'Replace stripping',
							"Do you want to replace existed stripping?",
							QMessageBox.Yes | QMessageBox.No)
			if choice == QMessageBox.Yes:
				index = existDF.index
				self._stripDF.drop(index,axis=0,inplace=True)
				self._stripDF.reset_index(inplace=True, drop=True)
			else:
				return
		if len(_badDF) != 0:
			choice = QMessageBox.question(self, 'Bad Event',
						 "Want to replace it?",
						 QMessageBox.Yes | QMessageBox.No)
			if choice == QMessageBox.Yes:
				index = _badDF.index
				self._badDF.drop(index,axis=0,inplace=True)
				self._badDF.reset_index(inplace=True, drop=True)
			else:
				return
		level = self.eve_type[self.levelGrp.checkedId()]
		ID = event.ID
		lat = event.lat
		lon = event.lon
		dep = event.dep
		mw = event.mw
		dis = event.dis
		bb = event.bb
		bakAzi = event.baz
		delta = event.delta

		if level =='D':
			newRow = {'ID': ID, 'lat':lat,
					  'lon':lon,'dep':dep,
					  'Mw':mw,'Del':dis,
					  'BB':bb,'bakAzi':bakAzi,'Level':'D'}
			msg = ('%s is Bad Event'%self._current_ID)
			gpar.log(__name__, msg, level='info', pri=True)
			self._badDF = self._badDF.append(newRow, ignore_index=True)
		else:
			if self._method == 'all':
				newRow = {'ID': ID, 'lat':lat,
						  'lon':lon,'dep':dep,
						  'Mw':mw,'Del':dis,
						  'BB':bb,'bakAzi':bakAzi,
						  'winName':win['name'], 'win':win,
						  'Level':level, 'delta': delta,
						  'codaResSt':event.codaResSt,
						  'codaSt':event.codaSt,
						  'crms':event.codaMod,
						  'twoResSt':event.twoResSt,
						  'twoSt':event.twoSt,
						  'trms':event.twoMod}
				self._stripDF = self._stripDF.append(newRow, ignore_index=True)
			elif self._method == 'coda':
				newRow = {'ID': ID, 'lat':lat,
						  'lon':lon,'dep':dep,
						  'Mw':mw,'Del':dis,
						  'winName':win['name'], 'win':win,
						  'BB':bb,'bakAzi':bakAzi,
						  'Level':level, 'delta': delta,
						  'codaResSt':event.codaResSt,
						  'codaSt':event.codaSt,
						  'crms':event.codaMod,
						  }
				self._stripDF = self._stripDF.append(newRow, ignore_index=True)
			elif self._method == 'twoline':
				newRow = {'ID': ID, 'lat':lat,
						  'lon':lon,'dep':dep,
						  'Mw':mw,'Del':dis,
						  'BB':bb,'bakAzi':bakAzi,
						  'Level':level, 'delta': delta,
						  'winName':win['name'], 'win':win,
						  'twoResSt':event.twoResSt,
						  'twoSt':event.twoSt,
						  'trms':event.twoMod}
				self._stripDF = self._stripDF.append(newRow, ignore_index=True)


	def _drawFig(self):
		self.fig.clear()
		a = u"\u00b0"
		if self._btype == 'beam':
			num_plots = len(self._current_beam)
			for _i, tr in enumerate(self._current_beam):
				ax = self.fig.add_subplot(num_plots, 1, _i+1)
				if hasattr(tr.stats, 'channel'):
					label = tr.stats.channel
				else:
					label=None
				time = np.arange(tr.stats.npts) * tr.stats.delta + tr.stats.sac.b
				ax.plot(time, tr.data, 'k', label=label)
				if not hasattr(self._current_event, 'arrivals'):
					self._current_event.getArrival()
				arrival = self._current_event.arrivals[self.beamphase]['TT']# - self._current_event.time
				ax.vlines(arrival, ax.get_ylim()[0],ax.get_ylim()[1],'r', label=self.beamphase)
				if self.ttbtn.isChecked():
					_arr = self._current_event.arrivals
					# del _arr[self.beamphase]
					for name, tt in _arr.items():
						if name is self.beamphase:
							continue
						ax.vlines(tt['TT'], ax.get_ylim()[0],ax.get_ylim()[1],'b',label=name)
				ax.legend()
				if _i == 0:
					ax.set_xlabel('Seconds')
				self.fig.suptitle('%s - %s\nDep:%s  Distance: %s%s'
					%(self._current_event.ID, self._btype, self._current_event.dep, self._current_event.dis, a))
		elif self._btype == 'slide':
			self.fig.suptitle('%s - %s\nDep:%s  Distance: %s%s'
				%(self._current_event.ID, self._btype, self._current_event.dep, self._current_event.dis, a))
			nfilts = len(self._current_slide.keys())
			ax = self.fig.subplots(4, nfilts, sharex='col', sharey='row')
			ax = ax.reshape(4,nfilts)
			for ind, (name,st) in enumerate(self._current_slide.items()):
				for _i, tr in enumerate(st):
					if hasattr(tr.stats, 'channel'):
						label = tr.stats.channel
					else:
						label=None
					time = np.arange(tr.stats.npts) * tr.stats.delta + tr.stats.sac.b
					ax[_i,ind].plot(time, tr.data, 'k', label=None)
					ax[_i, ind].set_xlim([np.min(time), np.max(time)])
					if label == 'Amplitude':
						peak = np.max(tr.data) + 1
						ax[_i,ind].set_ylim([-1, peak])
					elif label == 'Slowness':
						ax[_i,ind].set_ylim([0, 15])
						rp = self._current_event.rayp
						ax[_i, ind].hlines(rp, np.min(time), np.max(time), 'r', 'dashed')
					elif label == 'Back Azimuth':
						ax[_i, ind].set_ylim([0,360])
					elif label == 'coherence':
						ax[_i, ind].set_ylim([0,1])
					if not hasattr(self._current_event, 'arrivals'):
						self._current_event.getArrival()
					arrival = self._current_event.arrivals[self.beamphase]['TT']# - self._current_event.time
					ax[_i,ind].vlines(arrival, ax[_i,ind].get_ylim()[0],ax[_i,ind].get_ylim()[1],'r',label=self.beamphase)
					
					if self.ttbtn.isChecked():
						_arr = self._current_event.arrivals
						# del _arr[self.beamphase]
						for pname, tt in _arr.items():
							if pname is self.beamphase:
								continue
							ax[_i,ind].vlines(tt['TT'], ax[_i,ind].get_ylim()[0],ax[_i,ind].get_ylim()[1],'b',label=pname)
					ax[_i,ind].legend()
					# ax[_i,ind].set_aspect(aspect=0.3)
					if _i == 3:
						ax[_i,ind].set_xlabel('Seconds')
					if _i == 0:
						ax[_i,ind].set_title(name)
					if ind == 0:
						ax[_i,ind].set_ylabel(label)
		elif self._btype == 'vespetrum':
			num = len(self._current_energy)
			extent=[np.min(self._current_time),np.max(self._current_time),np.min(self._current_K),np.max(self._current_K)]
			vmin = float(self.ampmin.cleanText())
			vmax = float(self.ampmax.cleanText())
			if not hasattr(self._current_event, 'arrivals'):
				self._current_event.getArrival()
			for ind, _row in self._current_energy.iterrows():
				# abspow = _row.POWER
				name = _row.FILT
				if self.vepcb.currentText() == 'log10':
					abspow = np.log10(np.abs(_row.POWER))
				elif self.vepcb.currentText() == 'log':
					abspow = np.log(np.abs(_row.POWER))
				elif self.vepcb.currentText() == 'sqrt':
					abspow = np.sqrt(np.abs(_row.POWER))
				else:
					abspow = np.abs(_row.POWER)
				ax = self.fig.add_subplot(1, num, ind+1)
				ax.imshow(abspow, extent=extent, aspect='auto', cmap='Reds', vmin=vmin, vmax=vmax, origin='lower')
				arrival = self._current_event.arrivals[self.beamphase]['TT']
				ax.vlines(arrival, ax.get_ylim()[0],ax.get_ylim()[1],'k',label=self.beamphase)
				rp = self._current_event.rayp
				ax.hlines(rp, ax.get_xlim()[0],ax.get_xlim()[1], 'b')
				ax.hlines(-rp, ax.get_xlim()[0],ax.get_xlim()[1], 'b')
				if self.ttbtn.isChecked():
					_arr = self._current_event.arrivals
						# del _arr[self.beamphase]
					for name, tt in _arr.items():
						if name is self.beamphase:
							continue
						ax.vlines(tt['TT'], ax.get_ylim()[0],ax.get_ylim()[1],'b',label=name)
				ax.legend()
				ax.set_xlabel('Seconds')
				if ind == 0:
					ax.set_ylabel(self._current_type)
				ax.set_title(name)
			if self._current_type == 'slowness':
				title = '%s - %s\nSlant Stack at a Backazimuth of %.1f %sN\nDep:%s  Distance: %s%s' \
						%(self._btype, self._current_ID, self._current_event.baz,a,
							self._current_event.dep, self._current_event.dis, a)
			elif self._current_type == 'theta':
				title = '%s - %s\nSlant Stack at a slowness of %.2f s/deg\nDep:%s  Distance: %s%s' \
						%(self._btype, self._current_ID, self._current_event.rayp,
							self._current_event.dep, self._current_event.dis, a)
			self.fig.suptitle(title)
		elif self._btype == 'strip':
			_i = self.wincb.currentIndex()
			win = self.trinWin[_i]
			if len(self._stripDF) != 0:
				existDF = self._stripDF[(self._stripDF.ID == self._current_event.ID) & (self._stripDF.winName == win['name'])]
			else:
				existDF = pd.DataFrame()
			if len(self._badDF) != 0:
				_badDF = self._badDF[self._badDF.ID == self._current_event.ID]
			else:
				_badDF = pd.DataFrame()
			if len(existDF) == 0 and len(_badDF) == 0:
				choice = QMessageBox.question(self, 'Stripping?',
						"Haven't stripping yet, want to do it?",
						QMessageBox.Yes | QMessageBox.No)
				if choice == QMessageBox.Yes:
					self._appStrip()
				else:
					self._btype = 'beam'
					self.sbcb.setCurrentIndex(0)
					self._updatePlot()
			elif len(_badDF) != 0:
				choice = QMessageBox.question(self, 'Bad event!',
						 "Want to reevalua it?",
						 QMessageBox.Yes | QMessageBox.No)
				if choice == QMessageBox.Yes:
					index = _badDF.index
					self._badDF.drop(index,axis=0,inplace=True)
					self._badDF.reset_index(inplace=True, drop=True)
					self.sbcb.setCurrentIndex(0)
					self._updatePlot()
				else:
					self.sbcb.setCurrentIndex(0)
					self._updatePlot()
			elif len(existDF) != 0:
				trinwin = existDF.win.iloc[0]
				stime = trinwin['stime']
				etime = trinwin['etime']
				delta = self._current_beam[0].stats.delta
				# npts = int((etime - stime)/delta) + 1
				npts = int((etime - stime)/delta)
				# time = np.linspace(stime, etime, npts)
				time = stime + np.arange(npts) * delta

				sind = int(stime / delta)
				# eind = int(etime / delta)
				if self._method == 'all':
					codamode = existDF.crms.iloc[0]
					twomode = existDF.trms.iloc[0]
					nfilter = len(codamode)
					codaSt = existDF.codaSt.iloc[0]
					twoSt = existDF.twoSt.iloc[0]
					cRes = existDF.codaResSt.iloc[0]
					tRes = existDF.twoResSt.iloc[0]
					timeR = np.arange(cRes[0].stats.npts)*cRes[0].stats.delta - trinwin['noise']
					data_time = np.arange(twoSt[0].stats.npts) * delta + (twoSt[0].stats.starttime - self._current_beam[0].stats.starttime)
					ax = self.fig.subplots(2, nfilter)
					if nfilter == 1:
						ax = ax.reshape(2, nfilter)
					for ind in range(nfilter):
						data = np.abs(scipy.signal.hilbert(self._current_beam[ind].data))[sind:sind+npts]
						ax[0,ind].plot(time,np.log10(data),'k', label='beam')
						data_coda = codaSt[ind].data
						ax[0,ind].plot(time,np.log10(data_coda),'r', label='coda')
						data_two = twoSt[ind].data
						ax[0,ind].plot(data_time, data_two,'b', label='twoline')
						ax[0,ind].set_xlim([stime, etime])
						ax[0,ind].set_ylim([-1, 5])
						ax[0,ind].set_xlabel('Seconds')
						ax[0,ind].legend()
						label_c = "Coda: Mean RMS = %s"%(codamode['RMS'].iloc[ind])
						label_t = "Twoline: Mean RMS = %s"%(twomode['RMS'].iloc[ind])
						ax[1,ind].plot(timeR,cRes[ind].data, 'r', label=label_c)
						ax[1,ind].plot(timeR, tRes[ind].data, 'b',label=label_t)
						ax[1,ind].legend()
						ax[1,ind].set_xlabel('Seconds')
						ax[1,ind].set_xlim([-trinwin['noise']/2, trinwin['noise']/2+trinwin['coda']])
						ax[0,ind].set_title('Filter: %s'%twomode['FILT'].iloc[ind])
						if ind is 0:
							ax[0,ind].set_ylabel('log10(Amp)')
							ax[1,ind].set_ylabel('Amp')

				elif self._method == 'coda':
					codamode = existDF.crms.iloc[0]
					nfilter = len(codamode)
					codaSt = existDF.codaSt.iloc[0]
					cRes = existDF.codaResSt.iloc[0]
					timeR = np.arange(cRes[0].stats.npts)*cRes[0].stats.delta - trinwin['noise']
					ax = self.fig.subplots(2, nfilter)
					for ind in range(nfilter):
						data = np.abs(scipy.signal.hilbert(self._current_beam[ind].data))[sind:sind+npts]
						ax[0,ind].plot(time,np.log10(data),'k', label='beam')
						data_coda = codaSt[ind].data
						ax[0,ind].plot(time,np.log10(data_coda),'r', label='coda')
						ax[0,ind].set_xlim([stime, etime])
						ax[0,ind].set_ylim([-1, 5])
						ax[0,ind].set_xlabel('Seconds')
						ax[0,ind].legend()
						label_c = "Coda: Mean RMS = %s"%(codamode['RMS'].iloc[ind])
						ax[1,ind].plot(timeR,cRes[ind].data, 'r', label=label_c)
						ax[1,ind].legend()
						ax[1,ind].set_xlabel('Seconds')
						ax[1,ind].set_xlim([-trinwin['noise']/2, trinwin['noise']/2+trinwin['coda']])
						ax[0,ind].set_title('Filter: %s'%codamode['FILT'].iloc[ind])
						if ind is 0:
							ax[0,ind].set_ylabel('log10(Amp)')
							ax[1,ind].set_ylabel('Amp')
				elif self._method == 'twoline':
					twomode = existDF.trms.iloc[0]
					nfilter = len(twomode)
					twoSt = existDF.twoSt.iloc[0]
					tRes = existDF.twoResSt.iloc[0]
					timeR = np.arange(tRes[0].stats.npts)*tRes[0].stats.delta - trinwin['noise']
					data_time = np.arange(twoSt[0].stats.npts) * delta + (twoSt[0].stats.starttime - self._current_beam[0].stats.starttime)
					ax = self.fig.subplots(2, nfilter)
					for ind in range(nfilter):
						data = np.abs(scipy.signal.hilbert(self._current_beam[ind].data))[sind:sind+npts]
						ax[0,ind].plot(time,np.log10(data),'k', label='beam')
						data_two = twoSt[ind].data
						ax[0,ind].plot(data_time, data_two,'b', label='twoline')
						ax[0,ind].set_xlim([stime, etime])
						ax[0,ind].set_ylim([-1, 5])
						ax[0,ind].set_xlabel('Seconds')
						ax[0,ind].legend()
						label_t = "Twoline: Mean RMS = %s"%(twomode['RMS'].iloc[ind])
						ax[1,ind].plot(timeR, tRes[ind].data, 'b',label=label_t)
						ax[1,ind].legend()
						ax[1,ind].set_xlabel('Seconds')
						ax[1,ind].set_xlim([-trinwin['noise']/2, trinwin['noise']/2+trinwin['coda']])
						ax[0,ind].set_title('Filter: %s'%twomode['FILT'].iloc[ind])
						if ind is 0:
							ax[0,ind].set_ylabel('log10(Amp)')
							ax[1,ind].set_ylabel('Amp')
				self.fig.suptitle('Coda Strip for %s using %s method in win %s\nDep:%s  Distance: %s%s'
					%(self._current_event.ID, self._method, trinwin['name'],
						self._current_event.dep, self._current_event.dis, a))


		self._canvasDraw()

	#def _plotTT(self):
	#	if self.ttbtn.isChecked() is False:


	def _updatePlot(self):

		self._activeAmp()
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
		self.savefile,_ = QFileDialog.getSaveFileName(self,
										'Save as', os.getcwd(), files_types)
		self.savefile = str(self.savefile)
		if os.path.splitext(self.savefile)[1].lower() == '.pkl':
			self._savePickle(self.savefile)
		elif os.path.splitext(self.savefile)[1].lower() == '.csv':
			self._saveCSV(self.savefile)

	def _savePickle(self, filename):
		self._stripDF.to_pickle(filename)
		name = os.path.splitext(filename)
		badname = name[0]+'.D'+name[1]
		if len(self._badDF) != 0:
			self._badDF.to_pickle(badname)

	def _saveCSV(self, filename):
		_stripDF = self._stripDF
		_stripDF.drop(['codaSt','twoSt','twoResSt','codaResSt'])
		_stripDF.to_csv(filename,index=False,sep=',')
		if len(self._badDF) != 0:
			_badDF = self._badDF
			name = os.path.splitext(filename)
			badname = name[0] +'.D' +name[1]
			_badDF.to_csv(badname, index=False, sep=',')

	def _openFile(self):
		filename,_ = QFileDialog.getOpenFileName(self,'Load Pickle File',
													os.getcwd(), 'Pickle Format (*.pkl)', '20')
		if filename:
			filename = str(filename)
			self._stripDF = pd.read_pickle(filename)
			name = os.path.splitext(filename)
			badname = name[0]+'.D'+name[1]
			if os.path.exists(badname):
				self._badDF = pd.read_pickle(badname)
			self._pltEvent()
			self.savefile = str(filename)

	def _openArray(self):
		filename,_ = QFileDialog.getOpenFileName(self, 'Load array',
											os.getcwd(), 'Pickle Format (*.pkl)', '20')
		if filename:
			filename = str(filename)
			ar = util.loadArray(filename)
			self._refreshArray(ar)

	def _refreshArray(self, ar):

		self.array = ar
		self._plt_drag = None

		# init events in the array
		self._events = ar.events #defines list self._events
		self.savefile = ar.name+'.strip.pkl'
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
		self._eventInfo(next(self._eventCycle))
		self.setWindowTitle('Array Analysis: %s'%self.array.name)
		self.evecb.clear()
		for eve in self._eqlist:
			self.evecb.addItem(eve)
		self._drawFig()

	def _savePlot(self):
		# path = os.getcwd()
		# path = os.path.join(path,self.array.name,self._current_event.ID)
		file_types = "Image Format (*.png *.pdf *.ps *.eps);; ALL (*)"
		filename,_ = QFileDialog.getSaveFileName(self, 'Save Plot',
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
			ask = QMessageBox.question(self, 'Save stripping?',
				'Do you want to save your coda data?',
				QMessageBox.Save |
				QMessageBox.Discard |
				QMessageBox.Cancel, QMessageBox.Save)
			if ask == QMessageBox.Save:
				self._saveFileFormat()
				self.close()
			elif ask == QMessageBox.Cancel:
				event.ignore()


	class defWindow(QDialog):
		def __init__(self, parent=None, windowvalue=None):
			"""
			Coda strip window dialog
			"""

			QDialog.__init__(self, parent)
			self.setWindowTitle('Create new coda strip window')
			self.noisewin = QDoubleSpinBox(decimals=1, maximum=400, minimum=20, singleStep=10, value=10)
			self.codawin = QDoubleSpinBox(decimals=1, maximum=400, minimum=20, singleStep=10, value=10)
			self.stime = QDoubleSpinBox(decimals=1, maximum=600, minimum=0, singleStep=50, value=50)
			self.etime = QDoubleSpinBox(decimals=1, maximum=2400, minimum=1600, singleStep=50, value=50)
			self.smooth = QDoubleSpinBox(decimals=1, maximum=20, minimum=1, singleStep=1, value=4)

			self.winName = QLineEdit('Window Name')
			self.winName.selectAll()

			self.model = QLineEdit('ak135')
			self.model.selectAll()

			grid = QGridLayout()
			grid.addWidget(QLabel('Window Name'), 0, 0)
			grid.addWidget(self.winName, 0, 1)
			grid.addWidget(QLabel('Noise Win.'), 1, 0)
			grid.addWidget(self.noisewin, 1, 1)
			grid.addWidget(QLabel('Coda Win.'), 2, 0)
			grid.addWidget(self.codawin, 2, 1)
			grid.addWidget(QLabel('Start Time.'), 3, 0)
			grid.addWidget(self.stime, 3, 1)
			grid.addWidget(QLabel('End Time.'), 4, 0)
			grid.addWidget(self.etime, 4, 1)
			grid.addWidget(QLabel('Smooth.'), 5, 0)
			grid.addWidget(self.smooth, 5, 1)
			grid.addWidget(QLabel('Model.'), 6, 0)
			grid.addWidget(self.model, 6, 1)
			grid.setVerticalSpacing(10)

			btnbox = QDialogButtonBox(QDialogButtonBox.Ok |
											QDialogButtonBox.Cancel)

			btnbox.accepted.connect(self.accept)
			btnbox.rejected.connect(self.reject)

			layout = QVBoxLayout()
			layout.addWidget(QLabel('Define noise window and coda window for stripping'))
			layout.addLayout(grid)
			layout.addWidget(btnbox)

			if windowvalue is not None:
				self.winName.setText(windowvalue['name'])
				self.noisewin.setValue(windowvalue['noise'])
				self.codawin.setValue(windowvalue['coda'])
				self.stime.setValue(windowvalue['stime'])
				self.etime.setValue(windowvalue['etime'])
				self.smooth.setValue(windowvalue['smooth'])
				self.model.setText(windowvalue['model'])
			self.setLayout(layout)
			self.setSizeGripEnabled(False)

		def getValues(self):
			"""
			Return window dialog values as a dictionary
			"""

			return dict(name=str(self.winName.text()),
						noise=float(self.noisewin.cleanText()),
						coda=float(self.codawin.cleanText()),
						stime=float(self.stime.cleanText()),
						etime=float(self.etime.cleanText()),
						smooth=float(self.smooth.cleanText()),
						model=str(self.model.text()))

# class for event stacking in arrays
class stackArray(QtWidgets.QMainWindow):

	def __init__(self, arraylist=None, parent=None, ap=None):
		if ap is None:
			self.qApp = QApplication(sys.argv)
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
						   'cancel': 'c',
						   'accept': 'a',
						   'stack': 's'}

		self._list = arlist
		self._initArrayList()
		self._arrayCycle = cycle(self._namelist)
		self.dis = [{'name':'All-Dis',
					 'mindis': 50.0,
					 'maxdis': 75.0,
					 'step':25.0,
					 'overlap':0,
					 'write':False}]
		self._arrayInfo(next(self._arrayCycle))
		self._initReg()
		QMainWindow.__init__(self)
		self.setupUI()

	def setupUI(self):
		self.main_widget = QWidget(self)
		self._initMenu()
		self._createStatusBar()
		self._initPlots()

		l = QVBoxLayout(self.main_widget)

		l.addLayout(self.btnbar)
		l.addLayout(self.btnbar2)
		l.addWidget(self.canvas)
		self.setCentralWidget(self.main_widget)
		self.setGeometry(300, 300, 1200, 800)
		self.setWindowTitle('Array Stack')
		self.show()

	def _killLayout():
		pass
	def _initPlots(self):
		self.fig = Figure(dpi=100, constrained_layout=True)
		self.canvas = FigureCanvas(self.fig)
		self.canvas.setFocusPolicy(PyQt5.QtCore.Qt.StrongFocus)
		self._drawFig()

		self.fig.canvas.mpl_connect('key_press_event', self._selectRegOnPress)

	def _initMenu(self):
		# Next and Prev array
		nxt = QtWidgets.QPushButton('Next >>',
			shortcut=self._shortcuts['arr_next'], parent=self.main_widget)
		nxt.clicked.connect(self._pltNextArray)
		nxt.setToolTip('shortcut <b>n</d>')
		nxt.setMaximumWidth(150)
		prv = QtWidgets.QPushButton('Prev >>',
			shortcut=self._shortcuts['arr_prev'], parent=self.main_widget)
		prv.clicked.connect(self._pltPrevArray)
		prv.setToolTip('shortcut <b>p</d>')
		prv.setMaximumWidth(150)

		# Array drop-down
		self.arcb = QComboBox(self)
		for arr in self._namelist:
			self.arcb.addItem(arr)
		self.arcb.activated.connect(self._pltArray)
		self.arcb.setMaximumWidth(1000)
		self.arcb.setMinimumWidth(80)
		# filter selection
		self.filtcb = QComboBox(self)
		self.filtcb.addItem('all')
		for filt in self._current_filter:
			self.filtcb.addItem(filt)
		self.filtcb.activated.connect(self._drawStack)
		self.filtcb.setMaximumWidth(1000)
		self.filtcb.setMinimumWidth(80)

		# Select region

		# Stacking earthquakes in array
		self.stbtn = QtWidgets.QPushButton('Stack',
					shortcut=self._shortcuts['stack'], parent=self.main_widget)
		self.stbtn.setCheckable(True)
		self.stbtn.setStyleSheet('QPushButton:checked {background-color: lightgreen;}')
		self.stbtn.setToolTip('shortcut <b>s</b>')
		self.stbtn.clicked.connect(self._drawStack)
		self.nbtn = QtWidgets.QPushButton('Norm',
					parent=self.main_widget)
		self.nbtn.setCheckable(True)
		self.nbtn.setStyleSheet('QPushButton:checked {background-color: lightgreen;}')
		self.nbtn.clicked.connect(self._drawStack)
		self.ebtn = QtWidgets.QPushButton('ERR',
					parent=self.main_widget)
		self.ebtn.setCheckable(True)
		self.ebtn.setStyleSheet('QPushButton:checked {background-color: lightgreen;}')
		self.ebtn.clicked.connect(self._drawStack)
		# Select distance
		self.discb = QComboBox(self)
		self.discb.activated.connect(self._changeStack)
		self._updateWindow()

		disEdit = QtWidgets.QPushButton('Edit')
		disEdit.resize(disEdit.sizeHint())
		disEdit.clicked.connect(self._editDis)

		disDelt = QtWidgets.QPushButton('Delete')
		disDelt.resize(disEdit.sizeHint())
		disDelt.clicked.connect(self._deleteDis)
		# Select region
		self.regcb = QComboBox(self)
		self.regcb.activated.connect(self._changeRegion)
		self._updateRegion()
		regEdit = QtWidgets.QPushButton('Edit')
		regEdit.resize(regEdit.sizeHint())
		regEdit.clicked.connect(self._editRegion)
		regDelt = QtWidgets.QPushButton('Delete')
		regDelt.resize(regDelt.sizeHint())
		regDelt.clicked.connect(self._deleteRegion)
		#button to plot all regin in one plot
		self.allbtn = QPushButton('All Region')
		self.allbtn.setCheckable(True)
		self.allbtn.setStyleSheet('QPushButton:checked {background-color: lightgreen;}')
		self.allbtn.clicked.connect(self._stackAll)
		#reset region button
		self.rsbtn = QtWidgets.QPushButton('Reset')
		self.rsbtn.clicked.connect(self._resetReg)

		self.btnbar = QHBoxLayout()
		self.btnbar.addWidget(prv)
		self.btnbar.addWidget(nxt)
		self.btnbar.addWidget(QLabel('Array'))
		self.btnbar.addWidget(self.arcb)

		vline = QFrame()
		vline.setFrameStyle(QFrame.VLine | QFrame.Raised)
		self.btnbar.addWidget(vline)
		self.btnbar.addWidget(self.stbtn)
		self.btnbar.addWidget(QLabel('Step'))
		self.btnbar.addWidget(self.discb)
		self.btnbar.addWidget(disEdit)
		self.btnbar.addWidget(disDelt)
		self.btnbar.addStretch(1)

		self.btnbar2 = QHBoxLayout()
		self.btnbar2.addWidget(QLabel('Region'))
		self.btnbar2.addWidget(self.regcb)
		self.btnbar2.addWidget(regEdit)
		self.btnbar2.addWidget(regDelt)
		self.btnbar2.addWidget(self.allbtn)
		self.btnbar2.addWidget(vline)
		self.btnbar2.addWidget(self.rsbtn)
		self.btnbar2.addWidget(vline)
		self.btnbar2.addWidget(self.nbtn)
		self.btnbar2.addWidget(vline)
		self.btnbar2.addWidget(self.ebtn)
		self.btnbar2.addWidget(vline)
		self.btnbar2.addWidget(QLabel('Filter'))
		self.btnbar2.addWidget(self.filtcb)
		self.btnbar2.addStretch(1)

		#Menubar
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(QtGui.QIcon().fromTheme('document-save'),
						   'Save Reg', self._saveFile)

	def _saveFile(self):
		if len(self._region) >= 2:
			_region = self._region[1:]
			for _reg in _region:
				name = _reg['name']
				array = self._current_array['name']
				_df = self.regDf[name]
				savename = os.path.join(array,name+'.pkl')
				_df.to_pickle(savename)

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

	def _initReg(self):
		self._region = [{'name':'global',
						 'latmin':-90.0, 'latmax':90.0,
						 'lonmin': -180.0, 'lonmax': 180.0}]
		self.stackSt = {}
		self.stdSt = {}
		win = self._current_array_df['win'].iloc[0]
		window = [win['noise'], win['coda']]
		self.regDf = {'global':self._current_array_df}
		_stackSt, _stdSt = stackTR(self._current_array_df,
								  pklname=None,win=window,
								  mindis=self.dis[0]['mindis'],
								  maxdis=self.dis[0]['maxdis'],
								  step=self.dis[0]['step'],
								  overlap=self.dis[0]['overlap'],
								  write=self.dis[0]['write'])
		self.stackSt = {'global': _stackSt}
		self.stdSt = {'global': _stdSt}


	def _arrayInfo(self, name):
		ardf = self._arlist[self._arlist.NAME == name].iloc[0]
		array = {'name':ardf.NAME,'lat':ardf.LAT,'lon':ardf.LON}
		self._current_array = array
		self._current_array_df = ardf.DF
		self._current_filter = self._current_array_df.iloc[0].crms.FILT
		# self._initReg()

	def _createStatusBar(self):
		sb = QStatusBar()
		sb.setFixedHeight(18)
		self.setStatusBar(sb)
		self.statusBar().showMessage('Ready')

	def _drawFig(self):
		self.fig.clear()
		# self.fig.add_subplot(1, 1, 1)
		# _ax = self.fig.get_axes()
		_ax = self.fig.add_subplot(1, 1, 1)

		m = Basemap(projection='cyl', lon_0=0, lat_0=0.0,
					area_thresh=10000,ax=_ax)
		x0, y0 = m(self._current_array['lon'], self._current_array['lat'])

		m.drawcoastlines(ax=_ax)
		m.drawmapboundary(ax=_ax)
		m.fillcontinents(color='lightgray',lake_color='white',ax=_ax)
		parallels = np.arange(-90.0, 90.0, 60)
		# labels = [left, right, top, bottom]
		m.drawparallels(parallels,labels=[True, False, False, False],ax=_ax)
		meridians = np.arange(-180.0, 180.0, 60.0)
		m.drawmeridians(meridians,labels=[False, False, False, True],ax=_ax)
		m.scatter(x0, y0, marker='*',c='r',s=100,alpha=0.7,ax=_ax,zorder=10)
		if self.allbtn.isChecked() and len(self._region) < 2:
			x, y = m(self._current_array_df.lon.tolist(), self._current_array_df.lat.tolist())
			m.scatter(x, y, marker='o', c='blue', s=50, alpha=0.7,ax=_ax,zorder=10)
		elif self.allbtn.isChecked() and len(self._region) >= 2:
			_region = self._region[1:,]
			_current_df = self._current_array_df
			_rest_df = _current_df.copy()
			for _j, _reg in enumerate(_region):
				_name = _region['name']
				_df = self.regDf[_name]
				_rest_df = _rest_df[~((_rest_df['lat']>_reg['latmin']) &
						  			(_rest_df['lat']<_reg['latmax']) &
						  			(_rest_df['lon']>_reg['lonmin']) &
						  			(_rest_df['lon']<_reg['lonmax']))]
				x, y = m(_df.lon.tolist(), _df.lat.tolist())
				m.scatter(x, y, marker='o', c=color[_i], s=50, alpha=0.7,ax=_ax,zorder=10)
			x, y = m(_rest_df.lon.tolist(), _rest_df.lat.tolist())
			m.scatter(x, y, marker='o', c='k', s=50, alpha=0.7,ax=_ax,zorder=10)
		elif self.allbtn.isChecked() is False:
			_i = self.regcb.currentIndex()
			if _i == 0:
				x, y = m(self._current_array_df.lon.tolist(), self._current_array_df.lat.tolist())
				m.scatter(x, y, marker='o', c='blue', s=50, alpha=0.7,ax=_ax,zorder=10)
				self._pltRectangle()
			else:
				_reg = self._region[_i]
				_name = _reg['name']
				_df = self.regDf[_name]
				x, y = m(_df.lon.tolist(), _df.lat.tolist())
				m.scatter(x, y, marker='o', c=color[_i-1], s=50, alpha=0.7,ax=_ax,zorder=10)


		self.fig.suptitle('Earthquakes in array %s'%self._current_array['name'])
		self._canvasDraw()


	def _canvasDraw(self):

		for _i, _ax in enumerate(self.fig.get_axes()):
			_ax.set_xticklabels(_ax.get_xticks())
		self.fig.canvas.draw()
		self.canvas.setFocus()

	def _pltArray(self):
		_i = self.arcb.currentIndex()
		while next(self._arrayCycle) != self._namelist[_i]:
			pass
		self._arrayInfo(self._namelist[_i])
		self.filtcb.clear()
		self.filtcb.addItem('all')
		for filt in self._current_filter:
			self.filtcb.addItem(filt)
		self._initReg()
		self._drawFig()
		# self._resetReg()

	def _pltPrevArray(self):
		_j = self.arcb.currentIndex()
		for _i in range(len(self._namelist) - 1):
			prevarray = next(self._arrayCycle)
		self._arrayInfo(prevarray)
		self.filtcb.clear()
		self.filtcb.addItem('all')
		for filt in self._current_filter:
			self.filtcb.addItem(filt)
		self.arcb.setCurrentIndex(_j-1)
		self._initReg()
		self._drawFig()
		# self._resetReg()

	def _pltNextArray(self):
		_i = self.arcb.currentIndex()
		self._arrayInfo(next(self._arrayCycle))
		self.filtcb.clear()
		self.filtcb.addItem('all')
		for filt in self._current_filter:
			self.filtcb.addItem(filt)
		self.arcb.setCurrentIndex(_i+1)
		# self._resetReg()
		self._initReg()
		self._drawFig()

	def _calStack(self):
		_i = self.discb.currentIndex()
		self._arrayInfo(self._current_array['name'])
		savefile = None

		win = self._current_array_df['win'].iloc[0]
		window = [win['noise'], win['coda']]

		_j = self.regcb.currentIndex()
		_reg = self._region[_j]
		if self.dis[_i]['write']:
			savefile = _reg['name'] + '.'+self.dis[_i]['name'] + '.sac'
		_current_df = self._current_array_df
		_df = _current_df[(_current_df['lat']>_reg['latmin']) &
						  (_current_df['lat']<_reg['latmax']) &
						  (_current_df['lon']>_reg['lonmin']) &
						  (_current_df['lon']<_reg['lonmax'])]
		_df.reset_index(inplace=True,drop=True)
		self.regDf[_reg['name']] = _df

		_stackSt, _stdSt = stackTR(_df,
								  pklname=savefile,win=window,
								  mindis=self.dis[_i]['mindis'],
								  maxdis=self.dis[_i]['maxdis'],
								  step=self.dis[_i]['step'],
								  overlap=self.dis[_i]['overlap'],
								  write=self.dis[_i]['write'])
		self.stackSt[_reg['name']] = _stackSt
		self.stdSt[_reg['name']] = _stdSt
		# if self.stbtn.isChecked():
		# 	self._drawStack()
		# else:
		# 	self._drawFig()

	def _drawStack(self):
		self.fig.clear()
		if self.stbtn.isChecked() is False:
			self._drawFig()
			return
		# self.fig.add_subplot(121)
		_i = self.discb.currentIndex()
		this_dis = self.dis[_i]
		win = self._current_array_df['win'].iloc[0]
		window = [win['noise'], win['coda']]
		step_forward = this_dis['step'] * (1 - this_dis['overlap'])
		n = int((this_dis['maxdis'] - this_dis['mindis'])/step_forward)
		ftype = self.filtcb.currentText()
		if ftype == 'all':
			current_filter = self._current_filter
		else:
			current_filter = [ftype]
		n_filt = len(current_filter)
		if self.allbtn.isChecked() is False or len(self._region) < 2:
			_i = self.regcb.currentIndex()
			_current_df = self._current_array_df
			_name = self._region[_i]['name']
			_df = self.regDf[_name]
			_stackSt = self.stackSt[_name]
			_stdSt = self.stdSt[_name]
			#n = len(_stackSt)
			# gs = self.fig.add_gridspec(n,2)
			gs = gridspec.GridSpec(ncols=n_filt+1, nrows=n, figure=self.fig)
			_ax = self.fig.add_subplot(gs[:,0])
			m = Basemap(projection='cyl', lon_0=0, lat_0=0.0,
						area_thresh=10000,ax=_ax)
			x0, y0 = m(self._current_array['lon'], self._current_array['lat'])
			alon = _current_df[(~_current_df.lon.isin(_df.lon)) & (~_current_df.lat.isin(_df.lat))].lon.tolist()
			alat = _current_df[(~_current_df.lon.isin(_df.lon)) & (~_current_df.lat.isin(_df.lat))].lat.tolist()
			x, y = m(alon, alat)
			xt, yt = m(_df.lon.tolist(), _df.lat.tolist())
			m.drawcoastlines(ax=_ax)
			m.drawmapboundary(ax=_ax)
			m.fillcontinents(color='lightgray',lake_color='white',ax=_ax)
			parallels = np.arange(-90.0, 90.0, 60)
			m.drawparallels(parallels,ax=_ax)
			meridians = np.arange(-180.0, 180.0, 60.0)
			m.drawmeridians(meridians,ax=_ax)
			if _i == 0:
				c = 'blue'
			else:
				c = color[_i-1]
			m.scatter(x0, y0, marker='*',c='r',s=100,alpha=0.7,ax=_ax,zorder=10)
			m.scatter(x, y, marker='o', c='k', s=50, alpha=0.7,ax=_ax,zorder=10)
			m.scatter(xt, yt, marker='o', c='r', s=50, alpha=0.7,ax=_ax,zorder=10)

			# self.fig.add_subplot(122)
			delta = _stackSt[0].stats.delta
			npts = _stackSt[0].stats.npts
			time = np.arange(npts)*delta + _stackSt[0].stats.sac.b
			for ind, f in enumerate(current_filter):
				_st = _stackSt.select(station=f).copy()
				_st.sort(['channel'])
				_std_st = _stdSt.select(station=f).copy()
				_std_st.sort(['channel'])
				delta = _st[0].stats.delta
				sind = int(window[0]/delta)
				eind = sind + int(window[1]/delta)
				for i in range(n):
					_ax_st = self.fig.add_subplot(gs[i,ind+1])
					if i == n-1:
						_ax_st.set_xlabel('Time (s)')
					if i == 0:
						_ax_st.set_title('Filter: %s'%f)
					peak, data = norm(_st[i].data, sind, eind)
					if self.nbtn.isChecked():
						_ax_st.plot(time, data,'darkred', label=_st[i].stats.channel)
						if self.ebtn.isChecked():
							_ax_st.errorbar(time, data, yerr=2*_std_st[i].data,
									 marker='.',mew=0.1, ecolor='red', linewidth=0.2, markersize=0.2,
									 capsize=0.1, alpha=0.5)
						_ax_st.set_ylim([-0.1, 1.1])
					else:
						_ax_st.plot(time, _st[i].data,'darkred', label=_st[i].stats.channel)
						if self.ebtn.isChecked():
							_ax_st.errorbar(time, _st[i].data, yerr=2*_std_st[i].data,
									 marker='.',mew=0.1, ecolor='red', linewidth=0.2, markersize=0.2,
									 capsize=0.1, alpha=0.5)
						peak = peak + 0.1
						_ax_st.set_ylim([-0.1, peak])
					_ax_st.hlines(0,time[0],time[-1],'k')
					_ax_st.set_xlim([-window[0], window[0]+window[1]])
					_ax_st.legend()
		else:
			_region = self._region[1:]
			_current_df = self._current_array_df
			#_i = self.discb.currentIndex()
			#this_dis = self.dis[_i]
			#step_forward = this_dis['step'] * (1 - this_dis['overlap'])
			#n = int((this_dis['maxdis'] - this_dis['mindis'])/step_forward) + 1
			gs = self.fig.add_gridspec(n,n_filt+1)
			_ax = self.fig.add_subplot(gs[:,0])
			m = Basemap(projection='cyl', lon_0=0.0, lat_0=0.0,
						area_thresh=10000,ax=_ax)
			x0, y0 = m(self._current_array['lon'], self._current_array['lat'])
			x, y = m(self._current_array_df.lon.tolist(), self._current_array_df.lat.tolist())
			m.drawcoastlines(ax=_ax)
			m.drawmapboundary(ax=_ax)
			m.fillcontinents(color='lightgray',lake_color='white',ax=_ax)
			parallels = np.arange(-90.0, 90.0, 60)
			m.drawparallels(parallels,ax=_ax)
			meridians = np.arange(-180.0, 180.0, 60.0)
			m.drawmeridians(meridians,ax=_ax)
			x0, y0 = m(self._current_array['lon'], self._current_array['lat'])
			m.scatter(x0, y0, marker='*',c='r',s=100,alpha=0.7,ax=_ax,zorder=10)

			for _i, _reg in enumerate(_region):
				_name = _reg['name']
				# print(_name)
				_df = self.regDf[_name]
				x, y = m(_df.lon.tolist(), _df.lat.tolist())
				m.scatter(x, y, marker='o', c=color[_i], s=50, alpha=0.7,ax=_ax,zorder=10)
				_stackSt = self.stackSt[_name]
				_stdSt = self.stdSt[_name]
				delta = _stackSt[0].stats.delta
				npts = _stackSt[0].stats.npts
				time = np.arange(npts)*delta + _stackSt[0].stats.sac.b
				for ind, f in enumerate(current_filter):
					_st = _stackSt.select(station=f).copy()
					_st.sort(['channel'])
					_std_st = _stdSt.select(station=f).copy()
					_std_st.sort(['channel'])
					delta = _st[0].stats.delta
					sind = int(window[0]/delta)
					eind = sind + int(window[1]/delta)
					for i in range(n):
						_ax_st = self.fig.add_subplot(gs[i,ind+1])
						if i == n-1:
							_ax_st.set_xlabel('Time (s)')
						if i == 0:
							_ax_st.set_title('Filter: %s'%f)
						peak, data = norm(_st[i].data, sind, eind)
						label = _name+':'+_st[i].stats.channel
						if self.nbtn.isChecked():
							_ax_st.plot(time, data,color=color[_i],label=label)
							if self.ebtn.isChecked():
								_ax_st.errorbar(time, data, yerr=2*_std_st[i].data,
											 marker='.',mew=0.1, ecolor=color[_i], linewidth=0.2, markersize=0.2,
											 capsize=0.1, alpha=0.5)
							_ax_st.set_ylim([-0.1, 1.1])
						else:
							_ax_st.plot(time, _st[i].data,color='dark'+color[_i],label=label)
							if self.ebtn.isChecked():
								_ax_st.errorbar(time, _st[i].data, yerr=2*_std_st[i].data,
											 marker='.',mew=0.1, ecolor=color[_i], linewidth=0.2, markersize=0.2,
											 capsize=0.1, alpha=0.5)
							peak = peak+0.1
							_ax_st.set_ylim([-0.1, peak])
						_ax_st.hlines(0,time[0],time[-1],'k')
						_ax_st.set_xlim([-window[0], window[0]+window[1]])
						_ax_st.legend()
		self._canvasDraw()

	def _stackAll(self):
		if self.stbtn.isChecked():
			self._drawStack()
		else:
			self._drawFig()

	def _line_select_callback(self, eclick, erelease):
		x1, y1 = eclick.xdata, eclick.ydata
		x2, y2 = erelease.xdata, erelease.ydata
		# msg= 'Startposition: (%f, %f)\tendposition: (%f, %f)'%(x1, y1, x2, y2)
		# gpar.log(__name__,msg,level='info',pri=True)


	def _pltRectangle(self):
		_ax = self.fig.get_axes()[0]
		self._RS = RectangleSelector(_ax, self._line_select_callback,
											   drawtype='box', useblit=True,
											   button=[1],
											   minspanx=1, minspany=1,
											   interactive=True,
											   state_modifier_keys={'move': ' ','center': 'ctrl',
											   'square': 'shift','clear': self._shortcuts['cancel']})

	def _selectRegOnPress(self,event):

		if event.key is not None:
			event.key = event.key.lower()
		if event.inaxes is None:
			return
		if event.key == self._shortcuts['accept'] and self._RS.active:
			extents=self._RS.extents
			_value = dict(name='name',lonmin=extents[0],lonmax=extents[1],latmin=extents[2],latmax=extents[3])
			self._newReg(_value)


	def _newReg(self, value=None):
		newReg = self.defReg(regionValue=value)
		if newReg.exec_():
			self._region.append(newReg.getValues())
			self._updateRegion()
			self.regcb.setCurrentIndex(len(self._region)-1)
			self._calStack()
			# self._appStack()

	def _editRegion(self):

		_i =self.regcb.currentIndex()
		this_region = self._region[_i]
		editRegion = self.defReg(self, this_region)
		if editRegion.exec_():
			self._region[_i] = editRegion.getValues()
			self.updateRegion()
			self.regcb.setCurrentIndex(_i)
			self._calStack()
			self._drawStack()

	def _deleteRegion(self):
		_i = self.regcb.currentIndex()
		name = self._region[_i]['name']
		if name == 'global':
			msg = QMessageBox()
			msg.setIcon(QMessageBox.Information)
			msg.setText("Warning Text")
			msg.setInformativeText("Global region is not deletebale")
			msg.exec_()
			return
		else:
			self._region.pop(_i)
			self._updateRegion()

	def _changeRegion(self, index):
		if index == len(self._region):
			self._newReg()
			if self.stbtn.isChecked():
				self._drawStack()
			else:
				self._drawFig()
		else:
			if self.stbtn.isChecked():
				self._drawStack()
			else:
				self._drawFig()


	def _updateRegion(self):
		self.regcb.clear()
		self.regcb.setCurrentIndex(-1)
		for _i, _f in enumerate(self._region):
			self.regcb.addItem('Region: %s'%(_f['name']))
		self.regcb.addItem('Create new region')

	def _resetReg(self):
		self._region = [{'name':'global',
						 'latmin':-90.0, 'latmax':90.0,
						 'lonmin': -180.0, 'lonmax': 180.0}]
		self.regDf = {'global':self._current_array_df}
		self._updateRegion()
		self._drawFig()

	def _changeStack(self,index):
		if index == len(self.dis):
			self._newDis()
			if self.stbtn.isChecked():
				self._drawStack()
			else:
				self._drawFig()
		else:
			self._calStack()
			if self.stbtn.isChecked():
				self._drawStack()
			else:
				self._drawFig()

	def _newDis(self):
		newDis = self.defDisStep(self)

		if newDis.exec_():
			self.dis.append(newDis.getValues())
			self._updateWindow()
			self.discb.setCurrentIndex(len(self.dis)-1)
			self._calStack()

	def _editDis(self):
		_i = self.discb.currentIndex()
		this_window = self.dis[_i]
		editWindow = self.defDisStep(self, this_window)
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

	class defReg(QDialog):
		def __init__(self, parent=None, regionValue=None):
			QDialog.__init__(self, parent)
			self.setWindowTitle('Assign Name for the Region')
			self.Name = QLineEdit('Name')
			self.Name.selectAll()
			self.latmin = QDoubleSpinBox(decimals=1, maximum=90.0, minimum=-90.0, singleStep=5, value=0)
			self.latmax = QDoubleSpinBox(decimals=1, maximum=90.0, minimum=-90.0, singleStep=5, value=0)
			self.lonmin = QDoubleSpinBox(decimals=1, maximum=180.0, minimum=-180.0, singleStep=5, value=0)
			self.lonmax = QDoubleSpinBox(decimals=1, maximum=180.0, minimum=-180.0, singleStep=5, value=0)
			# self.saveTr = ['True', 'False']

			grid = QGridLayout()
			grid.addWidget(QLabel('Region Name'), 0, 0)
			grid.addWidget(self.Name, 0, 1)
			grid.addWidget(QLabel('Min. Lat'), 1, 0)
			grid.addWidget(self.latmin, 1, 1)
			grid.addWidget(QLabel('Max. Lat'), 2, 0)
			grid.addWidget(self.latmax, 2, 1)
			grid.addWidget(QLabel('Min. Lon'), 3, 0)
			grid.addWidget(self.lonmin, 3, 1)
			grid.addWidget(QLabel('Max. Lon'), 4, 0)
			grid.addWidget(self.lonmax, 4, 1)
			grid.setVerticalSpacing(10)

			btnbox = QDialogButtonBox(QDialogButtonBox.Ok |
											QDialogButtonBox.Cancel)

			btnbox.accepted.connect(self.accept)
			btnbox.rejected.connect(self.reject)

			layout = QVBoxLayout()
			layout.addWidget(QLabel('Define noise window and coda window for stripping'))
			layout.addLayout(grid)
			layout.addWidget(btnbox)

			if regionValue is not None:
				self.Name.setText(regionValue['name'])
				self.latmin.setValue(regionValue['latmin'])
				self.latmax.setValue(regionValue['latmax'])
				self.lonmin.setValue(regionValue['lonmin'])
				self.lonmax.setValue(regionValue['lonmax'])
			self.setLayout(layout)
			self.setSizeGripEnabled(False)

		def getValues(self):
			return dict(name=str(self.Name.text()),
						latmin=float(self.latmin.cleanText()),
						latmax=float(self.latmax.cleanText()),
						lonmin=float(self.lonmin.cleanText()),
						lonmax=float(self.lonmax.cleanText()))

	class defDisStep(QDialog):
		def __init__(self, parent=None, stepvalue=None):

			QDialog.__init__(self, parent)
			self.setWindowTitle('Create new stacking distance step')
			self.mindis = QDoubleSpinBox(decimals=1, maximum=180, minimum=0, singleStep=1, value=50)
			self.maxdis = QDoubleSpinBox(decimals=1, maximum=180, minimum=0, singleStep=1, value=75)
			self.step = QDoubleSpinBox(decimals=1, maximum=100, minimum=0.1, singleStep=0.1, value=0.1)
			self.overlap = QDoubleSpinBox(decimals=2, maximum=1.0, minimum=0.0, singleStep=0.1, value=0.1)
			self.Name = QLineEdit('Name')
			self.Name.selectAll()
			self.saveTr = ['True', 'False']

			grid = QGridLayout()
			grid.addWidget(QLabel('Name'), 0, 0)
			grid.addWidget(self.Name, 0, 1)
			grid.addWidget(QLabel('Min. Dis'), 1, 0)
			grid.addWidget(self.mindis, 1, 1)
			grid.addWidget(QLabel('Max. Dis'), 2, 0)
			grid.addWidget(self.maxdis, 2, 1)
			grid.addWidget(QLabel('Step'), 3, 0)
			grid.addWidget(self.step, 3, 1)
			grid.addWidget(QLabel('Overlap'), 4, 0)
			grid.addWidget(self.overlap, 4, 1)

			_savebtn = [QRadioButton("Yes"), QRadioButton("No")]
			self.saveGrp = QButtonGroup()
			self.saveGrp.setExclusive(True)
			sbtn = QHBoxLayout()

			for _i, _btn in enumerate(_savebtn):
				self.saveGrp.addButton(_btn, _i)
				sbtn.addWidget(_btn)
			grid.addWidget(QLabel('Save Stack'), 5, 0)
			grid.addLayout(sbtn, 5, 1)

			btnbox = QDialogButtonBox(QDialogButtonBox.Ok |
											QDialogButtonBox.Cancel)

			btnbox.accepted.connect(self.accept)
			btnbox.rejected.connect(self.reject)

			layout = QVBoxLayout()
			layout.addWidget(QLabel('Define distance steps and overlap for stacking'))
			layout.addLayout(grid)
			layout.addWidget(btnbox)

			if stepvalue is not None:
				self.Name.setText(stepvalue['name'])
				self.mindis.setValue(stepvalue['mindis'])
				self.maxdis.setValue(stepvalue['maxdis'])
				self.step.setValue(stepvalue['step'])
				self.overlap.setValue(stepvalue['overlap'])
			self.setLayout(layout)
			self.setSizeGripEnabled(False)

		def getValues(self):

			savefile = self.saveTr[self.saveGrp.checkedId()]

			return dict(name=str(self.Name),
						mindis=float(self.mindis.cleanText()),
						maxdis=float(self.maxdis.cleanText()),
						step=float(self.step.cleanText()),
						overlap=float(self.overlap.cleanText()),
						write=bool(savefile))



def codaStrip(eve, method='all',
			  siglen=200, noise=200,beamphase='PKiKP',
			  phase_list=['P','PP','PcP','ScP','PKiKP','SP','ScS'],
			  model='ak135', stime=400.0, etime=1800.0,
			  window=10, write=False):
	"""
	Function to remove background coda noise for events
	"""
	if phase_list is None:
		phase_list = eve.phase_list
	if not hasattr(eve, 'arrivals'):
		eve.getArrival(phase_list=phase_list,model=model)

	if not hasattr(eve, 'beam'):
		msg = ('Earthquake object has not calculate the beamforming yet')
		gpar.log(__name__,msg,level='error',pri=True)
	st = eve.beam
	delta = st[0].stats.delta
	noi_sind = int((eve.arrivals['P']['TT'] - 300.0)/delta)
	noi_win = int(100.0/delta)
	filts=[]
	
	starttime = st[0].stats.starttime
	tt1 = eve.arrivals[beamphase]['TT'] - noise #- starttime
	tt2 = eve.arrivals[beamphase]['TT'] + siglen #- starttime
	# tari = eve.arrivals[beamphase]['UTC'] - starttime
	tari = eve.arrivals[beamphase]['TT']
	n_tr = len(st)
	npts = st[0].stats.npts
	if npts%2 !=0:
		npts = npts-1
	data = np.empty((n_tr,npts))
	# mean = np.empty((n_tr, 1))
	for ind, tr in enumerate(st):
		npt = tr.stats.npts
		if npt%2 != 0:
			tdata = tr.data[:-1]
		else:
			tdata = tr.data
		tmp_data = np.abs(scipy.signal.hilbert(tdata))
		# tmp_data = moving_ave(tmp_data, window)
		# mean[ind,:] = np.mean(tmp_data[noi_sind:noi_sind+noi_win])
		# print(mean)
		# data[ind,:] = tmp_data - mean
		# print(data[ind,:])
		data[ind,:] = tmp_data

		filts.append(tr.stats.channel)
	data = moving_ave(data, window)
	# data = np.abs(scipy.signal.hilbert(tr.data))
	sig_pts = int(siglen/delta) + 1
	noi_pts = int(noise/delta) + 1
	noi_ind1 = int(tt1/delta)
	sig_ind = int(tari/delta)
	noi_ind2 = int(tt2/delta)
	time_before = tt1 + np.arange(int(noise/delta)+1) * delta

	# data_before = np.empty((n_tr, noi_pts))
	data_before = data[:, noi_ind1: noi_ind1 + noi_pts]
	# print(np.min(data_before))

	data_sig = data[:, sig_ind:sig_ind + sig_pts]

	time_after = tt2 + np.arange(int(noise/delta)+1) * delta
	data_after = data[:, noi_ind2: noi_ind2+noi_pts]
	# print(np.min(data_after))
	sind = int(stime/delta)
	# npts = int((etime - stime)/delta) + 1
	npts = int((etime - stime)/delta)
	# time = np.matrix(np.linspace(stime, etime, npts))
	time = stime + np.arange(npts) * delta
	obs_data = data[:, sind: sind+npts]
	ind = int((tari - stime)/delta)
	res_ind = int((tt1 - stime)/delta)
	pts = int((noise*2+siglen)/delta)+1
	if method == 'all':
		#fitting coda model
		coda_par = codaFit(np.append(time_before,time_after),np.append(data_before,data_after,axis=1))
		# print(coda_par)
		#getting predict noise signal in linear scale
		coda_data = np.asarray(np.exp(np.transpose(coda_par[0,:]) - np.transpose(coda_par[1,:]) \
					*np.log(time) - np.transpose(coda_par[2,:])*time))
		# coda_data = np.asarray(coda_data)
		#getting residual signal after removing the predict noise
		# coda_res = moving_ave(obs_data, window) - coda_data
		coda_res = obs_data - coda_data
		res = np.mean(coda_res[:,ind:ind+sig_pts],axis=-1)
		#store coda model information
		_df = pd.DataFrame(columns=['FILT','lnA','B','C','RMS'])
		_df['FILT'] = filts
		_df['lnA'] = np.asarray(coda_par)[0]
		_df['B'] = np.asarray(coda_par)[1]
		_df['C'] = np.asarray(coda_par)[2]
		_df['RMS'] = res
		# codamod = {'RMS':res,'lnA':coda_par[0][0],'B':coda_par[1][0],'C':coda_par[2][0]}
		# eve.codaMod = codamod
		eve.codaMod = _df
		codaSt = obspy.core.stream.Stream()
		resSt = obspy.core.stream.Stream()
		for i in range(n_tr):
			_tr = obspy.core.trace.Trace()
			_tr.stats.delta = delta
			_tr.stats.npts = npts
			_tr.stats.starttime = starttime + stime
			_tr.stats.channel = filts[i]
			_tr.data = coda_data[i,:]
			codaSt.append(_tr)
			_tr = obspy.core.trace.Trace()
			_tr.stats.delta = delta
			_tr.stats.npts = pts
			_tr.stats.starttime = eve.arrivals[beamphase]['UTC'] - noise
			_tr.stats.channel = filts[i]
			_tr.data = coda_res[i,res_ind:res_ind+pts]
			resSt.append(_tr)
		eve.codaSt = codaSt
		eve.codaResSt = resSt
		#fittint twoline model
		#For multi-filters all calculations are using np.matrix
		#all np.matrix are converted back to np.array to store
		twoline_par_before = twoLineFit(time_before, data_before)
		twoline_par_after = twoLineFit(time_after, data_after)
		y1 = twoline_par_before[0,:] + twoline_par_before[1,:] * tari
		y2 = twoline_par_after[0,:] + twoline_par_after[1,:] * tt2
		k = (y2 - y1)/(tt2 - tari)
		b = y2 - k * tt2
		t1 = np.matrix(np.linspace(tt1,tari, int(noise/delta)+1))
		d1 = np.asarray(np.transpose(twoline_par_before[0,:]) + np.transpose(twoline_par_before[1,:]) * t1)
		t2 = np.matrix(np.linspace(tari+delta, tt2, int(siglen/delta)))
		d2 = np.asarray(np.transpose(k) * t2 + np.transpose(b))
		t3 = np.matrix(np.linspace(tt2+delta,tt2+noise, int(noise/delta)))
		d3 = np.asarray(np.transpose(twoline_par_after[0,:]) + np.transpose(twoline_par_after[1,:]) * t3)
		two_data = np.append(d1,d2,axis=-1)
		two_data = np.append(two_data,d3,axis=-1)
		two_res = moving_ave(obs_data[:,res_ind: res_ind+pts], window) - 10**two_data
		res = np.mean(two_res[:,int(int(noise)/delta):int(int(noise)/delta)+sig_pts],axis=-1)
		_df = pd.DataFrame(columns=['FILT','kn1','bn1','kn2','bn2','RMS'])
		_df['FILT'] = filts
		_df['kn1'] = np.asarray(twoline_par_before)[1]
		_df['bn1'] = np.asarray(twoline_par_before)[0]
		_df['kn2'] = np.asarray(twoline_par_after)[1]
		_df['bn2'] = np.asarray(twoline_par_after)[0]
		_df['RMS'] = res
		# twomod = {'kn1':twoline_par_before[1][0],'bn1':twoline_par_before[0][0],
		# 		  'kn2':twoline_par_after[1][0],'bn2':twoline_par_after[0][0],'RMS':res}
		eve.twoMod = _df
		twoSt = obspy.core.stream.Stream()
		resSt = obspy.core.stream.Stream()
		pts = int((noise*2+siglen)/delta)+1
		# res_ind = int((tt1 - stime)/delta)
		for i in range(n_tr):
			_tr = obspy.core.trace.Trace()
			_tr.stats.delta = delta
			_tr.stats.starttime = starttime + tt1
			_tr.stats.channel = filts[i]
			_tr.data = two_data[i]
			twoSt.append(_tr)
			_trr = obspy.core.trace.Trace()
			_trr.stats.delta = delta
			_trr.stats.starttime = eve.arrivals[beamphase]['UTC'] - noise
			_trr.stats.channel = filts[i]
			_trr.stats.npts = pts
			_trr.data = two_res[i]
			resSt.append(_trr)
		eve.twoSt = twoSt
		eve.twoResSt = resSt
	elif method == 'coda':
		#fitting coda model
		coda_par = codaFit(np.append(time_before,time_after),np.append(data_before,data_after,axis=1))
		#getting predict noise signal in linear scale
		coda_data = np.asarray(np.exp(np.transpose(coda_par[0,:]) - np.transpose(coda_par[1,:]) \
					*np.log(time) - np.transpose(coda_par[2,:])*time))
		#getting residual signal after removing the predict noise
		coda_res = moving_ave(obs_data, window) - coda_data
		res = np.mean(coda_res[:,ind:ind+sig_pts],axis=-1)
		#store coda model information
		_df = pd.DataFrame(columns=['FILT','lnA','B','C','RMS'])
		_df['FILT'] = filts
		_df['lnA'] = np.asarray(coda_par)[0]
		_df['B'] = np.asarray(coda_par)[1]
		_df['C'] = np.asarray(coda_par)[2]
		_df['RMS'] = res
		eve.codaMod = _df
		codaSt = obspy.core.stream.Stream()
		resSt = obspy.core.stream.Stream()
		# res_ind = int((tt1 - stime)/delta)
		# pts = int((noise*2+siglen)/delta)+1
		for i in range(n_tr):
			_tr = obspy.core.trace.Trace()
			_tr.stats.delta = delta
			_tr.stats.npts = npts
			_tr.stats.starttime = starttime + stime
			_tr.stats.channel = filts[i]
			_tr.data = coda_data[i,:]
			codaSt.append(_tr)
			_tr = obspy.core.trace.Trace()
			_tr.stats.delta = delta
			_tr.stats.npts = pts
			_tr.stats.starttime = eve.arrivals[beamphase]['UTC'] - noise
			_tr.stats.channel = filts[i]
			_tr.data = coda_res[i,res_ind:res_ind+pts]
			resSt.append(_tr)
		eve.codaSt = codaSt
		eve.codaResSt = resSt
	elif method == 'twoline':
		#fittint twoline model
		twoline_par_before = twoLineFit(time_before, data_before)
		twoline_par_after = twoLineFit(time_after, data_after)
		y1 = twoline_par_before[0,:] + twoline_par_before[1,:] * tari
		y2 = twoline_par_after[0,:] + twoline_par_after[1,:] * tt2
		k = (y2 - y1)/(tt2 - tari)
		b = y2 - k * tt2
		t1 = np.matrix(np.linspace(tt1,tari, int(noise/delta)+1))
		d1 = np.asarray(np.transpose(twoline_par_before[0,:]) + np.transpose(twoline_par_before[1,:]) * t1)
		t2 = np.matrix(np.linspace(tari+delta, tt2, int(siglen/delta)))
		d2 = np.asarray(np.transpose(k) * t2 + np.transpose(b))
		t3 = np.matrix(np.linspace(tt2+delta,tt2+noise, int(noise/delta)))
		d3 = np.asarray(np.transpose(twoline_par_after[0,:]) + np.transpose(twoline_par_after[1,:]) * t3)
		two_data = np.append(d1,d2,axis=-1)
		two_data = np.append(two_data,d3,axis=-1)
		two_res = moving_ave(obs_data[:,res_ind: res_ind+pts], window) - 10**two_data
		res = np.mean(two_res[:,int(int(noise)/delta):int(int(noise)/delta)+sig_pts],axis=-1)
		_df = pd.DataFrame(columns=['FILT','kn1','bn1','kn2','bn2','RMS'])
		_df['FILT'] = filts
		_df['kn1'] = np.asarray(twoline_par_before)[1]
		_df['bn1'] = np.asarray(twoline_par_before)[0]
		_df['kn2'] = np.asarray(twoline_par_after)[1]
		_df['bn2'] = np.asarray(twoline_par_after)[0]
		_df['RMS'] = res
		# twomod = {'kn1':twoline_par_before[1][0],'bn1':twoline_par_before[0][0],
		# 		  'kn2':twoline_par_after[1][0],'bn2':twoline_par_after[0][0],'RMS':res}
		eve.twoMod = _df
		twoSt = obspy.core.stream.Stream()
		resSt = obspy.core.stream.Stream()
		pts = int((noise*2+siglen)/delta)+1
		# res_ind = int((tt1 - stime)/delta)
		for i in range(n_tr):
			_tr = obspy.core.trace.Trace()
			_tr.stats.delta = delta
			_tr.stats.starttime = starttime + tt1
			_tr.stats.channel = filts[i]
			_tr.data = two_data[i]
			twoSt.append(_tr)
			_trr = obspy.core.trace.Trace()
			_trr.stats.delta = delta
			_trr.stats.starttime = eve.arrivals[beamphase]['UTC'] - noise
			_trr.stats.channel = filts[i]
			_trr.stats.npts = pts
			_trr.data = two_res[i]
			resSt.append(_trr)
		eve.twoSt = twoSt
		eve.twoResSt = resSt
	return eve




def codaFit(time, amp):
	"""
	Function to remove background coda noise by fitting coda model s=A*B^-t*exp(-Ct) in log scale
	"""
	amp = np.matrix(np.log(amp))
	n = len(time)
	# amp = amp.reshape(n,1)
	amp = amp.transpose()
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

	# return np.asarray(m)
	return m

def twoLineFit(time, amp):
	"""
	Function to do the two line method fitting to get the coda
	"""

	amp = np.matrix(np.log10(amp))
	n = len(time)
	# amp = amp.reshape(n,1)
	amp = amp.transpose()
	A = np.zeros((n,2))
	A[:,0] = np.ones(time.shape)
	A[:,1] = time

	A = np.matrix(A)
	G = A.transpose() * A
	d = A.transpose() * amp
	m = np.linalg.inv(G) * d

	# return np.asarray(m)
	return m

def moving_ave(x, window=5):
	"""
	Function to smooth data
	"""
	ind,_ = x.shape
	weigths = np.ones(window)/window
	y = np.empty(x.shape)
	for i in range(ind):
		y[i,:] = np.convolve(x[i,:],weigths,mode='same')

	return y

def norm(data,sind, eind):

	peak = np.max(np.absolute(data[sind:eind]))
	data = data/peak

	return peak, data

def stack(df,win):

	data = df.DATA.values
	std_data = np.std(data,axis=0)/np.sqrt(len(data))
	#stack_data = np.sum(data,axis=0)/len(df)
	stack_data = np.mean(data,axis=0)
	# peak = np.max(np.absolute(stack_data[win[0]:win[1]]))
	# stack_data = stack_data/peak

	return stack_data, std_data

def misfit(data1,data2):

	l2 = np.sum((data1-data2)**2)

	return l2

def stackTR(obsdf, pklname=None,win=[200.0,200.0],
			mindis=50., maxdis=75.,
		   step=5.0,overlap=0.0, write=False):
	
	deltas = obsdf.delta.unique()
	filters = obsdf.iloc[0].crms.FILT
	if len(deltas) != 1:
		delta = np.max(deltas)
		sample = 1.0/delta
		#npt = obsdf[obsdf.delta == delta].iloc[0].codaResSt[0].stats.npts
		dts = deltas[deltas != delta]
		for dt in dts:
			index = obsdf[obsdf.delta == dt].index
			for ind in index:
				obsdf.iloc[ind].codaResSt.resample(sample)
	else:
		delta = deltas[0]
		#npt = obsdf.iloc[0].codaResSt[0].stats.npts
	sind = int(win[0]/delta)
	eind = sind + int(win[1]/delta)
	npt = int((win[0]*2 + win[1])/delta) + 1
	time = np.arange(npt) * delta - win[0]
	DATA = [''] * len(obsdf)
	for ind, row in obsdf.iterrows():
		codaResSt = row.codaResSt
		tmp_data = np.empty((len(filters), npt))
		for i, tr in enumerate(codaResSt):
			_, td = norm(tr.data, sind, eind)
			if len(td) > npt:
				td = td[0:npt]
			elif len(td) < npt:
				td = np.pad(td,(0, npt-len(td)), 'edge')
			# _, tmp_data[i] = norm(tr.data, sind, eind)
			tmp_data[i] = td
		DATA[ind] = tmp_data
	obsdf['DATA'] = DATA
	n = len(obsdf)
	st_obs = Stream()
	st_std = Stream()
	step_forward = step * (1-overlap)
	n = int((maxdis - mindis)/step_forward) + 1

	ds = np.linspace(mindis, maxdis, n)[:-1]

	for d in ds:
		_df = obsdf[(obsdf.Del >= d) & (obsdf.Del < d+step)]
		n_eq = len(_df)
		if len(_df) != 0:
			stack_obs, std_obs = stack(_df, [sind, eind])
		else:
			stack_obs = np.zeros((len(filters), npt))
			std_obs = np.zeros((len(filters), npt))
		for i, filt in enumerate(filters):
			tr_obs = Trace()
			tr_obs.stats.sac = AttribDict()
			tr_obs.stats.delta = delta
			tr_obs.stats.npts = npt
			tr_obs.stats.network = 'stack'
			tr_obs.stats.station = filt
			tr_obs.stats.channel = 'd'+str(d)+'-n'+str(n_eq)
			tr_obs.stats.sac.delta = delta
			tr_obs.stats.sac.b = -win[0]
			tr_obs.stats.sac.e = win[1]
			tr_obs.stats.sac.npts = npt
			tr_std = tr_obs.copy()
			tr_std.stats.network = 'std'
			tr_obs.data = stack_obs[i]
			tr_std.data = std_obs[i]
			st_obs.append(tr_obs)
			st_std.append(tr_std)

	if write:
		if pklname is None:
			pklname = 'stack.pkl'
		st_obs.write(pklname,format='PICKLE')

	return st_obs, st_std
