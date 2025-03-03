import pandas as pd
import numpy as np 
from obspy import read
from obspy import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth
from obspy.taup import TauPyModel
import os
import matplotlib.pyplot as plt 
from gpar.arrayProcess import fkslowness
from gpar.arrayProcess import getTimeTable
import matplotlib.pyplot as plt
from cmcrameri import cm


ID = '2011.08.19.05.36.33'
cmar = pd.read_pickle('WRA.pkl')
for e in cmar.events:
	if e.ID == ID:
		eve = e
stadf = cmar.geometry
eve.getArrival()
arr = eve.arrivals['PKiKP']
st = eve.stream
# parameters for sliding window
grdpts_x=601; grdpts_y=601
filt = [1,2,4,True]
sll_x=-3.0; sll_y=-3.0
sl_s=0.01
starttime=1130; endtime=1140
unit='deg'

arrayName = 'WRA'


delta = st[0].stats.delta
ntr = len(st)

use_sta = pd.DataFrame()
for tr in st:
	sta = tr.stats.station
	row = stadf[stadf.STA == sta].iloc[0]
	use_sta = use_sta.append(row, ignore_index=True)

timeTable = getTimeTable(use_sta, sll_x, sll_y, sl_s, grdpts_x, grdpts_y,unit=unit)

relpow, abspow = fkslowness(st,ntr,timeTable,sll_x=sll_x,sll_y=sll_y,
							sl_s=sl_s,grdpts_x=grdpts_x,grdpts_y=grdpts_y,
							freqmin=1, freqmax=2,
							starttime=starttime, endtime=endtime)

ix, iy = np.unravel_index(abspow.argmax(),abspow.shape)
slow_x = sll_x + (ix-1) * sl_s
slow_y = sll_y + (iy-1) * sl_s
sx = np.arange(grdpts_x) * sl_s + sll_x
sy = np.arange(grdpts_y) * sl_s + sll_y
X, Y = np.meshgrid(sx, sy, indexing='ij')
rayp = eve.arrivals['PKiKP']['RP']
fig, ax = plt.subplots()
apow = 10 * np.log10(abspow / np.max(abspow))
# ax.imshow(apow.T,origin='lower', extent=(-3, 3, -3, 3),vmin=-0.1, vmax=0, cmap=cm.grayC)
c = ax.pcolormesh(X, Y, apow, vmin=-0.1, vmax=0, cmap=cm.grayC)
cbar = fig.colorbar(c, ax=ax,shrink=0.9)
phi = np.arange(0, 2*np.pi + np.pi/100, np.pi/100)
x = np.cos(phi) * rayp
y = np.sin(phi) * rayp
ax.plot(x, y, 'b')
ax.scatter(slow_x, slow_y, marker='.',s=50,color='r')
ax.scatter(0, 0, marker='+',s=50,color='white')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_xlabel('sx (s/deg)')
ax.set_ylabel('sy (s/deg)')
ax.set_aspect('equal',adjustable='box')
plt.savefig('slowness_fk.png')
plt.show()