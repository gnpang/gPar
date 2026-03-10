"""
Example: Sliding Beamforming Array Processing with gPar
========================================================
This script demonstrates how to run sliding beamforming (time-domain FK analysis)
on seismic array data using the gPar toolbox.

The sliding beamforming method sweeps a short time window across the data and
finds the best-fitting slowness vector at each time step, returning:
  - Amplitude (beam power)
  - Horizontal slowness (s/deg)
  - Back azimuth (degrees)
  - Coherence

Reference:
    Rost, S., & Thomas, C. Array seismology: Methods and applications.
    Rev. Geophys. 40(3), 2-1 (2002).

Usage:
    Edit the parameters below to match your data, then run:
        python calSlideBeam.py
"""

import pandas as pd
import numpy as np
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt

from gpar.arrayProcess import getTimeTable, slideBeam, getTimeShift

# ---------------------------------------------------------------------------
# User Parameters
# ---------------------------------------------------------------------------

# Path to the array geometry file (CSV/pickle with columns: STA, LAT, LON)
GEOMETRY_FILE = 'WRA.pkl'

# Event ID to process (must exist in the loaded array object)
EVENT_ID = '2011.08.19.05.36.33'

# Array name
ARRAY_NAME = 'WRA'

# Reference point of the array [lat, lon]
REF_POINT = [-19.94, 134.34]

# Slowness grid parameters (units: s/degree)
SLL_X = -3.0        # minimum slowness in X (E-W)
SLL_Y = -3.0        # minimum slowness in Y (N-S)
SL_S  =  0.05       # slowness step size
GRDPTS_X = 121      # number of grid points in X
GRDPTS_Y = 121      # number of grid points in Y
UNIT = 'deg'        # slowness unit: 'deg', 'km', or 'rad'

# Time window for sliding beamforming (seconds relative to stream start)
STARTTIME = 1100.0
ENDTIME   = 1200.0

# Sliding window parameters
WIN_LEN = 2.0       # window length in seconds
OVERLAP = 0.5       # fractional overlap between windows (0-1)

# Beam stacking method: 'linear', 'psw', or 'root'
STACK = 'linear'

# Amplitude metric: 1=max, 2=mean, 3=root-mean-square
SFLAG = 1

# Refine slowness estimate near the best-fitting grid point
REFINE = True

# Filter bank: dict of {name: [freqmin, freqmax, corners, zerophase]}
FILTS = {
    'filt_1': [1, 2, 4, True],
    'filt_2': [2, 4, 4, True],
}

# Output figure file (set to None to display interactively)
OUTPUT_FIGURE = 'slidebeam_result.png'

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print('Loading array data from %s ...' % GEOMETRY_FILE)
cmar = pd.read_pickle(GEOMETRY_FILE)

# Find the target event
eve = None
for e in cmar.events:
    if e.ID == EVENT_ID:
        eve = e
        break

if eve is None:
    raise ValueError('Event %s not found in %s' % (EVENT_ID, GEOMETRY_FILE))

stadf = cmar.geometry
st = eve.stream.copy()
ntr = len(st)
delta = st[0].stats.delta

print('Loaded event %s with %d traces (delta=%.3f s)' % (EVENT_ID, ntr, delta))

# ---------------------------------------------------------------------------
# Build array geometry (station coordinates relative to reference point)
# ---------------------------------------------------------------------------

use_sta = pd.DataFrame()
for tr in st:
    sta = tr.stats.station
    row = stadf[stadf.STA == sta].iloc[0]
    use_sta = pd.concat([use_sta, row.to_frame().T], ignore_index=True)

print('Array geometry built for %d stations.' % len(use_sta))

# ---------------------------------------------------------------------------
# Run sliding beamforming for each filter
# ---------------------------------------------------------------------------

print('Running sliding beamforming ...')
print('  Time window : %.1f - %.1f s' % (STARTTIME, ENDTIME))
print('  Win length  : %.1f s  overlap: %.0f%%' % (WIN_LEN, OVERLAP * 100))
print('  Slowness grid: %d x %d points, step=%.3f s/deg' % (GRDPTS_X, GRDPTS_Y, SL_S))

results = {}
for fname, filt in FILTS.items():
    print('  Filter %s: %.1f - %.1f Hz ...' % (fname, filt[0], filt[1]))
    rel = slideBeam(
        stream=st,
        ntr=ntr,
        delta=delta,
        geometry=use_sta,
        arrayName=ARRAY_NAME,
        grdpts_x=GRDPTS_X,
        grdpts_y=GRDPTS_Y,
        filt=filt,
        sflag=SFLAG,
        stack=STACK,
        sll_x=SLL_X,
        sll_y=SLL_Y,
        sl_s=SL_S,
        refine=REFINE,
        starttime=STARTTIME,
        endtime=ENDTIME,
        unit=UNIT,
        winlen=WIN_LEN,
        overlap=OVERLAP,
    )
    results[fname] = rel
    print('  Done. %d time steps computed.' % rel.shape[1])

# ---------------------------------------------------------------------------
# Build time axis for results
# ---------------------------------------------------------------------------

ndel = WIN_LEN * (1 - OVERLAP)
bnpts = int((ENDTIME - STARTTIME) / ndel)
t_axis = STARTTIME + np.arange(bnpts) * ndel + WIN_LEN / 2.0  # centre of each window

# ---------------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------------

nfilts = len(FILTS)
fig, axes = plt.subplots(4, nfilts, figsize=(5 * nfilts, 8),
                         sharex='col', sharey='row')

if nfilts == 1:
    axes = axes[:, np.newaxis]  # ensure 2-D array

labels = ['log10(Amplitude)', 'Slowness (s/deg)', 'Back Azimuth (deg)', 'Coherence']

for col, (fname, rel) in enumerate(results.items()):
    filt = FILTS[fname]
    axes[0, col].set_title('%s  (%.1f-%.1f Hz)' % (fname, filt[0], filt[1]))
    for row in range(4):
        ax = axes[row, col]
        ax.plot(t_axis, rel[row, :], linewidth=0.8)
        if col == 0:
            ax.set_ylabel(labels[row])
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

for col in range(nfilts):
    axes[-1, col].set_xlabel('Time (s)')

fig.suptitle('Sliding Beamforming – Event %s – Array %s' % (EVENT_ID, ARRAY_NAME))
plt.tight_layout()

if OUTPUT_FIGURE:
    plt.savefig(OUTPUT_FIGURE, dpi=150)
    print('Figure saved to %s' % OUTPUT_FIGURE)
else:
    plt.show()

print('Done.')
