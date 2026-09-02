# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:36:37 2026

@author: wimme
"""
# Imports
# Science
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')


# Absoluten Pfad zum Ordner 6-gauge_data berechnen
base_dir = os.path.dirname(__file__)
daq_path = os.path.abspath(os.path.join(base_dir, "0-gauge_data"))

# Pfad hinzufügen
sys.path.append(daq_path)

try:
    from Python_DAQ import *
except:
    from DAQ_Python.Python_DAQ import *

# LOCATION of the EVENT of interest
loc_folder = os.path.abspath(os.path.join(base_dir, "..", "gauge_dat"))


loc = os.path.abspath(os.path.join(loc_folder, loc_file))
loc_zero = os.path.abspath(os.path.join(loc_folder, loc_file_zero))

"""

READ IN DATA TO ANALYSE !!!!!!!!!!!!!

"""

# %% ### Plot macroscopic checks

plt.figure(dpi=300)
plt.title("General forces overview")
plt.suptitle(loc_folder + loc_file, alpha=.2, size=5)

# Thinner lines
lw = 0.8

colors = ["#0072B2",  # blue
          "#009E73",  # green
          "#D55E00",  # orange
          "#800080"]  # purple

# Compute time arrays
time_ac = np.linspace(0, len(ch_1)/1e6, len(ch_1))
f_ech_sm = len(time_sm)/(time_sm[-1] - time_sm[0])
f_ech_ac = 1e6
A = 3000 * f_ech_sm / f_ech_ac

cor_1, cor_2 = 30000, int(np.shape(time_sm)[0])- 100000

print("SM time stemps: "+str(np.shape(time_sm)[0]))

# ============================================================
# Trigger peak detection
# ============================================================

# Minimum cooldown between two trigger peaks, in seconds
trigger_cooldown = 1

# Minimum trigger peak amplitude
trigger_peak_amplitude = 400

# Convert cooldown from seconds to samples
min_peak_distance = int(trigger_cooldown * f_ech_sm)

# Only search for peaks in the plotted region
trigger_section = trigger[cor_1:cor_2]
time_section = time_sm[cor_1:cor_2]

# Find trigger maxima
peaks, properties = find_peaks(
    trigger_section,
    height=trigger_peak_amplitude,
    distance=min_peak_distance
)

# Convert peak indices back to full-array indices
peaks_full = peaks + cor_1


# Plot signals
plt.plot(time_sm[cor_1:cor_2], fn_sm[cor_1:cor_2], color=colors[0], lw=lw, label=r"$F_n$")
plt.plot(time_sm[cor_1:cor_2], fs_sm[cor_1:cor_2], color=colors[1], lw=lw, label=r"$F_s$")
plt.plot(time_sm[cor_1:cor_2], ((fs_sm / fn_sm) * 1000)[cor_1:cor_2], color=colors[2], lw=lw,label=r"$\mu$=$F_s$/$F_n$")
plt.plot(time_sm[cor_1:cor_2], trigger[cor_1:cor_2], color=colors[3], lw=lw, label="Trigger")

plt.scatter(
    time_sm[peaks_full],
    trigger[peaks_full],
    color="black",
    s=15,
    zorder=5
)

label_positions = [ (20, "top"), (10, "center"), (0, "bottom")]

for number, peak_idx in enumerate(peaks_full, start=1):
    offset, va = label_positions[(number - 1) % 3]
    plt.annotate(
        str(number),
        xy=(time_sm[peak_idx], trigger[peak_idx]),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=7,
        color="black",
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.8
        )
    )
    
# Axes labels
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")

# Dotted grid behind the plot
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

# Small legend
plt.legend(fontsize=10)

plt.axis(ymin=0, ymax=450)

# Save and show
plt.savefig(
    os.path.abspath(
        os.path.join(daq_path, "froces_overview.png")
    )
)

plt.show()

# %% ### Strain profile average

fig, axes = plt.subplots(dpi=300, sharex=True)
# --- Front side ---
s2_init_f = np.mean(s2[0:10,:], axis=1)
axes.plot(x_front*1000, s2_init_f*1e3, marker="x", label="Front", color=colors[0])

axes.set_ylabel(r"$\varepsilon_{yy}$")
axes.grid(which="both")

# --- Back side ---
s2_init_b = np.mean(s2[10:,:], axis=1)
axes.plot(x_back*1000, s2_init_b*1e3, marker="o", label="Back", color=colors[1])

# --- Shared y-limits ---
ymin = 1.1 * min(0, min(np.mean(gages[1::3, 19500:20500], axis=1))) * 1e3
ymax = 1.1 * max(np.mean(gages[1::3, 19500:20500], axis=1)) * 1e3

axes.set_xlabel("position (mm)")
axes.set_ylabel(r"$\varepsilon_{yy}$ [mV]")
axes.legend()

axes.set_title("Strain profile")
axes.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
axes.set_ylim((ymin, ymax))


plt.tight_layout()
plt.show()

# %% ### Strain sensors overt time

# Time axis (4 MHz)
fs = sampling_freq_in
n_samples = gages.shape[1]
time = np.arange(n_samples) / fs

# mV
factor_mV = 1e3
factor_ms = 1e3

lw = 0.7

# --- Level all signals to start at 0 ---
eps_xx_0 = eps_xx - eps_xx[:, [0]]
eps_yy_0 = eps_yy - eps_yy[:, [0]]
eps_xy_0 = eps_xy - eps_xy[:, [0]]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=300, sharex=True)
fig.suptitle("All strains Overview — event " + str(loc_file[7:9]))

#Choose time sequence by array slice 0 -> -1 = all
seq_1=0
seq_2=-1

order = []
for i in range(10):
    order.extend([i, i + 10])

for i, n in enumerate(order):
    axes[0].plot( time[seq_1:seq_2] * factor_ms, eps_xx_0[n][seq_1:seq_2] * factor_mV + 0.8 * i, lw=lw, color=colors[0], alpha=0.7 )

axes[0].set_ylabel(r"$\varepsilon_{xx} [mV]$")
axes[0].grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

# eps_yy
for i, n in enumerate(order):
    axes[1].plot( time[seq_1:seq_2] * factor_ms, eps_yy_0[n][seq_1:seq_2] * factor_mV + 0.8 * i, lw=lw, color=colors[1], alpha=0.7 )

axes[1].set_ylabel(r"$\varepsilon_{yy} [mV]$")
axes[1].grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

# eps_xy
for i, n in enumerate(order):
    axes[2].plot( time[seq_1:seq_2] * factor_ms, eps_xy_0[n][seq_1:seq_2] * factor_mV + 0.8 * i, lw=lw, color=colors[2], alpha=0.7 )

axes[2].set_ylabel(r"$\varepsilon_{xy} [mV]$")
axes[2].set_xlabel("time (ms)")
axes[2].grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

plt.tight_layout()
#plt.savefig(os.path.abspath(os.path.join(daq_path, "gauge(t)_event"+ str(loc_file[7:9]) +"_overview.png")))
plt.show()