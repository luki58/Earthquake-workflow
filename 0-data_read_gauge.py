"""
This files is used to pick for each event of each experiment the moment at which
the crack is detected on each gage.
"""

# Imports
# Science
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

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
loc_file = r"event-002.npy"

# name of the reference file, containing unloaded signal, usually event-001
loc_file_zero = r"event-001.npy"

loc = os.path.abspath(os.path.join(loc_folder, loc_file))
loc_zero = os.path.abspath(os.path.join(loc_folder, loc_file_zero))

# %%
# control smoothing of the data (rolling average) and starting point
roll_smooth = 1

# gages of interest and other channels
gages_true_number = np.arange(1, 61, 1)#all

# convert gage number into channel number
gages_channels = gauge_number_to_channel(gages_true_number)
nchannels = len(gages_channels)

# channels containing the normal and shear force, and the trigger
forces_channels = [15, 31]
trigger_channel = 47

# Load Parameters
# if x is not set in the parameters, set a default x :
x_front = np.linspace(0.010545, 0.010545*11, 10)
x_back = np.linspace(0.005273, 0.010545*11, 10)

x = np.array([0.018, 0.024, 0.03 , 0.036, 0.042, 0.048, 0.054, 0.06 , 0.066,
       0.072, 0.078, 0.084, 0.09 , 0.096, 0.102, 0.108, 0.114, 0.12 ,
       0.126, 0.132])

clock = 4e7
sampling_freq_in = clock/10

# Fast acquisition
# Load data
data = np.load(loc, allow_pickle=True)
data_zero = np.load(loc_zero, allow_pickle=True)
data_raw_event = data

# smooth data
data = smooth(data, roll_smooth)
data = np.transpose(np.transpose(data)-np.mean(data_zero, axis=1))

# assign specific channels to specific variables
forces = data[forces_channels, :]
mu = data[forces_channels[1], :].mean() / data[forces_channels[0], :].mean()
gages = data[gages_channels]
data_raw = data[gages_channels]
gages_zero = data_zero[gages_channels]
data_rawzero = data_zero[gages_channels]
fast_time = np.arange(len(gages[0]))/sampling_freq_in


# load slow monitoring
sm = np.load(loc_folder+"\slowmon.npy", allow_pickle=True)
time_sm = np.load(loc_folder+"\slowmon_time.npy")

# extract observables
gages_sm = sm[gages_channels]
gages_sm = np.transpose(np.transpose(gages_sm)-np.mean(gages_zero, axis=1))


forces_sm = sm[forces_channels, :]
fn_sm = forces_sm[0]
fs_sm = forces_sm[1]

trigger = 100*sm[trigger_channel]

# Convert voltage to strains or forces, and rosette to tensor

if not 0:
    for i in range(nchannels//3):
        ch_1 = gages[3*i]
        ch_2 = gages[3*i+1]
        ch_3 = gages[3*i+2]
        ch_1, ch_2, ch_3 = voltage_to_strains(ch_1, ch_2, ch_3, amp=1000)
        ch_1, ch_2, ch_3 = rosette_to_tensor(ch_1, ch_2, ch_3)
        gages[3*i] = ch_1
        gages[3*i+1] = ch_2
        gages[3*i+2] = ch_3

    for i in range(nchannels//3):
        ch_1 = gages_zero[3*i]
        ch_2 = gages_zero[3*i+1]
        ch_3 = gages_zero[3*i+2]
        ch_1, ch_2, ch_3 = voltage_to_strains(ch_1, ch_2, ch_3, amp=1000)
        ch_1, ch_2, ch_3 = rosette_to_tensor(ch_1, ch_2, ch_3)
        gages_zero[3*i] = ch_1
        gages_zero[3*i+1] = ch_2
        gages_zero[3*i+2] = ch_3

    for i in range(nchannels//3):
        ch_1 = gages_sm[3*i]
        ch_2 = gages_sm[3*i+1]
        ch_3 = gages_sm[3*i+2]
        ch_1, ch_2, ch_3 = voltage_to_strains(ch_1, ch_2, ch_3, amp=1000)
        ch_1, ch_2, ch_3 = rosette_to_tensor(ch_1, ch_2, ch_3)
        gages_sm[3*i] = ch_1
        gages_sm[3*i+1] = ch_2
        gages_sm[3*i+2] = ch_3

    forces = voltage_to_force(forces)
    forces_sm = voltage_to_force(forces_sm)
    fn_sm = voltage_to_force(fn_sm)
    fs_sm = voltage_to_force(fs_sm)

    s1_sm = gages_sm[0::3]
    s1_sm[10:] = np.flip(s1_sm[10:,:],axis=0)
    s2_sm = gages_sm[1::3]
    s2_sm[10:] = np.flip(s2_sm[10:,:],axis=0)
    s3_sm = gages_sm[2::3]
    s3_sm[10:] = np.flip(s3_sm[10:,:],axis=0)
    
    eps_yy_sm = s2_sm
    # Implement s1 - s3 for front and s3 - s1 for back
    eps_xy_sm = []
    for i in range(s1_sm.shape[0]):
        if i < 10:
            eps_xy_sm.append(0.5 * (s1_sm[i, :] - s3_sm[i, :]))
        else:
            eps_xy_sm.append(0.5 * (s3_sm[i, :] - s1_sm[i, :]))
    eps_xy_sm = np.array(eps_xy_sm)
    eps_xx_sm = (s1_sm + s3_sm) - s2_sm
    
    s1 = gages[0::3]
    s1[10:] = np.flip(s1[10:,:],axis=0)
    s2 = gages[1::3]
    s2[10:] = np.flip(s2[10:,:],axis=0)
    s3 = gages[2::3]
    s3[10:] = np.flip(s3[10:,:],axis=0)
    
    eps_yy = s2
    eps_xy = []
    for i in range(s1.shape[0]):
        if i < 10:
            eps_xy.append(0.5 * (s1[i, :] - s3[i, :]))
        else:
            eps_xy.append(0.5 * (s3[i, :] - s1[i, :]))
    eps_xy = np.array(eps_xy)
    eps_xx = (s1 + s3) - s2

# %% ### save data

to_save_fast = {
    "gages": gages,
    "x": x,
    "eps_yy": eps_yy,
    "eps_xx": eps_xx,
    "eps_xy": eps_xy,
    "fast_time": fast_time,
    "data_raw": data_raw,
    "data_rzero": data_rawzero,
    "data_raw_event": data_raw_event
}

to_save_sm = {
    "gages_sm": gages_sm,
    "time_sm": time_sm,
    "x": x,
    "eps_yy_sm": eps_yy_sm,
    "eps_xx_sm": eps_xx_sm,
    "eps_xy_sm": eps_xy_sm,
    "trigger_sm": trigger,
    "fn_sm": fn_sm,
    "fs_sm": fs_sm,
    "sm": sm
}

np.savez(os.path.abspath(os.path.join(daq_path, loc_file[:-4])) + ".npz", **to_save_fast)

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

cor_1, cor_2 = 0, int(np.shape(time_sm)[0])

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
plt.legend(fontsize=10, loc = "lower right")

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
s2_init_f = np.mean(eps_xy[0:10,:1000], axis=1)
axes.plot(x_front*1000, s2_init_f*1e3, marker="x", label="Front", color=colors[0])

axes.set_ylabel(r"$\varepsilon_{xy}$")
axes.grid(which="both")

# --- Back side ---
s2_init_b = np.mean(eps_xy[10:,:1000], axis=1)
axes.plot(x_back*1000, s2_init_b*1e3, marker="o", label="Back", color=colors[1])

# --- Shared y-limits ---
ymin = 1.1 * min(0, min(np.mean(gages[1::3, 19500:20500], axis=1))) * 1e3
ymax = 1.1 * max(np.mean(gages[1::3, 19500:20500], axis=1)) * 1e3

axes.set_xlabel("position (mm)")
axes.set_ylabel(r"$\varepsilon_{xy}$ [mV]")
axes.legend()

axes.set_title("Strain profile event " + str(loc_file[7:9]))
axes.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
#axes.set_ylim((ymin, ymax))


plt.tight_layout()
plt.savefig(os.path.abspath(os.path.join(daq_path, "strain_profile_event"+ str(loc_file[7:9]) +".png")))
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
plt.savefig(os.path.abspath(os.path.join(daq_path, "gauge(t)_event"+ str(loc_file[7:9]) +"_overview.png")))
plt.show()

# %% ### Strain sensors overt time focused on each event isolated 

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


# Choose time sequence by array slice 0 -> -1 = all
seq_1 = 15000
seq_2 = 25000

order = []
for i in range(10):
    order.extend([i, i + 10])   # 0,10,1,11,...,9,19

fig, ax = plt.subplots(5, 4, figsize=(10, 8), dpi=400, sharex=True)

for i, n in enumerate(order):
    r = i // 4   # row index
    c = i % 4    # column index

    ax[r, c].plot(
        time[seq_1:seq_2] * factor_ms,
        gaussian_filter1d(eps_xy_0[n][seq_1:seq_2] * factor_mV, sigma=10),
        lw=lw,
        color=colors[0],
        alpha=0.7
    )

    ax[r, c].grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    ax[r, c].set_title(f"Gauge {n}")

# Only label outer axes
for r in range(5):
    ax[r, 0].set_ylabel(r"$\varepsilon_{xy}$ [mV]")

for c in range(4):
    ax[-1, c].set_xlabel("time (ms)")

plt.tight_layout()
plt.savefig(os.path.abspath(os.path.join(daq_path, "gauge(t)_event" + str(loc_file[7:9]) + "_xy.png")))
plt.show()