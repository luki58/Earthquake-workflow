# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:56:09 2026

@author: Lukas
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import json

from scipy.stats import linregress

# ---------------------------------------------------------
# File Paths
# ---------------------------------------------------------

# Absolute path
base_dir = os.path.dirname(__file__)
# File paths
file_coords = os.path.abspath(os.path.join(base_dir, r"4-grain_tracking/trajectories_full.json"))
file_coords_corrected = os.path.abspath(os.path.join(base_dir, r"5-correct_distortion/trajectories_corrected.json"))
file_stats = os.path.abspath(os.path.join(base_dir, r"4-grain_tracking/grain_stats.json"))
file_stats_corrected = os.path.abspath(os.path.join(base_dir, r"5-correct_distortion/grain_stats_corrected.json"))
#%% Load Data and Analysis
# -----
# Load trajectories
# -----

with open(file_coords, "r") as f:
    trajectories = json.load(f)

for gid, traj in trajectories.items():
    for key, val in traj.items():
        trajectories[gid][key] = np.array(val)

grain_ids = [eval(gid) for gid in trajectories.keys()]

with open(file_coords_corrected, "r") as f:
    trajectories_corrected = json.load(f)

for gid, traj in trajectories_corrected.items():
    for key, val in traj.items():
        trajectories_corrected[gid][key] = np.array(val)

grain_ids_corrected = [eval(gid) for gid in trajectories_corrected.keys()]

# -----
# Load stats
# -----

with open(file_stats, "r") as f:
    grain_stats = json.load(f)
    
grain_stats = {eval(gid): stats for gid, stats in grain_stats.items()}

with open(file_stats_corrected, "r") as f:
    grain_stats_corrected = json.load(f)

grain_stats_corrected = {eval(gid): stats for gid, stats in grain_stats_corrected.items()}

# Extract slopes
slopes_dx      = np.array([grain_stats[gid]["slope_dx"] for gid in grain_ids])
slopes_dy      = np.array([grain_stats[gid]["slope_dy"] for gid in grain_ids])

slopes_dx_corr = np.array([grain_stats_corrected[gid]["slope_dx"] for gid in grain_ids_corrected])
slopes_dy_corr = np.array([grain_stats_corrected[gid]["slope_dy"] for gid in grain_ids_corrected])

# Compute averages
mean_slope_dx      = np.mean(slopes_dx)
mean_slope_dy      = np.mean(slopes_dy)
mean_slope_dx_corr = np.mean(slopes_dx_corr)
mean_slope_dy_corr = np.mean(slopes_dy_corr)

#%% Corrected DATA Plots

# ============================================================
# Corrected DATA Plots
# ============================================================

colors = [
    "#19717D",  # teal
    "#2B8901",  # green
    "#CC3311",  # orange-red (color-blind safe)
    "#0072B2"   # blue (color-blind safe)
]

grain_nr = 40
gid_str = str(grain_ids[grain_nr])

plt.figure(figsize=(10,4), dpi=500)

s1, s2 = -1, 0

# Example: plot corrected dx0 for grain 0
plt.plot(trajectories[gid_str]["dx"], label=r"$\Delta x$", color=colors[0], linewidth=.75)
plt.plot(trajectories_corrected[gid_str]["dx"], label=r"$\Delta x^{corr}$", color=colors[1], linewidth=.75)

plt.title(f"Corrected Displacement for Grain {grain_nr}")
plt.xlabel("Frames")
plt.ylabel(r"$\Delta x$")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
#%%

# Collect all dx0 curves
all_dx0 = []
all_dx0_corr = []

for gid_str, traj in trajectories.items():
    dx0 = np.array(traj["dx0"])
    dx0_corr = np.array(trajectories_corrected[gid_str]["dx0"])
    
    all_dx0.append(dx0)
    all_dx0_corr.append(dx0_corr)

# Convert to arrays
all_dx0 = np.array(all_dx0)
all_dx0_corr = np.array(all_dx0_corr)

# Compute mean over grains (axis=0 = frame axis)
mean_dx0 = np.mean(all_dx0, axis=0)
mean_dx0_corr = np.mean(all_dx0_corr, axis=0)

# Compute correct slope via linear regression
frames = np.arange(len(mean_dx0))

slope_mean_dx, intercept_mean_dx, _, _, _ = linregress(frames, mean_dx0)
fit_mean_dx = slope_mean_dx * frames + intercept_mean_dx

slope_mean_dx_corr, intercept_mean_dx_corr, _, _, _ = linregress(frames, mean_dx0_corr)
fit_mean_dx_corr = slope_mean_dx_corr * frames + intercept_mean_dx_corr

# Plot
plt.figure(figsize=(10,4), dpi=500)

s1, s2 = 1000, 1500

# Mean curves
plt.plot(mean_dx0[s1:s2], label=r"Mean $\Delta x_0$", color=colors[0], linewidth=.9)
plt.plot(mean_dx0_corr[s1:s2], label=r"Mean $\Delta x_0^{corr}$", color=colors[1], linewidth=.9)

# Correct slope lines (dashed)
plt.plot(fit_mean_dx[s1:s2], '--', color=colors[0], linewidth=1.4,
         label=rf"Slope = {slope_mean_dx:.3e} px/frame")

plt.plot(fit_mean_dx_corr[s1:s2], '--', color=colors[1], linewidth=1.4,
         label=rf"Slope corr = {slope_mean_dx_corr:.3e} px/frame")

plt.title("Mean Displacement and Correct Linear Slope")
plt.xlabel("Frames")
plt.ylabel(r"$\Delta x_0$")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
