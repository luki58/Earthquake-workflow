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
file_coords = os.path.abspath(os.path.join(base_dir, r"5-correct_distortion/trajectories_corrected.json"))
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

# Collect all dx0 curves
all_dx0 = []
i = 0

for gid_str, traj in trajectories.items():
    dx0 = np.array(traj["dx0"])
    dy0 = np.array(traj["dy0"])

    # Convert to arrays
    result = np.sqrt(dx0**2 + dy0**2)

    # Plot
    plt.figure(dpi=300)
    
    plt.plot(result, color=colors[1], linewidth=.9)
    
    plt.title("Grain: " + str(i))
    plt.xlabel("Frames")
    plt.ylabel("Speed [px/frame]")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    i+=1

#%%

# ------------------------------------------------------------
# Compute representative x-position for each grain
# ------------------------------------------------------------

grain_positions = []   # list of (grain_id, mean_x)

for gid_str, traj in trajectories.items():
    # If you have absolute x positions:
    if "x" in traj:
        x = np.array(traj["x"])
    else:
        # If only dx0 exists, reconstruct x from cumulative displacement
        dx0 = np.array(traj["dx0"])
        x = np.cumsum(dx0)

    mean_x = np.mean(x)
    grain_positions.append((gid_str, mean_x))

# ------------------------------------------------------------
# Sort grains by x-position (bottom → top)
# ------------------------------------------------------------

grain_positions_sorted = sorted(grain_positions, key=lambda t: t[1])

# Extract sorted grain IDs
sorted_grain_ids = [gid for gid, _ in grain_positions_sorted]


# ------------------------------------------------------------
# Reorder trajectories dictionary
# ------------------------------------------------------------

trajectories_sorted = {gid: trajectories[gid] for gid in sorted_grain_ids}

# ------------------------------------------------------------
# Build matrix: rows = grains, columns = frames
# ------------------------------------------------------------

heatmap_data = []

for gid_str, traj in trajectories_sorted.items():
    dx0 = np.array(traj["dx0"])
    dy0 = np.array(traj["dy0"])
    speed = np.sqrt(dx0**2 + dy0**2)
    heatmap_data.append(speed)

heatmap_data = np.array(heatmap_data)   # shape: (num_grains, num_frames)

# ------------------------------------------------------------
# Plot heatmap
# ------------------------------------------------------------

plt.figure(figsize=(12, 8), dpi=300)

plt.imshow(
    heatmap_data,
    aspect='auto',
    cmap='inferno',     # great for scientific heatmaps
    origin='lower'      # grain 0 at bottom
)

plt.colorbar(label="Speed [px/frame]")

plt.xlabel("Frame index")
plt.ylabel("Grain ID with increasing position in x")
plt.title("2D Heatmap of Particle Speeds (sqrt(dx0² + dy0²))")

plt.tight_layout()
plt.show()