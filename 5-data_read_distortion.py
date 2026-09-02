# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:18:15 2026

@author: Lukas Wimmer
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
# Read correction model
file_path = os.path.abspath(os.path.join(base_dir, r"5-correct_distortion\corrDisto_macro.log"))
# Read new coordinates
file_coords = os.path.abspath(os.path.join(base_dir, r"5-correct_distortion\trajectories_corrected.json"))

params = {}

with open(file_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0]
        val = parts[-1]
        try:
            params[key] = float(val)
        except ValueError:
            pass

# Extract parameters
x_c = params.get("xc_corrDistor")
y_c = params.get("yc_corrDistor")

K1 = params.get("K1_corrDistor")
K2 = params.get("K2_corrDistor")
K3 = params.get("K3_corrDistor")

P1 = params.get("P1_corrDistor")
P2 = params.get("P2_corrDistor")
P3 = params.get("P3_corrDistor", 0.0)   # optional

print("Loaded distortion parameters:")
for k, v in params.items():
    print(f"{k:20s} = {v}")
    
# ---------------------------------------------------------
# Model visualization
# ---------------------------------------------------------

# Image size
W = 1280
H = 800

# Pixel grid (distorted coords)
x_d = np.arange(W)
y_d = np.arange(H)
xx_d, yy_d = np.meshgrid(x_d, y_d)

# Centered coordinates
X = xx_d - x_c
Y = yy_d - y_c

r2 = X**2 + Y**2
r4 = r2**2
r6 = r2**3

# Undistorted coordinates (analytical)
radial = K1*r2 + K2*r4 + K3*r6

x_u = xx_d + X*radial + P1*(r2 + 2*X**2) + 2*P2*X*Y*(1 + P3*r2)
y_u = yy_d + Y*radial + 2*P1*X*Y + P2*(r2 + 2*Y**2)*(1 + P3*r2)

# Correction (difference)
dx = x_u - xx_d
dy = y_u - yy_d
mag = np.sqrt(dx**2 + dy**2)

# Heatmap of correction magnitude
plt.figure(figsize=(10, 6), dpi=300)
im = plt.imshow(mag, cmap="inferno", origin="upper")
plt.colorbar(im, label="Correction magnitude (pixels)")
plt.title("Analytical Brown–Conrady Undistortion Magnitude (1280×800)")
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.show()

#%% Load Data and Analysis
# ---------------------------------------------------------
# Load corrected trajectories
# ---------------------------------------------------------

with open(file_coords, "r") as f:
    trajectories = json.load(f)

# Convert lists back to numpy arrays
for gid, traj in trajectories.items():
    for key, val in traj.items():
        trajectories[gid][key] = np.array(val)

# Convert JSON keys "(x,y)" → tuples
grain_ids = [eval(gid) for gid in trajectories.keys()]

# ----
# NUMERICAL ANALYSIS ACROSS ALL GRAINS
# ----

grain_stats = {}   # store per-grain results

for gid in grain_ids:
    gid_str = str(gid) 
    dx0 = np.array(trajectories[gid_str]["dx0"])
    dy0 = np.array(trajectories[gid_str]["dy0"])
    frames = np.arange(len(dx0))

    # --- Noise / variability ---
    std_dx = float(np.std(dx0))
    std_dy = float(np.std(dy0))

    # --- Linear regression (trend + speed) ---
    slope_dx, intercept_dx, r_dx, _, _ = linregress(frames, dx0)
    slope_dy, intercept_dy, r_dy, _, _ = linregress(frames, dy0)

    # --- Residuals (deviation from linearity) ---
    fit_dx = slope_dx * frames + intercept_dx
    fit_dy = slope_dy * frames + intercept_dy

    res_dx = dx0 - fit_dx
    res_dy = dy0 - fit_dy

    res_std_dx = float(np.std(res_dx))
    res_std_dy = float(np.std(res_dy))

    # Store everything (convert to Python floats)
    grain_stats[str(gid)] = {
        "std_dx": std_dx,
        "std_dy": std_dy,
        "slope_dx": float(slope_dx),
        "slope_dy": float(slope_dy),
        "r2_dx": float(r_dx**2),
        "r2_dy": float(r_dy**2),
        "res_std_dx": res_std_dx,
        "res_std_dy": res_std_dy
    }

# ----
#  2.1 AVERAGE RESULTS ACROSS ALL GRAINS
# ----

all_std_dx     = np.array([grain_stats[str(g)]["std_dx"] for g in grain_ids])
all_std_dy     = np.array([grain_stats[str(g)]["std_dy"] for g in grain_ids])
all_slope_dx   = np.array([grain_stats[str(g)]["slope_dx"] for g in grain_ids])
all_slope_dy   = np.array([grain_stats[str(g)]["slope_dy"] for g in grain_ids])
all_r2_dx      = np.array([grain_stats[str(g)]["r2_dx"] for g in grain_ids])
all_r2_dy      = np.array([grain_stats[str(g)]["r2_dy"] for g in grain_ids])
all_res_std_dx = np.array([grain_stats[str(g)]["res_std_dx"] for g in grain_ids])
all_res_std_dy = np.array([grain_stats[str(g)]["res_std_dy"] for g in grain_ids])

print("\n=== Average Results Across All Grains ===")
print(f"Mean STD(dx0):       {np.mean(all_std_dx):.6e}")
print(f"Mean STD(dy0):       {np.mean(all_std_dy):.6e}")
print(f"Mean slope dx0:      {np.mean(all_slope_dx):.6e} px/frame")
print(f"Mean slope dy0:      {np.mean(all_slope_dy):.6e} px/frame")
print(f"Mean R² dx0:         {np.mean(all_r2_dx):.6f}")
print(f"Mean R² dy0:         {np.mean(all_r2_dy):.6f}")
print(f"Mean residual STD dx {np.mean(all_res_std_dx):.6e}")
print(f"Mean residual STD dy {np.mean(all_res_std_dy):.6e}")

# ----
# SAVE GRAIN STATS
# ----

stats_output_path = "5-correct_distortion/grain_stats_corrected.json"

with open(stats_output_path, "w") as f:
    json.dump(grain_stats, f, indent=2)



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

grain_nr = 0
gid_str = str(grain_ids[grain_nr])

plt.figure(figsize=(10,4), dpi=350)

s1, s2 = -1, 0

# Example: plot corrected dx0 for grain 0
plt.plot( trajectories[gid_str]["dy0"], label="dy0 corrected", color=colors[0])

plt.title(f"Corrected Displacement for Grain {grain_nr}")
plt.xlabel("Frames")
plt.ylabel(r"$\Delta x_0$")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# ----
# Absolute DISPLACEMENT
# ----

abs_dx_all = []
abs_dy_all = []
mag_all = []

for gid in grain_ids:
    gid_str = str(gid)   # JSON keys are strings
    dx = np.array(trajectories[gid_str]["dx0"])   # now absolute displacement
    dy = np.array(trajectories[gid_str]["dy0"])

    abs_dx_all.append(np.abs(dx))
    abs_dy_all.append(np.abs(dy))

    mag_all.append(np.sqrt(dx**2 + dy**2))

abs_dx_all = np.array(abs_dx_all)
abs_dy_all = np.array(abs_dy_all)
mag_all   = np.array(mag_all)

# ---- Plot mean displacement over all grains ----
mean_abs_dx = np.mean(abs_dx_all, axis=0)
mean_abs_dy = np.mean(abs_dy_all, axis=0)
mean_mag    = np.mean(mag_all, axis=0)

plt.figure(figsize=(10,5), dpi=350)
plt.plot(mean_abs_dx, label=r'mean $|\Delta x|$', color=colors[0])
plt.plot(mean_abs_dy, label=r'mean $|\Delta y|$', color=colors[1])
plt.plot(mean_mag, label=r'mean $\sqrt{\Delta x^2 + \Delta y^2}$', color=colors[2])

plt.xlabel("Frame")
plt.ylabel("Displacement [px]")
plt.title("Mean Absolute Displacement Across All Grains")
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.ylim((0,0.05))
plt.show()
#%%

# ----
# LOG-SCALED DISPLACEMENT HEATMAP
# ----

# To avoid log(0) by adding a tiny epsilon
eps = 1e-6

abs_dx_all = []
abs_dy_all = []
mag_all = []

seq1, seq2 = 0, -1

for gid in grain_ids:
    gid_str = str(gid)   # JSON keys are strings
    dx = trajectories[gid_str]["dx"][seq1:seq2]   # now absolute displacement
    dy = trajectories[gid_str]["dy"][seq1:seq2]

    abs_dx_all.append(np.abs(dx))
    abs_dy_all.append(np.abs(dy))
    mag_all.append(np.sqrt(dx**2 + dy**2))

abs_dx_all = np.array(abs_dx_all)
abs_dy_all = np.array(abs_dy_all)
disp_mag_all = np.array(mag_all)

log_mag = np.log10(disp_mag_all + eps)

plt.figure(figsize=(10,6), dpi=350)
im = plt.imshow(log_mag, aspect='auto', cmap='viridis')
plt.colorbar(im, label="log10(√(Δx² + Δy²)) [px]")
plt.xlabel("Frame")
plt.ylabel("Grain ID")
plt.title("Log-Scaled Displacement Magnitude Heatmap")
plt.tight_layout()
plt.show()

#%%

# ----
#  SELECT A SPECIFIC GRAIN FOR DETAILED PLOTS
# ----

grain_choice = 10

seq1, seq2 = 0, -1

gid = grain_ids[grain_choice]
gid_str = str(gid)
dx0 = trajectories[gid_str]["dx0"]
dy0 = trajectories[gid_str]["dy0"]
frames = np.arange(len(dx0))

# Recompute fit for this grain
slope_dx = grain_stats[gid_str]["slope_dx"]
intercept_dx = dx0[0] - slope_dx * frames[0]
fit_dx = slope_dx * frames + intercept_dx

slope_dy = grain_stats[gid_str]["slope_dy"]
intercept_dy = dy0[0] - slope_dy * frames[0]
fit_dy = slope_dy * frames + intercept_dy

# --- Plot dx0 + fit + theoretical ---
plt.figure(figsize=(10,4), dpi=300)
plt.plot(dx0[seq1:seq2], label=r"$\Delta x_0$ corrected", linewidth=.95, color=colors[0])
plt.plot(fit_dx[seq1:seq2], '--', label=r"Regression fit $\Delta x_0$", color=colors[1])
plt.title(rf"$\Delta x_0$ for Grain {grain_choice}")
plt.xlabel("Frame")
plt.ylabel(r"$\Delta x_0$")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# --- Plot dy0 + fit + theoretical ---
plt.figure(figsize=(10,4), dpi=300)
plt.plot(dy0[seq1:seq2], label=r"$\Delta y_0$ corrected", linewidth=.95, color=colors[0])
plt.plot(fit_dy[seq1:seq2], '--', label=r"Regression fit $\Delta y_0$", color=colors[1])
plt.title(rf"$\Delta y_0$ for Grain {grain_choice}")
plt.xlabel("Frame")
plt.ylabel(r"$\Delta y_0$")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# --- Residuals ---
res_dx = dx0 - fit_dx
res_dy = dy0 - fit_dy

plt.figure(figsize=(10,4), dpi=300)
plt.plot(res_dx[seq1:seq2], label=r"Residual $\Delta x_0$", linewidth=.95, color=colors[0])
plt.plot(res_dy[seq1:seq2], label=r"Residual $\Delta y_0$", linewidth=.75, color=colors[2])
plt.title(f"Residuals for Grain {grain_choice} after Cooridante correction")
plt.xlabel("Frame")
plt.ylabel("Residual")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
#%%

# ----
#  SPATIAL HEATMAP OF R²(dx0) OVERLAID ON ORIGINAL IMAGE
# ----

import os
import tifffile as tiff
from matplotlib.patches import Circle

# Absolute path
base_dir = os.path.dirname(__file__)

# Read TIFF image
img_path = os.path.abspath(os.path.join(base_dir, r"1-raw_data/frame_00000.tif"))
image = tiff.imread(img_path).astype(np.uint16)

# Extract R²(dx0) values
r2_dx_all = np.array([grain_stats[str(g)]["r2_dx"] for g in grain_ids])

# Normalize range: min → 1.0
r2_min = np.min(r2_dx_all)
r2_max = 1.0

# Prepare colormap
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=r2_min, vmax=r2_max)

# Create figure
fig, ax = plt.subplots(figsize=(8, 8), dpi=350)

# Show original image
ax.imshow(image, cmap="gray")

# Plot each grain at its initial position
for i, gid in enumerate(grain_ids):
    gid_str = str(gid)   # JSON keys are strings
    x0 = trajectories[gid_str]["x"][0]
    y0 = trajectories[gid_str]["y"][0]

    color = cmap(norm(r2_dx_all[i]))
    circ = Circle((x0, y0), radius=6, edgecolor=color, facecolor=color, alpha=0.9)
    ax.add_patch(circ)

ax.set_title(r"Spatial Quality Map: R²($\Delta x_0$)")
ax.set_axis_off()

# Create horizontal colorbar BELOW the image
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    orientation="horizontal",
    pad=0.03,          # distance below the image
    fraction=0.04      # thickness of the bar
)
cbar.set_label(r"R²($\Delta x_0$)")

plt.tight_layout()
plt.show()