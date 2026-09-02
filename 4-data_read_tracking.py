 # -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 15:14:30 2026

@author: Lukas

Grain Comparison Project - Data Loading Script

Load both DIC and CSV datasets into comparable data structures.
Ready for you to add your own analysis code.

No configuration files needed - everything in this single script.
"""
#%% LIBS and DATA load
# ============================================================
# 0. LIBRARIES
# ============================================================

import glob
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json

from scipy.stats import linregress

# ============================================================
# 1. Load FILES and create DATA-STRUCTURE
# ============================================================

base_dir = os.path.dirname(__file__)
DIC_DIRECTORY = os.path.abspath(os.path.join(base_dir, r"4-grain_tracking"))   # <-- change to your folder

def load_dic_file(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Skip malformed lines
            if len(parts) < 13:
                continue

            try:
                row = list(map(float, parts))
                data.append(row)
            except ValueError:
                # Skip any line that cannot be parsed
                continue
    return data


dic_files = sorted(
    glob.glob(os.path.join(DIC_DIRECTORY, "dic_out_*.txt")),
    key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split("_")[-1])
)

dic_data = {}
for i, dic_file in enumerate(dic_files):
    grains = load_dic_file(dic_file)
    dic_data[i] = grains

print(f"Loaded {len(dic_data)} frames.")


# ----
# 1.1 BUILD GRAIN IDs BASED ON (X0, Y0) Primary File
# ----

# Collect all unique grain IDs from the first frame
grain_ids = [(row[0], row[1]) for row in dic_data[0]]
print(f"Tracking {len(grain_ids)} grains.")


# ----
# 1.2 EXTRACT TRAJECTORIES FOR ALL GRAINS
# ----

trajectories = {}  # grain_id -> dict of arrays

for gid in grain_ids:
    X0, Y0 = gid

    # Storage arrays
    xs, ys = [], []
    dxs, dys = [], []
    dx0, dy0 = [], []
    thetas, dthetas = [], []
    radii = []
    ncc, ncc_rescue, ncc_subpix = [], [], []

    # Loop over frames
    for frame_idx, frame in dic_data.items():
        for row in frame:
            if row[0] == X0 and row[1] == Y0:

                # Extract all fields
                X0_ref, Y0_ref = row[0], row[1]
                theta0 = row[2]
                radius = row[3]
                DX, DY = row[4], row[5]
                Dtheta = row[6]
                dx, dy = row[7], row[8]
                dtheta = row[9]
                NCC = row[10]
                NCC_res = row[11]
                NCC_sub = row[12]

                # Append raw values
                dx0.append(DX)
                dy0.append(DY)
                dxs.append(dx)
                dys.append(dy)
                thetas.append(Dtheta)
                dthetas.append(dtheta)
                radii.append(radius)
                ncc.append(NCC)
                ncc_rescue.append(NCC_res)
                ncc_subpix.append(NCC_sub)   

                # Position over image series (temporary)
                xs.append(X0_ref + DX)
                ys.append(Y0_ref + DY)

                break
    # ----
    # 1.2.1 FIX REFERENCE POSITION OF PRIMARY TRAJECTORY from Subpixel adjustment (dic_0)
    # ----

    # dic_0 displacement correction
    DX0 = dxs[0]
    DY0 = dys[0]

    # Correct initial position
    xs = np.array(xs)
    ys = np.array(ys)

    xs -= DX0
    ys -= DY0

    # Set first displacement to zero
    dxs = np.array(dxs)
    dys = np.array(dys)

    dxs[0] = 0
    dys[0] = 0

    # ----
    # 1.2.2 Store everything
    # ----

    trajectories[gid] = {
        "x": np.array(xs),
        "y": np.array(ys),
        "dx": np.array(dxs),
        "dy": np.array(dys),
        "dx0": np.array(dx0),
        "dy0": np.array(dy0),
        "theta": np.array(thetas),
        "dtheta": np.array(dthetas),
        "radius": np.array(radii),
        "ncc": np.array(ncc),
        "ncc_rescue": np.array(ncc_rescue),
        "ncc_subpix": np.array(ncc_subpix)
    }


# ----
# 1.3 SAVE FULL TRAJECTORY DATA
# ---- 

output_path = "4-grain_tracking/trajectories_full.json"

# Convert NumPy arrays to lists for JSON
json_ready = {
    str(gid): {k: v.tolist() for k, v in traj.items()}
    for gid, traj in trajectories.items()
}

with open(output_path, "w") as f:
    json.dump(json_ready, f, indent=2)
    
#%% NUMERICAL ANALYSIS

# ============================================================
# 2. NUMERICAL ANALYSIS ACROSS ALL GRAINS
# ============================================================

grain_stats = {}   # store per-grain results

for gid in grain_ids:
    dx0 = np.array(trajectories[gid]["dx0"])
    dy0 = np.array(trajectories[gid]["dy0"])
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

stats_output_path = "4-grain_tracking/grain_stats.json"

with open(stats_output_path, "w") as f:
    json.dump(grain_stats, f, indent=2)

print(f"\nSaved grain statistics to: {stats_output_path}")

#%% PLOTS

# ============================================================
# PLOTS
# ============================================================

colors = [
    "#19717D",  # teal
    "#2B8901",  # green
    "#CC3311",  # orange-red (color-blind safe)
    "#0072B2"   # blue (color-blind safe)
]

# ---
# Quick insight
# ---

grain_nr = 0

plt.figure(figsize=(10,4), dpi=350)
for i in range(0,10):
    #plt.plot(trajectories[grain_ids[i]]["dx"][1:], label="x")
    plt.plot(trajectories[grain_ids[grain_nr]]["dx0"], label="x", color=colors[0])
    
plt.title(f"Displacement for Grain {grain_nr}")
plt.xlabel("Frames")
plt.ylabel(r"$\Delta x_0$")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
#%%

# ----
# Absolute DISPLACEMENT
# ----

abs_dx_all = []
abs_dy_all = []
mag_all = []

for gid in grain_ids:
    dx = np.array(trajectories[gid]["dx0"])   # now absolute displacement
    dy = np.array(trajectories[gid]["dy0"])

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
plt.grid(True, linestyle='--', alpha=0.5)
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
    dx = trajectories[gid]["dx"][seq1:seq2]   # now absolute displacement
    dy = trajectories[gid]["dy"][seq1:seq2]

    abs_dx_all.append(np.abs(dx))
    abs_dy_all.append(np.abs(dy))
    mag_all.append(np.sqrt(dx**2 + dy**2))

abs_dx_all = np.array(abs_dx_all)
abs_dy_all = np.array(abs_dy_all)
disp_mag_all = np.array(mag_all)

log_mag = np.log10(disp_mag_all + eps)

plt.figure(figsize=(10,6), dpi=350)
im = plt.imshow(log_mag, aspect='auto', cmap='viridis')
plt.colorbar(im, label=r"log10($\sqrt{\Delta x^2 + \Delta y^2}$) [px]")
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
dx0 = trajectories[gid]["dx0"]
dy0 = trajectories[gid]["dy0"]
frames = np.arange(len(dx0))

# Recompute fit for this grain
slope_dx = grain_stats[str(gid)]["slope_dx"]
intercept_dx = dx0[0] - slope_dx * frames[0]
fit_dx = slope_dx * frames + intercept_dx

slope_dy = grain_stats[str(gid)]["slope_dy"]
intercept_dy = dy0[0] - slope_dy * frames[0]
fit_dy = slope_dy * frames + intercept_dy

# --- Plot dx0 + fit ---
plt.figure(figsize=(10,4), dpi=300)
plt.plot(dx0[seq1:seq2], label=r"$\Delta x_0$", linewidth=.95, color=colors[0])
plt.plot(fit_dx[seq1:seq2], '--', label=r"Linear fit $\Delta x_0$", color=colors[1])
plt.title(rf"$\Delta x_0$ for Grain {grain_choice}")
plt.xlabel("Frame")
plt.ylabel(r"$\Delta x_0$")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# --- Plot dy0 + fit ---
plt.figure(figsize=(10,4), dpi=300)
plt.plot(dy0[seq1:seq2], label=r"$\Delta y_0$", linewidth=.95, color=colors[0])
plt.plot(fit_dy[seq1:seq2], '--', label=r"Linear fit $\Delta y_0$", color=colors[1])
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
plt.title(f"Residuals for Grain {grain_choice}")
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
    x0 = trajectories[gid]["x"][0]
    y0 = trajectories[gid]["y"][0]

    color = cmap(norm(r2_dx_all[i]))
    circ = Circle((x0, y0), radius=6, edgecolor=color, facecolor=color, alpha=0.9)
    ax.add_patch(circ)

ax.set_title("Spatial Quality Map: R²(dx₀)")
ax.set_axis_off()

# Create horizontal colorbar BELOW the image
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    orientation="horizontal",
    pad=0.03,          # distance below the image
    fraction=0.04      # thickness of the bar
)
cbar.set_label("R²(dx₀)")

plt.tight_layout()
plt.show()