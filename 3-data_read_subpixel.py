# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:51:30 2026

@author: Lukas

Modified to include filtering of incorrectly detected grains
"""

import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import sys
import os

from scipy.ndimage import gaussian_filter

# Absolute path
base_dir = os.path.dirname(__file__)

# Read TIFF image

img_path = os.path.abspath(os.path.join(base_dir, r"1-raw_data\frame_00000.tif"))
image = tiff.imread(img_path).astype(np.uint16)

if image.ndim == 3:
    image = image[0,:,:]

img_threshold_8bit = 112
threshold_16bit = img_threshold_8bit * 257

sigma = 0.0  # adjust this (higher = more blur)
smoothed = gaussian_filter(image, sigma=sigma)

filename = os.path.abspath(os.path.join(base_dir, r"3-find_grains_subpixel\recentered.data"))

# ---- load grain file ----
with open(filename, "r") as f:
    n = int(f.readline().strip())
    data = np.loadtxt(f)

grains = {
    "x": data[:,0],
    "y": data[:,1],
    "rotation": data[:,2],
    "radius": data[:,3]
}

# Display image
# ---- plot image ----
fig, ax = plt.subplots(figsize=(8, 8), dpi=1200)
ax.imshow(smoothed, cmap="gray")

# ---- draw circles ----
for i in range(len(data[:,0])):
    x = data[i, 0]

    y = data[i, 1]
    r = data[i, 3]

    circle = Circle((x, y), r, edgecolor="red", facecolor="none", linewidth=.3)
    ax.add_patch(circle)

    ax.annotate(
        str(i),           # label
        (x, y-10),           # position
        color="red",
        fontsize=2,
        ha="center",
        va="center"
    )

# ---- final styling ----
ax.set_title("Subpixel adjustment: " + img_path, size=6)
ax.set_axis_off()
plt.show()

#%%
# --- Load profile.txt ---
filename = os.path.abspath(os.path.join(base_dir, r"3-find_grains_subpixel\profile.txt"))
data = []
with open(filename, "r") as f:
    for line in f:
        line = line.strip()
        if line == "":
            data.append(None)   # separator between the 3 profiles
        else:
            x, val = line.split()
            data.append((int(x), float(val)))

# --- Split into three profiles ---
profiles = []
current = []
for item in data:
    if item is None:
        if current:
            profiles.append(np.array(current))
            current = []
    else:
        current.append(item)
if current:
    profiles.append(np.array(current))

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, prof in enumerate(profiles):
    xs = prof[:,0]
    ys = prof[:,1]
    axes[i].plot(xs, ys, lw=1)
    axes[i].set_title(f"Intensity profile at line {i+1}")
    axes[i].set_ylabel("Intensity")

axes[-1].set_xlabel("Pixel index")

# --- Suggest threshold ---
# Combine all intensity values
all_vals = np.concatenate([p[:,1] for p in profiles])

# Background = lower 10% percentile
bg = np.percentile(all_vals, 10)

# Bead = upper 10% percentile
bead = np.percentile(all_vals, 90)

# Suggested threshold halfway between background and bead
threshold = (bg + bead) / 2

print(f"Suggested threshold ≈ {threshold:.1f}")

plt.tight_layout()
plt.show()

