# -*- coding: utf-8 -*-
"""
Improved version:
1. Manual reordering BEFORE filtering
2. Display image with NEW grain IDs
3. Then filtering step
"""

import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os
from scipy.ndimage import gaussian_filter

# Absolute path
base_dir = os.path.dirname(__file__)

# Read TIFF image
img_path = os.path.abspath(os.path.join(base_dir, r"1-raw_data/frame_00000.tif"))
image = tiff.imread(img_path).astype(np.uint16)

if image.ndim == 3:
    image = image[0, :, :]

sigma = 3.0
smoothed = gaussian_filter(image, sigma=sigma)

filename = os.path.abspath(os.path.join(base_dir, r"2-find_grains/grains.txt"))

# ---- load grain file ----
with open(filename, "r") as f:
    n = int(f.readline().strip())
    data = np.loadtxt(f)
    
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8), dpi=1200)
ax.imshow(image, cmap="gray")

for i in range(n):
    x, y, rot, r = data[i]
    circle = Circle((x, y), r, edgecolor="red", facecolor="none", linewidth=.3)
    ax.add_patch(circle)
    ax.annotate(str(i), (x, y), color="red", fontsize=5, ha="center", va="center")

ax.set_title("Original Grain IDs", size=6)
ax.set_axis_off()
plt.show()

print("\nLoaded", n, "grains.")
#%%
# ============================================================
#  MANUAL REORDERING STEP 
# ============================================================

def parse_index_list(text):
    """Parses input like '5, 10-12' into a set of integers."""
    if not text:
        return set()
    out = set()
    parts = text.split(',')
    for p in parts:
        p = p.strip()
        if '-' in p:
            a, b = p.split('-')
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(p))
    return out

print("\nOPTIONAL: Enter grain indices to place at the TOP (in this order).")
top_input = input("Top fixed grains: ").strip()
top_fixed = parse_index_list(top_input)

print("\nOPTIONAL: Enter grain indices to place at the BOTTOM (in this order).")
bottom_input = input("Bottom fixed grains: ").strip()
bottom_fixed = parse_index_list(bottom_input)

# Ensure valid indices
top_fixed = [i for i in top_fixed if i < n]
bottom_fixed = [i for i in bottom_fixed if i < n]

# Middle grains = all remaining grains not in top or bottom
middle = [i for i in range(n) if i not in top_fixed and i not in bottom_fixed]

# Final order BEFORE filtering
final_order = top_fixed + middle + bottom_fixed

print("\nFinal grain order BEFORE filtering:")
print(final_order)

# Reorder data
reordered_data = data[final_order]

# ============================================================
#  DISPLAY IMAGE WITH NEW GRAIN IDS
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8), dpi=1200)
ax.imshow(image, cmap="gray")

for new_id, row in enumerate(reordered_data):
    x, y, rot, r = row
    circle = Circle((x, y), r, edgecolor="yellow", facecolor="none", linewidth=.3)
    ax.add_patch(circle)
    ax.annotate(str(new_id), (x, y), color="yellow", fontsize=5, ha="center", va="center")

ax.set_title("Reordered Grain IDs", size=6)
ax.set_axis_off()
plt.show()
#%%
# ============================================================
#  FILTERING STEP (NOW AFTER REORDER)
# ============================================================

print("\nEnter grain indices to EXCLUDE (wrong detections).")
exclude_input = input("Exclude: ").strip()
exclude_indices = parse_index_list(exclude_input)

print("\nExcluding:", sorted(exclude_indices) if exclude_indices else "None")

# Filter based on NEW IDs
kept_indices = [i for i in range(len(reordered_data)) if i not in exclude_indices]
filtered_data = reordered_data[kept_indices]

print("\nRemaining grains after filtering:", len(filtered_data))

# ============================================================
#  SAVE RESULT
# ============================================================

output_filename = os.path.abspath(os.path.join(base_dir, r"2-find_grains/grains.txt"))

with open(output_filename, "w") as f:
    f.write(f"{len(filtered_data)}\n")
    for row in filtered_data:
        f.write(f"{int(row[0])}\t{int(row[1])}\t{int(row[2])}\t{int(row[3])}\n")

print("\nFinal reordered + filtered grains saved to:", output_filename)
print("="*60)
