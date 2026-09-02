# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:37:29 2026

@author: admin
"""

import os
import tifffile as tiff
import numpy as np

# -----------------------------
# INPUT / OUTPUT
# -----------------------------
input_tif = r"..\img_dat\event_3.tif"
output_folder = r"1-raw_data"

os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# PADDING SETTINGS
# -----------------------------
padding = 30   # adjustable padding in pixels

# -----------------------------
# LOAD STACKED TIFF
# -----------------------------
stack = tiff.imread(input_tif)
print(f"Loaded stack with shape: {stack.shape}")

# -----------------------------
# SAVE EACH FRAME WITH PADDING
# -----------------------------
for i, frame in enumerate(stack):

    # original dimensions
    h, w = frame.shape

    # padded dimensions
    new_h = h + 2 * padding
    new_w = w + 2 * padding

    # create padded frame (black border)
    padded = np.zeros((new_h, new_w), dtype=frame.dtype)

    # insert original frame into center
    padded[padding:padding + h, padding:padding + w] = frame

    # save
    out_path = os.path.join(output_folder, f"frame_{i:05d}.tif")
    tiff.imwrite(out_path, padded)

print(f"Saved {len(stack)} padded frames to: {output_folder}")
