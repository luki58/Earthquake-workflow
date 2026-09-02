#!/bin/bash

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PARAM_FILE="corrDisto_macro.log"
INPUT_JSON="../4-grain_tracking/trajectories_full.json"
OUTPUT_JSON="trajectories_corrected.json"

# ------------------------------------------------------------
# Check if trajectories_full.json exists
# ------------------------------------------------------------
if [ ! -f "$INPUT_JSON" ]; then
    echo "ERROR: trajectories_full.json does not exist."
    echo "You must create it first in step 4 before running distortion correction."
    exit 1
fi

# ------------------------------------------------------------
# Run Python distortion correction
# ------------------------------------------------------------
python3 << 'EOF'
import json
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------------
# Load distortion parameters
# ------------------------------------------------------------
params = {}
with open("corrDisto_macro.log") as f:
    for line in f:
        parts = line.split()
        if len(parts) == 2:
            key, val = parts
            try:
                params[key] = float(val)
            except:
                pass

xc = params["xc_corrDistor"]
yc = params["yc_corrDistor"]
K1 = params["K1_corrDistor"]
K2 = params["K2_corrDistor"]
K3 = params["K3_corrDistor"]
P1 = params["P1_corrDistor"]
P2 = params["P2_corrDistor"]
P3 = params["P3_corrDistor"]

# ------------------------------------------------------------
# Distortion correction function
# ------------------------------------------------------------
def undistort_point(x, y):
    X = x - xc
    Y = y - yc

    r2 = X**2 + Y**2
    r4 = r2**2
    r6 = r2**3

    radial = K1*r2 + K2*r4 + K3*r6

    x_u = x + X*radial + P1*(r2 + 2*X**2) + 2*P2*X*Y*(1 + P3*r2)
    y_u = y + Y*radial + 2*P1*X*Y + P2*(r2 + 2*Y**2)*(1 + P3*r2)

    return x_u, y_u

# ------------------------------------------------------------
# Load trajectories
# ------------------------------------------------------------
with open("../4-grain_tracking/trajectories_full.json") as f:
    data = json.load(f)

# ------------------------------------------------------------
# Apply correction to all grains
# ------------------------------------------------------------
corrected = {}

for gid, traj in tqdm(data.items(), desc="Correcting grains"):

    # Distorted positions
    x_d = np.array(traj["x"])
    y_d = np.array(traj["y"])
    n = len(x_d)

    # Undistort all positions
    x_u = np.zeros_like(x_d)
    y_u = np.zeros_like(y_d)

    for i in range(n):
        x_u[i], y_u[i] = undistort_point(x_d[i], y_d[i])

    # Allocate arrays
    dx_u  = np.zeros(n)
    dy_u  = np.zeros(n)
    dx0_u = np.zeros(n)
    dy0_u = np.zeros(n)

    # Compute dx, dy, dx0, dy0
    for i in range(n):
        if i > 0:
            dx_u[i] = x_u[i] - x_u[i-1]
            dy_u[i] = y_u[i] - y_u[i-1]
        else:
            dx_u[0] = 0
            dy_u[0] = 0

        dx0_u[i] = x_u[i] - x_u[0]
        dy0_u[i] = y_u[i] - y_u[0]

    # Build corrected grain with same structure
    new_traj = {}
    for key in traj.keys():
        if key == "x":
            new_traj["x"] = x_u.tolist()
        elif key == "y":
            new_traj["y"] = y_u.tolist()
        elif key == "dx":
            new_traj["dx"] = dx_u.tolist()
        elif key == "dy":
            new_traj["dy"] = dy_u.tolist()
        elif key == "dx0":
            new_traj["dx0"] = dx0_u.tolist()
        elif key == "dy0":
            new_traj["dy0"] = dy0_u.tolist()
        else:
            new_traj[key] = traj[key]

    corrected[gid] = new_traj

# ------------------------------------------------------------
# Save corrected JSON
# ------------------------------------------------------------
with open("trajectories_corrected.json", "w") as f:
    json.dump(corrected, f, indent=2)

EOF

echo "Applied successfully, corrected file: trajectories_corrected.json"