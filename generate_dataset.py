"""
generate_dataset.py
--------------------
Creates a synthetic training dataset for the shaft-bearing assembly
tolerance-optimization project.

Each row = one manufactured component that goes into a shaft-bearing
assembly. We simulate realistic engineering relationships:

  - Tighter tolerance  -> higher manufacturing cost
  - Grinding is more precise but costlier than Turning/Milling/Drilling
  - Each part has an allowable tolerance band depending on its Fit_Type
  - Status = PASS if the part's tolerance (plus small random process
    variation) stays inside the allowable band for its assembly role,
    otherwise FAIL

This gives the ML model a realistic, learnable pattern instead of a
pure lookup table, which is what makes it worth using ML at all.

Run this once to (re)create the training dataset:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 500

part_names = ["Shaft", "Bearing_Bore", "Housing", "Retaining_Ring", "Spacer"]
processes = ["Turning", "Grinding", "Milling", "Drilling"]
materials = ["Steel", "Aluminum", "Brass", "Cast_Iron"]
fit_types = ["Clearance", "Transition", "Interference"]

# Base cost multiplier per process (grinding = most precise & most costly)
process_cost_factor = {"Turning": 1.0, "Milling": 1.15, "Drilling": 0.9, "Grinding": 1.8}

# Achievable tolerance range (mm) per process - grinding can hold tighter tolerances
process_tol_range = {
    "Turning": (0.03, 0.12),
    "Milling": (0.04, 0.15),
    "Drilling": (0.06, 0.20),
    "Grinding": (0.005, 0.04),
}

# Allowable tolerance limit (mm) depending on the fit type required
fit_allowable_limit = {"Clearance": 0.12, "Transition": 0.06, "Interference": 0.03}

rows = []
for i in range(1, N + 1):
    part = np.random.choice(part_names)
    process = np.random.choice(processes)
    material = np.random.choice(materials)
    fit = np.random.choice(fit_types)

    nominal_dim = round(np.random.uniform(10, 100), 2)

    low, high = process_tol_range[process]
    tolerance = round(np.random.uniform(low, high), 4)

    allowable = fit_allowable_limit[fit]

    # cost model: base cost scales with part size, process factor, and
    # inversely with tolerance (tighter tolerance = more expensive)
    base_cost = nominal_dim * 0.8
    tightness_penalty = (0.15 - min(tolerance, 0.15)) * 250  # tighter -> pricier
    material_factor = {"Steel": 1.2, "Aluminum": 1.0, "Brass": 1.3, "Cast_Iron": 0.9}[material]
    cost = round((base_cost + tightness_penalty) * process_cost_factor[process] * material_factor, 2)

    # Pass/Fail: allow small random measurement noise so it's not a hard cutoff
    noise = np.random.normal(0, 0.005)
    status = "PASS" if (tolerance + noise) <= allowable else "FAIL"

    rows.append([
        f"P{i:04d}", part, nominal_dim, tolerance, process, material, fit, allowable, cost, status
    ])

df = pd.DataFrame(rows, columns=[
    "Part_ID", "Part_Name", "Nominal_Dimension_mm", "Tolerance_mm",
    "Manufacturing_Process", "Material", "Fit_Type",
    "Allowable_Tolerance_mm", "Cost_Index", "Status"
])

df.to_csv("data/tolerance_training_data.csv", index=False)
print(f"Created data/tolerance_training_data.csv with {len(df)} rows")
print(df["Status"].value_counts())
