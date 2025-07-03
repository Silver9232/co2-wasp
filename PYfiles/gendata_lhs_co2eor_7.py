# SPDX-FileCopyrightText: 2024 NORCE
# SPDX-License-Identifier: GPL-3.0

"""
Script to run Flow for random input variables (gas and water injection rates).
Injection times are scaled to have consistent injected volumes.
"""

import os
import math as mt
import numpy as np
import pandas as pd
from scipy.stats import qmc
import warnings
from mako.template import Template
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

import subprocess

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from resdata.summary import Summary
   
SEED = 7   
np.random.seed(SEED)

# Inputs
num_samples = 300 # Define number of simulation samples
grates_min, grates_max = 1000, 5000 # Define the ranges of Mscf
wrates_min, wrates_max = 5000, 15000 # Define the ranges of stb 15000, 20000
NSCHED = 20 # Number of cycles (one period is water injection followd by gas injection)
TIME = 91.25 # Duration of the water/gas injection period in days
GVOL = 270000 # Target gas volume to inject in a period (i.e., larger grates would have shorter injection periods so all grates cases inject the same volumes)
WVOL = 5*GVOL # Target gas volume to inject in a period (i.e., larger grates would have shorter injection periods so all grates cases inject the same volumes)
NPRUNS = 50 # Number of paralell runs (it should be limited by the number of your cpus)
DELETE = 1 # Set to 0 to no delete the simulation files (careful with your PC memmory)
FLOW = "flow" # Set the path to the flow executable
#FLOW = "/root/OPM/build/opm-simulators/bin/flow" # Set the path to the flow executable
EOR = "co2eor" # Select foam or co2eor
CASE = '1' # Set 1 if different cumulative volumes are allowed or 0 if it mantains the same volume (GVOL-WVOL)for all the realization
makoname = "eor1.mako"

# Create output directory
output_dir = f"{EOR}_lhs_seed-{SEED}"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it does not exist

###################### LHS SAMPLES ##############################################################
# Create Latin Hypercube Sampler instance for two parameters (gas rate and water rate)
sampler = qmc.LatinHypercube(d=2, rng=SEED)  # `d=2` because we have two parameters (GRATES and WRATES)
lhs_samples = sampler.random(n=num_samples)
assert lhs_samples.ndim == 2 and lhs_samples.shape[1] == 2, "LHS sample should be a 2D array with 2 columns."

# Scale the LHS samples to the desired ranges
GRATES_lhs = lhs_samples[:, 0] * (grates_max - grates_min) + grates_min
WRATES_lhs = lhs_samples[:, 1] * (wrates_max - wrates_min) + wrates_min

# Generate random samples for comparison
GRATES = np.random.uniform(grates_min, grates_max, num_samples)  # Randomly sampled gas injection rate values
WRATES = np.random.uniform(wrates_min, wrates_max, num_samples)  # Randomly sampled water injection rate values

# Combine samples into points (2D array)
points = np.column_stack((GRATES, WRATES))
points_lhs = np.column_stack((GRATES_lhs, WRATES_lhs))

# Plot the LHS and random points for comparison
plt.figure(figsize=(12, 6))

# Plot LHS-generated points
plt.subplot(1, 2, 1)
plt.scatter(GRATES_lhs, WRATES_lhs, color='blue', label='LHS Points')
plt.xlabel('Gas Rate (stb/day)')
plt.ylabel('Water Rate (stb/day)')
plt.title('LHS-Generated '+ str(num_samples) + ' points')
plt.legend()
plt.grid(True)

# Plot random-generated points
plt.subplot(1, 2, 2)
plt.scatter(GRATES, WRATES, color='green', label='Random Points')
plt.xlabel('Gas Rate (stb/day)')
plt.ylabel('Water Rate (stb/day)')
plt.title('Randomly-Generated '+ str(num_samples) +' points')
plt.legend()
plt.grid(True)
# Save the figure with both plots
plt.tight_layout()
plt.savefig(f"{output_dir}/lhs_vs_random_points.png")


# Calculate pairwise Euclidean distances using pdist
distances = pdist(points, metric='euclidean')
distances_lhs = pdist(points_lhs, metric='euclidean')

# Create a boxplot to visualize the distribution of distances for both methods
plt.figure(figsize=(10, 7))
plt.boxplot([distances, distances_lhs], vert=True, patch_artist=True, tick_labels=['Random', 'LHS'])
plt.title('Boxplot of Euclidean Distances Between All Generated Points')
plt.ylabel('Euclidean Distance')
plt.grid(True)

# Save the boxplot figure
plt.savefig(f"{output_dir}/distances_boxplot.png")
plt.close()


# Initialize results storage
names = {name: [] for name in ['oil_pro_vol', 'wat_inj_vol', 'wat_pro_vol', 'gas_inj_vol', 'gas_pro_vol', "gas_retained",
                                'co2_injected', 'co2_backprod', 'co2_retained','prod_bhp','inj_bhp', 'ratio_oil_to_inj_vol']}

# Generate the configuration files, run simulations, read data, and delete if required
mytemplate = Template(filename=makoname)

for i, (grate, wrate) in enumerate(zip(GRATES_lhs, WRATES_lhs)):
    var = {"flow": FLOW, "eor": EOR, "wtime":TIME, "grate": grate, "wrate":wrate, "gvol":GVOL, "wvol":WVOL, "nsched":NSCHED}
    filled_template = mytemplate.render(**var)
     # Save the filled template to a text file
    output_file = os.path.join(output_dir, f"{EOR}_{i}.toml")
    with open(output_file, "w", encoding="utf8") as file:
        file.write(filled_template)

for i in range(mt.floor(len(GRATES_lhs) / NPRUNS)):
    command = ""
    for j in range(NPRUNS):
        if DELETE == 1:
            command += f"pyopmnearwell -i {output_dir}/{EOR}_{NPRUNS*i+j}.toml -o {output_dir}/{EOR}_{NPRUNS*i+j} -g single -v 0 & "
        else:
            command += f"pyopmnearwell -i {output_dir}/{EOR}_{NPRUNS*i+j}.toml -o {output_dir}/{EOR}_{NPRUNS*i+j} -g single & "
    command += "wait"
    os.system(command)
    for j in range(NPRUNS):
        smspec = Summary(f"./{output_dir}/{EOR}_{NPRUNS*i+j}/{EOR.upper()}_{NPRUNS*i+j}.SMSPEC")
        names["wat_inj_vol"].append(smspec["FWIT"].values[-1])
        names["wat_pro_vol"].append(smspec["FWPT"].values[-1])
        names["gas_inj_vol"].append(smspec["FGIT"].values[-1])
        names["gas_pro_vol"].append(smspec["FGPT"].values[-1])
        names["gas_retained"].append(smspec["FGIT"].values[-1]-smspec["FGPT"].values[-1])
        names["prod_bhp"].append(smspec["WBHP:PROD"].values[-1])
        names["inj_bhp"].append(smspec["WBHP:INJW"].values[-1])
        names["oil_pro_vol"].append(smspec["FOPT"].values[-1])
        names["ratio_oil_to_inj_vol"].append(smspec["FOPT"].values[-1]/(smspec["FGIT"].values[-1]+smspec["FWIT"].values[-1]))
        names["co2_injected"].append(smspec["FNIT"].values[-1])
        names["co2_backprod"].append(smspec["FNPT"].values[-1])
        names["co2_retained"].append(smspec["FNIT"].values[-1]-smspec["FNPT"].values[-1])
        if DELETE == 1:
            os.system(f"rm -rf {output_dir}/{EOR}_{NPRUNS*i+j} {EOR}_{NPRUNS*i+j}.txt")
finished = NPRUNS * mt.floor(len(GRATES_lhs) / NPRUNS)
remaining = len(GRATES_lhs) - finished
command = ""
for i in range(remaining):
    if DELETE == 1:
        command += f"pyopmnearwell -i {output_dir}/{EOR}_{finished+i}.toml -o {output_dir}/{EOR}_{finished+i} -g single -v 0 & "
    else:
        command += f"pyopmnearwell -i {output_dir}/{EOR}_{finished+i}.toml -o {output_dir}/{EOR}_{finished+i} -g single & "
command += "wait"
os.system(command)
for i in range(remaining):
    smspec = Summary(f"./{output_dir}/{EOR}_{finished+i}/{EOR.upper()}_{finished+i}.SMSPEC")
    #ratio_oil_to_injected_volumes.append(smspec["FOPT"].values[-1]/(smspec["FGIT"].values[-1]+smspec["FWIT"].values[-1]))
    if DELETE == 1:
        os.system(f"rm -rf {output_dir}/{EOR}_{finished+i} {EOR}_{finished+i}.toml")

# Save variables to numpy objects
np.save(output_dir+'/'+'grates', GRATES_lhs)
np.save(output_dir+'/'+'wrates', WRATES_lhs)
for key in names.keys():
    np.save(output_dir+'/'+EOR+"_"+key, names[key])

# Plot the variables for quick inspection
for name in names.keys():
    fig, axis = plt.subplots()
    axis.plot(
        GRATES_lhs,
        names[name],
        color="b",
        linestyle="",
        marker="*",
        markersize=5,
    )
    axis.set_ylabel(name, fontsize=12)
    axis.set_xlabel(r"Gas injection rate [stb/day]", fontsize=12)
    fig.savefig(f"{output_dir}/{EOR}_{name}.png")
    

fig, axis = plt.subplots()
axis.plot(
        WRATES,
        names['wat_inj_vol'],
        color="b",
        linestyle="",
        marker="*",
        markersize=5,
    )
axis.set_ylabel(r"wat_inj_vol [stb]", fontsize=12)
axis.set_xlabel(r"Water injection rate [stb/day]", fontsize=12)
fig.savefig(f"{output_dir}/{EOR}_wat_inj_vol.png")
    
fig, axis = plt.subplots()
axis.plot(
        GRATES,
        WRATES,
        color="b",
        linestyle="",
        marker="*",
        markersize=5,
    )
axis.set_ylabel(r"Water injection rate [stb/day]", fontsize=12)
axis.set_xlabel(r"Gas injection rate [Mscf/day]", fontsize=12)
fig.savefig(f"{output_dir}/sample_space.png")

# Load the saved NumPy arrays
grates = np.load(output_dir + '/' + 'grates.npy')
wrates = np.load(output_dir + '/' + 'wrates.npy')

# Initialize the DataFrame with GRATES and WRATES
data = {
    "Gas Injection Rate (GRATES)": grates,
    "Water Injection Rate (WRATES)": wrates,
}

# Load the additional variables into the DataFrame
for key in names.keys():
    values = np.load(output_dir + '/' + EOR + "_" + key + '.npy')
    data[key] = values

# Create the DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Save the DataFrame to a CSV file (optional)
df.to_csv(output_dir + '/' + f'{output_dir}_simulation_results.csv', index=False)
