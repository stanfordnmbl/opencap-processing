'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_COM_analysis.py
    ---------------------------------------------------------------------------

    Copyright 2023 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''

import os
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import utilsKinematics
from utils import download_kinematics

# %% User inputs.
# Specify session id; see end of url in app.opencap.ai/session/<session_id>.
session_id = "3ef5cfad-cf8a-420b-af15-2d833a33cfb8"

# Specify trial names in a list; use None to process all trials in a session.
specific_trial_names = None 

# Specify where to download the data.
data_folder = os.path.join("./../Data", session_id)

# %% Download data.
trial_names = download_kinematics(session_id, folder=data_folder, trialNames=specific_trial_names)

# %% Get center of mass kinematics.
kinematics, center_of_mass = {}, {}
center_of_mass['values'], center_of_mass['speeds'], center_of_mass['accelerations'] = {}, {}, {}
for trial_name in trial_names:
    # Create object from class kinematics.
    kinematics[trial_name] = utilsKinematics.kinematics(data_folder, trial_name, lowpass_cutoff_frequency_for_coordinate_values=10)
    # Get center of mass values, speeds, and accelerations.
    center_of_mass['values'][trial_name] = kinematics[trial_name].get_center_of_mass_values(lowpass_cutoff_frequency=10)
    center_of_mass['speeds'][trial_name] = kinematics[trial_name].get_center_of_mass_speeds(lowpass_cutoff_frequency=10)
    center_of_mass['accelerations'][trial_name] = kinematics[trial_name].get_center_of_mass_accelerations(lowpass_cutoff_frequency=10)

# %% Plot center of mass vertical values and speeds.
fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
for trial_name in trial_names:
    # Align signals based on peak velocity.
    idx_peak_velocity = np.argmax(center_of_mass['speeds'][trial_name]['y'])
    time = center_of_mass['speeds'][trial_name]['time']
    x = time - time[idx_peak_velocity]
    # Plot center of mass values.
    y_values = center_of_mass['values'][trial_name]['y']
    y = y_values-y_values[0]
    axs[0].plot(x, y, label=trial_name, linewidth=3)
    # Plot center of mass speeds.
    y_speeds = center_of_mass['speeds'][trial_name]['y']
    y = y_speeds-y_speeds[0]
    axs[1].plot(x, y, label=trial_name, linewidth=3)
    
# Figure setttings.
for ax in axs:
    # Add labels.    
    ax.legend(fontsize=14)
    # Remove top and right borders.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Change font size.
    ax.tick_params(axis='both', which='major', labelsize=16)
    # Change size labels.
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
# Add labels.
axs[0].set_ylabel('CoM position (m)')
axs[1].set_ylabel('CoM velocity (m/s)')
axs[1].set_xlabel('Time (s)')
fig.align_ylabels(axs)

# %% Plot vertical forces from accelerations (F=ma).
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
gravity = 9.81
for trial_name in trial_names:
    # Align signals based on peak velocity.
    idx_peak_velocity = np.argmax(center_of_mass['speeds'][trial_name]['y'])
    time = center_of_mass['speeds'][trial_name]['time']
    x = time - time[idx_peak_velocity]
    # Plot vertical ground reaction force.
    y_accelerations = center_of_mass['accelerations'][trial_name]['y']
    y = (y_accelerations + gravity) / gravity # vGRF expressed in bodyweights: vGRF = m(a+g)/mg
    ax.plot(x, y, label=trial_name, linewidth=3)

# Figure setttings.
# Add labels. 
ax.set_ylabel('vGRF (bodyweight)')
ax.set_xlabel('Time (s)')   
ax.legend(fontsize=14)
# Remove top and right borders.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Change font size.
ax.tick_params(axis='both', which='major', labelsize=16)
# Change size labels.
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)

# %% Show figures.
plt.show()