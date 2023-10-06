'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_gait_analysis.py
    ---------------------------------------------------------------------------
    Copyright 2023 Stanford University and the Authors
    
    Author(s): Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
                
    Please contact us for any questions: https://www.opencap.ai/#contact

    This example shows how to run a kinematic analysis of gait data. It works
    with either treadmill or overground gait. You can compute scalar metrics 
    as well as gait cycle-averaged kinematic curves.
    
'''

import os
import sys
import numpy as np
import json
sys.path.append("../")
sys.path.append("../ActivityAnalyses")

from gait_analysis import gait_analysis
from utils import get_trial_id, download_trial, import_metadata
from utilsPlotting import create_custom_bar_subplots

# %% Paths.
baseDir = os.path.join(os.getcwd(), '..')
dataFolder = os.path.join(baseDir, 'Data')

# %% User-defined variables.
# Select example: options are treadmill and overground.

session_id = 'd701862d-ec04-4ea8-9165-69cfbf21a041'
trial_name = 'walking'

scalar_names = {'gait_speed','stride_length','step_width','cadence',
                'single_support_time', 'double_support_time'}

scalar_labels = {'gait_speed': "Gait speed (m/s)",
                 'stride_length':'Stride length (m)',
                 'step_width': 'Step width (m)',
                 'cadence': 'Cadence (steps/min)',
                 'single_support_time': 'Single support time (s)', 
                 'double_support_time': 'Double support time (s)'}

# Select how many gait cycles you'd like to analyze. Select -1 for all gait
# cycles detected in the trial.
n_gait_cycles = 1

# Select lowpass filter frequency for kinematics data.
filter_frequency = 6

# %% Gait analysis.
# Get trial id from name.
trial_id = get_trial_id(session_id,trial_name)    

# Set session path.
sessionDir = os.path.join(dataFolder, session_id)

# Download data.
trialName = download_trial(trial_id,sessionDir,session_id=session_id)

# Init gait analysis and get gait events.
legs = ['r','l']
gait, gait_events, ipsilateral = {}, {}, {}
for leg in legs:
    gait[leg] = gait_analysis(
        sessionDir, trialName, leg=leg,
        lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
        n_gait_cycles=n_gait_cycles)
    gait_events[leg] = gait[leg].get_gait_events()
    ipsilateral[leg] = gait_events[leg]['ipsilateralTime'][0,-1]

# Select last leg.
last_leg = 'r' if ipsilateral['r'] > ipsilateral['l'] else 'l'

# Compute scalars.
gait_scalars = gait[last_leg].compute_scalars(scalar_names)

# %% Return gait metrics for scalar chart.
# Instructions for the frontend will come later.
gait_metrics = {}
for scalar_name in scalar_names:
    gait_metrics[scalar_name] = np.round(gait_scalars[scalar_name]['value'], 2)
    print(scalar_name, gait_metrics[scalar_name])

# %% Example scalar plots.
# Thesholds for scalar metrics.
# TODO Stride length : Clear with at least 40% of height (cm)
# Step width : Clear with 7 cm or less
# Gait speed : Clear with 67m/min or more
# Cadence : Clear with 100steps/min or more
# Contact time
# ・Single support : Clear with increase/no change from the previous session
# ・Double support : Clear with decrease/no change from the previous session
metadataPath = os.path.join(sessionDir, 'sessionMetadata.yaml')
metadata = import_metadata(metadataPath)
subject_height = metadata['height_m']
gait_speed_threshold = 67/60
step_width_threshold = 0.25
stride_length_threshold = 1.4 # subject_height*0.4
cadence_threshold = 100
single_support_time_threshold = 0.4
double_support_time_threshold = 0.3
thresholds = {'gait_speed': gait_speed_threshold,
              'step_width': step_width_threshold,
              'stride_length': stride_length_threshold,
              'cadence': cadence_threshold,
              'single_support_time': single_support_time_threshold,
              'double_support_time': double_support_time_threshold}
# Whether below-threshold values should be colored in red (default) or green (reverse).
scalar_reverse_colors = ['step_width']

# %% Create json for deployement.
# Indices / Times
indices = {}
indices['start'] = int(gait_events[last_leg]['ipsilateralIdx'][0,0])
indices['end'] = int(gait_events[last_leg]['ipsilateralIdx'][0,-1])
times = {}
times['start'] = float(gait_events[last_leg]['ipsilateralTime'][0,0])
times['end'] = float(gait_events[last_leg]['ipsilateralTime'][0,-1])

# Metrics
metrics_out = {}
for scalar_name in scalar_names:
    metrics_out[scalar_name] = {}
    vertical_values = np.round(gait_scalars[scalar_name]['value'], 2)
    metrics_out[scalar_name]['label'] = scalar_labels[scalar_name]
    metrics_out[scalar_name]['value'] = vertical_values
    if scalar_name in scalar_reverse_colors:
        # Margin zone (orange) is 10% above threshold.
        metrics_out[scalar_name]['colors'] = ["green", "yellow", "red"]
        metrics_out[scalar_name]['min_limit'] = float(np.round(thresholds[scalar_name],2))
        metrics_out[scalar_name]['max_limit'] = float(np.round(1.10*thresholds[scalar_name],2))
    else:
        # Margin zone (orange) is 10% below threshold.
        metrics_out[scalar_name]['colors'] = ["red", "yellow", "green"]
        metrics_out[scalar_name]['min_limit'] = float(np.round(0.90*thresholds[scalar_name],2))
        metrics_out[scalar_name]['max_limit'] = float(np.round(thresholds[scalar_name],2))
        
# Datasets
colNames = gait[last_leg].coordinateValues.columns
data = gait[last_leg].coordinateValues.to_numpy()
coordValues = data[indices['start']:indices['end']+1]
datasets = []
for i in range(coordValues.shape[0]):
    datasets.append({})
    for j in range(coordValues.shape[1]):
        datasets[i][colNames[j]] = coordValues[i,j]
        
# Available options for line curve chart.
y_axes = list(colNames)
y_axes.remove('time')

# Dump data into json file.
# Create results dictionnary with indices and gait_metrics.
results = {'indices': times, 'metrics': metrics_out, 'datasets': datasets,
           'x_axis': 'time', 'y_axis': y_axes}
with open('results.json', 'w') as outfile:
    json.dump(results, outfile)
    
# %% Example plot
# Create data dictionary for each scalar.
data_dict_list = []
for scalar_name in scalar_names:
    vertical_values = [np.round(gait_scalars[scalar_name]['value'], 2)]
    data_dict = {
        'name': scalar_labels[scalar_name],
        'values': vertical_values,
    }
    if scalar_name in scalar_reverse_colors:
        data_dict['reverse_colors'] = True
        lower_bound = np.round(thresholds[scalar_name],2)
        # Margin zone (orange) is 10% above threshold.
        upper_bound = np.round(1.10*thresholds[scalar_name],2)   
    else:
        # Margin zone (orange) is 10% below threshold.
        lower_bound = np.round(0.90*thresholds[scalar_name],2)
        upper_bound = np.round(thresholds[scalar_name],2)
    data_dict['bounds'] = (lower_bound, upper_bound)
    data_dict_list.append(data_dict)

create_custom_bar_subplots(data_dict_list)