'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_dropjump_analysis.py
    ---------------------------------------------------------------------------
    Copyright 2024 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Carmichael Ong
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
                
    Please contact us for any questions: https://www.opencap.ai/#contact

    This example shows how to run a kinematic analysis of dropjump data.
    
'''

import os
import sys
sys.path.append("../")
sys.path.append("../ActivityAnalyses")
import numpy as np
import json
from dropjump_analysis import dropjump_analysis
from utils import get_trial_id, download_trial

# %% Paths.
baseDir = os.path.join(os.getcwd(), '..')
dataFolder = os.path.join(baseDir, 'Data')

# %% User-defined variables.
# example with KA
session_id = '907bd795-093b-44e7-9656-3c1b38da7dcc'
trial_name = 'dropjump_7'
#trial_name = 'dropjump_8' # this case fails as foot is noisy after toe off

scalars = {
        'jump_height': {'label': 'Jump height (cm)', 'order': 0, 'decimal': 1, 'multiplier': 100},
        'contact_time': {'label': 'Contact time (s)', 'order': 1, 'decimal': 2, 'multiplier': 1},
        'peak_knee_flexion_angle': {'label': 'Peak knee flexion angle (deg)', 'order': 2, 'decimal': 1, 'multiplier': 1},
        'peak_hip_flexion_angle': {'label': 'Peak hip flexion angle (deg)', 'order': 3, 'decimal': 1, 'multiplier': 1},
        'peak_ankle_dorsiflexion_angle': {'label': 'Peak ankle dorsiflexion angle (deg)', 'order': 4, 'decimal': 1, 'multiplier': 1},
        'peak_trunk_lean': {'label': 'Peak trunk lean angle (deg) (%, R/L)', 'order': 5, 'decimal': 1, 'multiplier': 1},
    }
scalar_names = list(scalars.keys())

# Select lowpass filter frequency for kinematics data.
filter_frequency = -1

# %% Analysis.
# Get trial id from name.
trial_id = get_trial_id(session_id,trial_name)    

# Set session path.
sessionDir = os.path.join(dataFolder, session_id)

# Download data.
trialName = download_trial(trial_id,sessionDir,session_id=session_id) 

# Init gait analysis.
dropjump = dropjump_analysis(
    sessionDir, trialName,
    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency)
dropjump_events = dropjump.get_dropjump_events()
drop_jump_results = dropjump.compute_scalars(scalar_names)
drop_jump_curves = dropjump.coordinateValues
drop_jump_com = dropjump._comValues

# %% Print scalar results.
print('\nMetrics:')
print('-----------------')
for key, value in drop_jump_results.items():
    rounded_value = round(value['value'], 2)
    print(f"{key}: {rounded_value} {value['units']}")

# %% For deployment
# Scalars
scalars['jump_height']['threshold'] = 50
scalars['contact_time']['threshold'] = 0.7
scalars['peak_knee_flexion_angle']['threshold'] = 70
scalars['peak_hip_flexion_angle']['threshold'] = 70
scalars['peak_ankle_dorsiflexion_angle']['threshold'] = 30
scalars['peak_trunk_lean']['threshold'] = 30

# Whether below-threshold values should be colored in red (default, higher is better) or green (reverse, lower is better).
scalar_reverse_colors = ['contact_time', 'peak_trunk_lean']
# Whether should be red-green-red plot
scalar_centered = []
# Whether to exclude some scalars
scalars_to_exclude = []

# Create dicts
metrics_out = {}
for scalar_name in scalar_names:
    if scalar_name in scalars_to_exclude:
        continue
    metrics_out[scalar_name] = {}
    vertical_values = np.round(drop_jump_results[scalar_name]['value'] *
                               scalars[scalar_name]['multiplier'], 
                               scalars[scalar_name]['decimal'])
    metrics_out[scalar_name]['label'] = scalars[scalar_name]['label']
    metrics_out[scalar_name]['value'] = vertical_values
    # metrics_out[scalar_name]['info'] = scalars[scalar_name]['info']
    metrics_out[scalar_name]['decimal'] = scalars[scalar_name]['decimal']
    if scalar_name in scalar_reverse_colors:
        # Margin zone (orange) is 10% above threshold.
        metrics_out[scalar_name]['colors'] = ["green", "yellow", "red"]
        metrics_out[scalar_name]['min_limit'] = float(np.round(scalars[scalar_name]['threshold'],scalars[scalar_name]['decimal']))
        metrics_out[scalar_name]['max_limit'] = float(np.round(1.10*scalars[scalar_name]['threshold'],scalars[scalar_name]['decimal']))
    elif scalar_name in scalar_centered:
        # Red, green, red
        metrics_out[scalar_name]['colors'] = ["red", "green", "red"]
        metrics_out[scalar_name]['min_limit'] = float(np.round(scalars[scalar_name]['threshold'][0],scalars[scalar_name]['decimal']))        
        metrics_out[scalar_name]['max_limit'] = float(np.round(scalars[scalar_name]['threshold'][1],scalars[scalar_name]['decimal'])) 
    else:
        # Margin zone (orange) is 10% below threshold.
        metrics_out[scalar_name]['colors'] = ["red", "yellow", "green"]
        metrics_out[scalar_name]['min_limit'] = float(np.round(0.90*scalars[scalar_name]['threshold'],scalars[scalar_name]['decimal']))
        metrics_out[scalar_name]['max_limit'] = float(np.round(scalars[scalar_name]['threshold'],scalars[scalar_name]['decimal']))

# Order dicts
metrics_out_ordered = metrics_out.copy()
for scalar_name in scalar_names:
    if scalar_name in metrics_out_ordered:
        # change the name of the key to str(scalars['order]) + scalar_name
        # the name should be a two-character string, if the order is only one digit, add a 0 in front
        order = scalars[scalar_name]['order']
        if order < 10:
            order = '0' + str(order)
        else:
            order = str(order)
        metrics_out_ordered[order + '_' + scalar_name] = metrics_out_ordered.pop(scalar_name)
        
# Time-series data
indices = {}
indices['start'] = int(dropjump_events['eventIdxs']['contactIdxs'][0])
indices['end'] = int(dropjump_events['eventIdxs']['toeoffIdxs'][0])
# Add buffer of 0.3s before and after the event
time = drop_jump_curves['time'].to_numpy()
sample_rate = np.round(np.mean(1/np.diff(time)))
buffer_time = 0.3
buffer_idx = int(buffer_time * sample_rate)
indices['start'] = np.max([0, indices['start'] - buffer_idx])
indices['end'] = np.min([len(drop_jump_curves['time']), indices['end'] + buffer_idx])
times = {'start': time[indices['start']], 'end': time[indices['end']]}

# Create time-series dataset
shoulder_model_translations = ['sh_tx_r', 'sh_ty_r', 'sh_tz_r', 'sh_tx_l', 'sh_ty_l', 'sh_tz_l']
colNames = drop_jump_curves.columns
data = drop_jump_curves.to_numpy()
# Add center of mass data
colCOMNames = drop_jump_com.columns
# Append com_ to colCOMNames
colCOMNames = ['COM_'+ colCOMNames[i] for i in range(len(colCOMNames))]
dataCOM = drop_jump_com.to_numpy()
coordValues = data[indices['start']:indices['end']+1]
comValues = dataCOM[indices['start']:indices['end']+1]
datasets = []
for i in range(coordValues.shape[0]):
    datasets.append({})
    for j in range(coordValues.shape[1]):
        # Exclude knee_angle_r_beta and knee_angle_l_beta
        if 'beta' in colNames[j] or 'mtp' in colNames[j] or colNames[j] in shoulder_model_translations:
            continue
        datasets[i][colNames[j]] = coordValues[i,j]

    for j in range(comValues.shape[1]):
        # Exclude time (already in)
        if 'time' in colCOMNames[j]:
            continue
        datasets[i][colCOMNames[j]] = comValues[i,j]
        
# Available options for line curve chart.
y_axes = list(colNames) + list(colCOMNames)
y_axes.remove('time')
y_axes.remove('knee_angle_r_beta')
y_axes.remove('knee_angle_l_beta')
y_axes.remove('mtp_angle_r')
y_axes.remove('mtp_angle_l')
y_axes = [x for x in y_axes if x not in shoulder_model_translations]

# Output json
results = {
    'indices': times, 
    'metrics': metrics_out_ordered, 
    'datasets': datasets,
    'x_axis': 'time', 
    'y_axis': y_axes,
    }
# with open('dropjump_analysis.json', 'w') as f:
#     json.dump(results, f, indent=4)
    