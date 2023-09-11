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
sys.path.append("../")
sys.path.append("../ActivityAnalyses")

from gait_analysis import gait_analysis
from utils import get_trial_id, download_trial
from utilsPlotting import plot_dataframe_with_shading

# %% Paths.
baseDir = os.path.join(os.getcwd(), '..')
dataFolder = os.path.join(baseDir, 'Data')

# %% User-defined variables.
# Select example: options are treadmill and overground.

session_id = '1a1e9c15-4bc3-4b72-bf9d-cc6ba9293c23'
trial_name = '10mwt'

scalar_names = {'gait_speed','stride_length','step_width','cadence',
                'stance_time','swing_time', 'single_support_time',
                'double_support_time'}

# Select how many gait cycles you'd like to analyze. Select -1 for all gait
# cycles detected in the trial.
n_gait_cycles = 1

# Select lowpass filter frequency for kinematics data.
filter_frequency = 6

# %% Gait analysis.
# Get trial id from name.
# trial_id = get_trial_id(session_id,trial_name)    

# Set session path.
sessionDir = os.path.join(dataFolder, session_id)

# Download data.
# trialName = download_trial(trial_id,sessionDir,session_id=session_id) 
trialName = '10mwt'

# Init gait analysis.
gait_r = gait_analysis(
    sessionDir, trialName, leg='r',
    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    n_gait_cycles=n_gait_cycles)
gait_l = gait_analysis(
    sessionDir, trialName, leg='l',
    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    n_gait_cycles=n_gait_cycles)
    
# Compute scalars and get events.
gaitResults = {}
gaitResults['scalars_r'] = gait_r.compute_scalars(scalar_names)
gaitResults['scalars_l'] = gait_l.compute_scalars(scalar_names)
gaitResults['gait_events_r'] = gait_r.get_gait_events()
gaitResults['gait_events_l'] = gait_l.get_gait_events()

# Select last leg
# TODO: add try catch
if gaitResults['gait_events_r']['ipsilateralTime'][0,-1] > gaitResults['gait_events_l']['ipsilateralTime'][0,-1]:
    last_leg = 'r'
else:
    last_leg = 'l'
# %% Return values.  
# Return indices for visualizer and line curve plot
idx_start = gaitResults['gait_events_' + last_leg]['ipsilateralIdx'][0,0]
idx_end = gaitResults['gait_events_' + last_leg]['ipsilateralIdx'][0,-1]

# The visualizer loads the file tagged `visualizerTransforms-json`. 
# For the gait dashboard, we should use the same file but play it from
# index idx_start to index idx_end.

# The line curve chart loads the file tagged `ik_results`.
# For the gait dashboard, we should use the same file but play it from
# index idx_start to index idx_end.

# Both files have the same number of frames. We want to display a vertical bar
# on the line curve chart that is temporarily aligned with the visualizer. Eg,
# when the visualizer displays frame #50, the vertical bar in the line curve
# chart should be at frame #50. The goal is to allow users to compare what is
# happening in the visualizer with the skeleton with what is happening in the
# data. For example, when the left foot touches the ground, the angle of the
# knee is 10 degrees.

# Return gait metrics for scalar chart
gait_speed = np.round(gaitResults['scalars_' + last_leg]['gait_speed'], 2)
gait_cadence = np.round(gaitResults['scalars_' + last_leg]['cadence'], 2)
stride_length = np.round(gaitResults['scalars_' + last_leg]['stride_length'], 2)
step_width = np.round(gaitResults['scalars_' + last_leg]['step_width'], 2)
stance_time = np.round(gaitResults['scalars_' + last_leg]['stance_time'], 2)
swing_time = np.round(gaitResults['scalars_' + last_leg]['swing_time'], 2)
single_support_time = np.round(gaitResults['scalars_' + last_leg]['single_support_time'], 2)
double_support_time = np.round(gaitResults['scalars_' + last_leg]['double_support_time'], 2)


# # %% Print scalar results.
# print('\nRight foot gait metrics:')
# print('(units: m and s)')
# print('-----------------')
# for key, value in gaitResults['scalars_r'].items():
#     rounded_value = round(value, 2)
#     print(f"{key}: {rounded_value}")
    
# print('\nLeft foot gait metrics:')
# print('(units: m and s)')
# print('-----------------')
# for key, value in gaitResults['scalars_l'].items():
#     rounded_value = round(value, 2)
#     print(f"{key}: {rounded_value}")

    
# # %% You can plot multiple curves, in this case we compare right and left legs.
# plot_dataframe_with_shading(
#     [gaitResults['curves_r']['mean'], gaitResults['curves_l']['mean']],
#     [gaitResults['curves_r']['sd'], gaitResults['curves_l']['sd']],
#     leg = ['r','l'],
#     xlabel = '% gait cycle',
#     title = 'kinematics (m or deg)',
#     legend_entries = ['right','left'])