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
sys.path.append("../")
sys.path.append("../ActivityAnalyses")

from gait_analysis import gait_analysis
import utils


# %% User-defined variables

# overground trial
# session_id = 'b39b10d1-17c7-4976-b06c-a6aaf33fead2'
# trial_name = 'gait_3'

# treadmill trial 1.25m/s
session_id = '4d5c3eb1-1a59-4ea1-9178-d3634610561c'
trial_name = 'walk_1_25ms'

scalar_names = {'gait_speed','stride_length','step_width','cadence',
                'single_support_time','double_support_time'}

# how many gait cycles you'd like to analyze
# -1 for all gait cycles detected in the trial
n_gait_cycles = -1 

# Lowpass filter frequency for kinematics data
filter_frequency = 6

# %% Gait analysis

# Get trial id from name
trial_id = utils.get_trial_id(session_id,trial_name)
    
# Local data dir -> will be deleted with lambda instance
sessionDir = os.path.join(os.path.abspath('../Data'),session_id)

# download data
trialName = utils.download_trial(trial_id,sessionDir,session_id=session_id) 

# init gait analysis
gait_r = gait_analysis(sessionDir, trialName, leg='r',
             lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
             n_gait_cycles=n_gait_cycles)
gait_l = gait_analysis(sessionDir, trialName, leg='l',
             lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
             n_gait_cycles=n_gait_cycles)
    
# compute scalars and get time-normalized kinematic curves
gaitResults = {}
gaitResults['scalars_r'] = gait_r.compute_scalars(scalar_names)
gaitResults['curves_r'] = gait_r.get_coordinates_normalized_time()
gaitResults['scalars_l'] = gait_l.compute_scalars(scalar_names)
gaitResults['curves_l'] = gait_l.get_coordinates_normalized_time()
    

# %% Print scalar results

print('\nRight foot gait metrics:')
print('(units: m and s)')
print('-----------------')
for key, value in gaitResults['scalars_r'].items():
    rounded_value = round(value, 2)
    print(f"{key}: {rounded_value}")
    
print('\nLeft foot gait metrics:')
print('(units: m and s)')
print('-----------------')
for key, value in gaitResults['scalars_l'].items():
    rounded_value = round(value, 2)
    print(f"{key}: {rounded_value}")

# %% Plot kinematic curves

for leg in ['r','l']:
    utils.plot_subplots_with_shading(gaitResults['curves_' + leg]['mean'], 
                           gaitResults['curves_' + leg]['sd'], columns=None,
                           leg=leg)