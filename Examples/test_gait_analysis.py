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
from utils import get_trial_id, download_trial
from utilsPlotting import plot_dataframe_with_shading

# %% Paths.
baseDir = os.path.join(os.getcwd(), '..')
dataFolder = os.path.join(baseDir, 'Data')

# %% User-defined variables.
# Select example: options are treadmill and overground.
example = 'overground'

if example == 'treadmill':
    session_id = '4d5c3eb1-1a59-4ea1-9178-d3634610561c' # 1.25m/s
    trial_name = 'walk_1_25ms'

elif example == 'overground':
    session_id = 'c5c492d7-90af-417d-a80c-77f0a825ab07'
    # trial_name = 'Gait_01'
    # trial_name = 'Gait_Valgo_Rodilla'
    trial_name = 'Gait_Valgo_Rodilla_02'
    
    # session_id = '9d384602-25a6-4d9e-8936-c766215c52e5'
    # trial_name = 'Gait_Assessment1'
    # trial_name = 'Gait_Assessment1_1'
    # trial_name = 'Gait_Assessment1_2'
    

scalar_names = {'gait_speed','stride_length','step_width','cadence',
                'double_support_time','step_length_symmetry'}

scalar_labels = {
                 'gait_speed': "Gait speed (m/s)",
                 'stride_length':'Stride length (m)',
                 'step_width': 'Step width (cm)',
                 'cadence': 'Cadence (steps/min)',
                 'double_support_time': 'Double support (% gait cycle)',
                 'step_length_symmetry': 'Step length symmetry (%, R/L)'}

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

# Init gait analysis.
legs = ['r','l']
gait, gait_events, ipsilateral = {}, {}, {}
for leg in legs:
    gait[leg] = gait_analysis(
        sessionDir, trial_name, leg=leg,
        lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
        n_gait_cycles=n_gait_cycles, gait_style='overground', trimming_start=0, trimming_end=0.5)
    gait_events[leg] = gait[leg].get_gait_events()
    ipsilateral[leg] = gait_events[leg]['ipsilateralTime'][0,-1]

# Select last leg.
last_leg = 'r' if ipsilateral['r'] > ipsilateral['l'] else 'l'
other_leg = 'l' if last_leg == 'r' else 'r'
last_leg = 'r'

# Compute scalars.
gait_scalars = gait[last_leg].compute_scalars(scalar_names)
gait_scalars['gait_speed']['decimal'] = 2
gait_scalars['step_width']['decimal'] = 1
gait_scalars['stride_length']['decimal'] = 2
gait_scalars['cadence']['decimal'] = 1
gait_scalars['double_support_time']['decimal'] = 1
gait_scalars['step_length_symmetry']['decimal'] = 1

# Change units
# Default = 1
for key in gait_scalars:
    gait_scalars[key]['multiplier'] = 1

gait_scalars['step_width']['multiplier'] = 100 # cm

# Curves
gaitCurves = {}
gaitCurves[last_leg] = gait[last_leg].get_coordinates_normalized_time()
# gaitCurves[other_leg] = gait[other_leg].get_coordinates_normalized_time()
print('last leg {}'.format(last_leg))


# REFERENCE
# Gait01: right leg
# Gait_Valgo_Rodilla_01: right leg
# Gait_Valgo_Rodilla_02: left leg
# step_length_symmetry':    107.06(Gait_01),    125.90(Gait_Valgo_Rodilla_02),  123.34(Gait_Valgo_Rodilla_01)
# stride_length':           1.08(Gait_01),      1.20(Gait_Valgo_Rodilla_02),    1.29(Gait_Valgo_Rodilla_01)
# cadence':                 91.1(Gait_01),      84.7 (Gait_Valgo_Rodilla_02),   86.74(Gait_Valgo_Rodilla_01)
# step_width':              0.23(Gait_01),      0.17 (Gait_Valgo_Rodilla_02),   0.20(Gait_Valgo_Rodilla_01)
# gait_speed':              0.78(Gait_01),      0.81 (Gait_Valgo_Rodilla_02),   0.93(Gait_Valgo_Rodilla_01)
# double_support_time':     39.24(Gait_01),     34.11(Gait_Valgo_Rodilla_02),   30.12(Gait_Valgo_Rodilla_01)

# AFTER TRIMMING AND RIGHT LEG FORCED SELECTION.
# step_length_symmetry':    120.90(Gait_01),    112.38(Gait_Valgo_Rodilla_02),  123.34(Gait_Valgo_Rodilla_01)
# stride_length':           1.03(Gait_01),      1.09(Gait_Valgo_Rodilla_02),    1.29(Gait_Valgo_Rodilla_01)
# cadence':                 90.00(Gait_01),     87.80(Gait_Valgo_Rodilla_02),   86.74(Gait_Valgo_Rodilla_01)
# step_width':              0.17(Gait_01),      0.20(Gait_Valgo_Rodilla_02),    0.20(Gait_Valgo_Rodilla_01)
# gait_speed':              0.78(Gait_01),      0.77(Gait_Valgo_Rodilla_02),    0.93(Gait_Valgo_Rodilla_01)
# double_support_time':     37.50(Gait_01),     41.46(Gait_Valgo_Rodilla_02),   30.12(Gait_Valgo_Rodilla_01)


# %% You can plot multiple curves, in this case we compare right and left legs.
# plot_dataframe_with_shading(
#     [gaitCurves[last_leg]['mean']],
#     [gaitCurves[last_leg]['sd']],
#     leg = [last_leg],
#     xlabel = '% gait cycle',
#     title = 'kinematics (m or deg)',
#     legend_entries = [last_leg])