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
sys.path.append("../UtilsDynamicSimulations/OpenSimAD")
baseDir = os.path.join(os.getcwd(), '..')
sys.path.append(baseDir)
activityAnalysesDir = os.path.join(baseDir, 'ActivityAnalyses')
sys.path.append(activityAnalysesDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)

from gait_analysis import gait_analysis
from utils import get_trial_id, download_trial
from utilsPlotting import plot_dataframe

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking

from utilsKineticsOpenSimAD import kineticsOpenSimAD 

# %% Paths.
dataFolder = os.path.join(baseDir, 'Data')

# %% User-defined variables.
session_id = 'bca0aad8-c129-4a62-bef3-b5de1659df5e'
trial_name = '10mwt'

scalar_names = {'gait_speed','stride_length','step_width','cadence',
                'single_support_time','double_support_time'}

# Select how many gait cycles you'd like to analyze. Select -1 for all gait
# cycles detected in the trial.
n_gait_cycles = 1 

# Select lowpass filter frequency for kinematics data.
filter_frequency = 6

# Settings for dynamic simulation.
# motion_type = 'walking_periodic_torque_driven'
# case = '2'
# solveProblem = True
# analyzeResults = True
motion_type = 'walking_periodic'
case = '18'
solveProblem = True
analyzeResults = True
runProblem = True

if case == '2' or case == '3':
    contact_configuration = 'dhondt2023'
else:
    contact_configuration = 'generic'

# %% Gait segmentation and kinematic analysis.
# Get trial id from name.
trial_id = get_trial_id(session_id,trial_name)    

# Set session path.
sessionDir = os.path.join(dataFolder, session_id)

if runProblem:
    # Download data.
    trialName = download_trial(trial_id,sessionDir,session_id=session_id) 
    
    # Data processing.
    legs = ['r']
    gait, gaitResults = {}, {}
    for leg in legs:
        gaitResults[leg] = {}
        gait[leg] = gait_analysis(
            sessionDir, trialName, leg=leg,
            lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
            n_gait_cycles=n_gait_cycles)
        # Compute scalars.
        gaitResults[leg]['scalars'] = gait[leg].compute_scalars(scalar_names)
        # Get gait events.
        gaitResults[leg]['events'] = gait[leg].get_gait_events()
        # Setup dynamic optimization problem.
        time_window = [float(gaitResults[leg]['events']['ipsilateralTime'][0, 0]),
                       float(gaitResults[leg]['events']['ipsilateralTime'][0, -1])]
        test= [1.2, 2.4] 
        settings = processInputsOpenSimAD(
            baseDir, dataFolder, session_id, trial_name, 
            motion_type, time_window=time_window, 
            contact_configuration=contact_configuration)
        
        settings['contact_configuration'] = contact_configuration
        if case == '4':    
            settings['tendon_compliances'] =  {'soleus_r': 17.5, 'gaslat_r': 17.5, 'gasmed_r': 17.5,
                                               'soleus_l': 17.5, 'gaslat_l': 17.5, 'gasmed_l': 17.5}
        if case == '5' or case == '6' or case == '7' or case == '11' or case == '12' or case == '13':
            settings['weights']['activationTerm'] = 10 
    
        if case == '7':
            settings['weights']['accelerationTrackingTerm'] = 10 
    
        if case == '8' or case == '11':
            settings['weights']['positionTrackingTerm'] = 20
        if case == '9' or case == '12':
            settings['weights']['positionTrackingTerm'] = 50
        if case == '10' or case == '13':
            settings['weights']['positionTrackingTerm'] = 100
    
        if case == '14':
            settings['weights']['activationTerm'] = 10
            settings['periodicConstraints']['coordinateValues'] = ['pelvis_ty']
            settings['periodicConstraints']['coordinateSpeeds'] = ['pelvis_ty']
            
        if case == '15':
            settings['weights']['activationTerm'] = 10
            settings['scaleIsometricMuscleForce'] = 1.5

        if case == '16':
            settings['weights']['activationTerm'] = 10
            settings['withReserveActuators'] = True
            settings['reserveActuatorCoordinates']= {
                'mtp_angle_l': 30, 'mtp_angle_r': 30}
            
        if case == '17':
            settings['weights']['activationTerm'] = 10
            settings['trackQdds'] = False
            
        if case == '18':
            settings['weights']['activationTerm'] = 10
            settings['coordinates_toTrack']['hip_flexion_l']['weight'] = 50
            settings['coordinates_toTrack']['hip_flexion_r']['weight'] = 50 
            settings['coordinates_toTrack']['knee_angle_l']['weight'] = 50 
            settings['coordinates_toTrack']['knee_angle_r']['weight'] = 50 
            settings['coordinates_toTrack']['ankle_angle_l']['weight'] = 50 
            settings['coordinates_toTrack']['ankle_angle_r']['weight'] = 50 
            
        # Simulation.
        run_tracking(baseDir, dataFolder, session_id, settings, case=case, 
                      solveProblem=solveProblem, analyzeResults=analyzeResults)
else:
    plotResultsOpenSimAD(dataFolder, session_id, trial_name, cases=['6', '16'], mainPlots=False)
    # test=1

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