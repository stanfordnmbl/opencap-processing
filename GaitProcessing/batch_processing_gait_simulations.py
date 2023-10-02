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
import json

from gait_analysis import gait_analysis
from utils import get_trial_id, download_trial
from utilsPlotting import plot_dataframe

from utilsProcessing import align_markers_with_ground_2
from utilsOpenSim import runIKTool

from data_info import get_data_info

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking

from utilsKineticsOpenSimAD import kineticsOpenSimAD 
from utilsKinematics import kinematics

# %% Paths.
dataFolder = os.path.join(baseDir, 'Data', 'ParkerStudy')

# %% User-defined variables.
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
motion_type = 'walking_periodic_formulation_0'
case = '0'
solveProblem = True
analyzeResults = True
runProblem = True
overwrite_aligned_data = False
overwrite_gait_results = False

# %% Gait segmentation and kinematic analysis.
trials_info = get_data_info(trial_indexes=[i for i in range(0,10)])
for trial in trials_info:
    # Get trial info.
    session_id = trials_info[trial]['sid']
    trial_name = trials_info[trial]['trial']
    print('Processing session {} - trial {}...'.format(session_id, trial_name))
    # Get trial id from name.
    trial_id = get_trial_id(session_id, trial_name)
    # Set session path.
    sessionDir = os.path.join(dataFolder, "{}_{}".format(trial, session_id))
    # Set kinematic folder path.
    pathKinematicsFolder = os.path.join(sessionDir, 'OpenSimData', 'Kinematics')

    if runProblem:
        # Download data.
        trialName, modelName = download_trial(trial_id, sessionDir, session_id=session_id)
        
        # Align markers with ground.
        suffixOutputFileName = 'aligned'
        trialName_aligned = trialName + '_' + suffixOutputFileName
        # Do if not already done or if overwrite_aligned_data is True.
        if not os.path.exists(os.path.join(sessionDir, 'OpenSimData', 'Kinematics', trialName_aligned + '.mot')) or overwrite_aligned_data:
            print('Aligning markers with ground...')            
            pathTRCFile_out = align_markers_with_ground_2(
                sessionDir, trialName,
                suffixOutputFileName=suffixOutputFileName,
                lowpass_cutoff_frequency_for_marker_values=filter_frequency)
            # Run inverse kinematics.
            print('Running inverse kinematics...')
            pathGenericSetupFile = os.path.join(
                baseDir, 'OpenSimPipeline', 
                'InverseKinematics', 'Setup_InverseKinematics.xml')
            pathScaledModel = os.path.join(sessionDir, 'OpenSimData', 'Model', modelName)        
            runIKTool(pathGenericSetupFile, pathScaledModel, pathTRCFile_out, pathKinematicsFolder) 
            
        # Data processing.
        print('Processing data...')
        legs = ['r', 'l']
        for leg in legs:
            case += '_{}'.format(leg)
            # Gait segmentation and analysis.
            pathOutputJsonFile = os.path.join(pathKinematicsFolder, 'gaitResults_{}.json'.format(leg))
            # Do if not already done.
            if not os.path.exists(pathOutputJsonFile) or overwrite_gait_results:
                gaitResults = {}
                gait = gait_analysis(
                    sessionDir, trialName_aligned, leg=leg,
                    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
                    n_gait_cycles=n_gait_cycles)
                # Compute scalars.
                gaitResults['scalars'] = gait.compute_scalars(scalar_names)
                # Get gait events.
                gaitResults['events'] = gait.get_gait_events()
                # Support serialization for json
                gaitResults['events']['ipsilateralTime'] = [float(i) for i in gaitResults['events']['ipsilateralTime'].flatten()]
                gaitResults['events']['contralateralTime'] = [float(i) for i in gaitResults['events']['contralateralTime'].flatten()]
                gaitResults['events']['contralateralIdx'] = [int(i) for i in gaitResults['events']['contralateralIdx'].flatten()]
                gaitResults['events']['ipsilateralIdx'] = [int(i) for i in gaitResults['events']['ipsilateralIdx'].flatten()]                
                
                # Dump gaitResults dict in Json file and save in pathKinematicsFolder.            
                with open(pathOutputJsonFile, 'w') as outfile:
                    json.dump(gaitResults, outfile)
            else:
                with open(pathOutputJsonFile) as json_file:
                    gaitResults = json.load(json_file)

            # Setup dynamic optimization problem.
            time_window = [
                gaitResults['events']['ipsilateralTime'][0],
                gaitResults['events']['ipsilateralTime'][-1]]
            # Adjust time window to add buffers
            time_window[0] = time_window[0] - 0.3
            time_window[1] = time_window[1] + 0.3

            print('Processing data for dynamic simulation...')
            settings = processInputsOpenSimAD(
                baseDir, sessionDir, session_id, trialName_aligned, 
                motion_type, time_window=time_window)
        
        # Simulation.
        # run_tracking(baseDir, dataFolder, session_id, settings, case=case, 
        #             solveProblem=solveProblem, analyzeResults=analyzeResults)
        # test=1
        
    else:
        suffixOutputFileName = 'aligned'
        trialName_aligned = trial_name + '_' + suffixOutputFileName
        plotResultsOpenSimAD(dataFolder, session_id, trialName_aligned, cases=['39', '40'], mainPlots=False)
        test=1

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