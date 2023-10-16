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
import numpy as np

from gait_analysis import gait_analysis
from utils import get_trial_id, download_trial, numpy_to_storage
from utilsPlotting import plot_dataframe

from utilsProcessing import align_markers_with_ground_3
from utilsOpenSim import runIKTool

from data_info import get_data_info, get_data_info_problems, get_data_alignment, get_data_select_previous_cycle, get_data_manual_alignment, get_data_select_window

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
case = '5'
legs = ['l']
solveProblem = True
analyzeResults = True
runProblem = True
overwrite_aligned_data = False
overwrite_gait_results = False
overwrite_tracked_motion_file = False

# Buffers
if case == '2':
    buffer_start = 0.5
    buffer_end = 0.3
elif case == '3':
    # Buffers
    buffer_start = 0.5
    buffer_end = 0.5
elif case == '4':
    # Buffers
    buffer_start = 0.7
    buffer_end = 0.5
elif case == '5':
    # Buffers
    buffer_start = 0.7
    buffer_end = 0.3


# %% Gait segmentation and kinematic analysis.
# ii = 91
trials_to_run = [86, 91] # [0, 3, 13, 35, 48, 62]

# trials_info = get_data_info(trial_indexes=[i for i in range(ii,ii+1)])
# trials_info = get_data_info(trial_indexes=[i for i in range(60,92)])
trials_info = get_data_info(trial_indexes=trials_to_run)

trials_info_problems = get_data_info_problems()
trials_info_alignment = get_data_alignment()
trials_select_previous_cycle = get_data_select_previous_cycle()
trials_manual_alignment = get_data_manual_alignment()
trials_select_window = get_data_select_window()

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
        try:
            trialName, modelName = download_trial(trial_id, sessionDir, session_id=session_id)
        except Exception as e:
            print(f"Error downloading trial {trial_id}: {e}")
            continue
        
        if trial in trials_info_alignment:
            # print("Skipping trial {} because it is an alignment trial.".format(trial))
            # continue
            # Align markers with ground.
            suffixOutputFileName = 'aligned'
            trialName_aligned = trialName + '_' + suffixOutputFileName
            
            angle = None
            if trial in trials_manual_alignment:
                angle = trials_manual_alignment[trial]['angle']
                
            select_window = []
            if trial in trials_select_window:
                select_window = trials_select_window[trial]
                
            
            
            # Do if not already done or if overwrite_aligned_data is True.
            if not os.path.exists(os.path.join(sessionDir, 'OpenSimData', 'Kinematics', trialName_aligned + '.mot')) or overwrite_aligned_data:
                print('Aligning markers with ground...')     
                try:       
                    pathTRCFile_out = align_markers_with_ground_3(
                        sessionDir, trialName,
                        suffixOutputFileName=suffixOutputFileName,
                        lowpass_cutoff_frequency_for_marker_values=filter_frequency,
                        angle=angle, select_window=select_window)
                    # Run inverse kinematics.
                    print('Running inverse kinematics...')
                    pathGenericSetupFile = os.path.join(
                        baseDir, 'OpenSimPipeline', 
                        'InverseKinematics', 'Setup_InverseKinematics.xml')
                    pathScaledModel = os.path.join(sessionDir, 'OpenSimData', 'Model', modelName)        
                    runIKTool(pathGenericSetupFile, pathScaledModel, pathTRCFile_out, pathKinematicsFolder)
                except Exception as e:
                    print(f"Error alignement trial {trial_id}: {e}")
                    continue
        else:
            trialName_aligned = trialName
                    
        # Data processing.
        print('Processing data...')        
        for leg in legs:
            if trial in trials_info_problems.keys():
                if leg in trials_info_problems[trial]['leg']:
                    print('Skipping leg {} for trial {} because it is a problem trial.'.format(leg, trial))
                    continue

            case_leg = '{}_{}'.format(case, leg)
            # Gait segmentation and analysis.
            pathOutputJsonFile = os.path.join(pathKinematicsFolder, 'gaitResults_{}.json'.format(leg))
            # Do if not already done.
            if not os.path.exists(pathOutputJsonFile) or overwrite_gait_results:
                try:
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
                    # Make sure there is a buffer after the last event, otherwise select the previous cycle.
                    coordinateValues = gait.coordinateValues
                    time = coordinateValues['time'].to_numpy()
                    
                    select_previous_cycle = False
                    if trial in trials_select_previous_cycle:
                        if leg in trials_select_previous_cycle[trial]['leg']:
                            select_previous_cycle = True
                    
                    if time[-1] - gaitResults['events']['ipsilateralTime'][-1] < buffer_end or select_previous_cycle:                    
                        gait = gait_analysis(
                            sessionDir, trialName_aligned, leg=leg,
                            lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
                            n_gait_cycles=2)
                        # Compute scalars.
                        # TODO: should we compute scalars for the previous cycle only, now it is average of both I guess.
                        gaitResults['scalars'] = gait.compute_scalars(scalar_names)
                        # Get gait events.
                        gaitResults['events'] = gait.get_gait_events()
                        # Support serialization for json
                        gaitResults['events']['ipsilateralTime'] = [float(i) for i in gaitResults['events']['ipsilateralTime'][1,:].flatten()]
                        gaitResults['events']['contralateralTime'] = [float(i) for i in gaitResults['events']['contralateralTime'][1,:].flatten()]
                        gaitResults['events']['contralateralIdx'] = [int(i) for i in gaitResults['events']['contralateralIdx'][1,:].flatten()]
                        gaitResults['events']['ipsilateralIdx'] = [int(i) for i in gaitResults['events']['ipsilateralIdx'][1,:].flatten()]
                        
                        print('Buffer after the last event is less than 0.3s for the last gait cycle or bad data, selecting the previous cycle.')
                        if time[-1] - gaitResults['events']['ipsilateralTime'][-1] < buffer_end:
                            raise ValueError('Buffer after the last event is less than {}s for the selected gait cycle, please check the data.'.format(buffer_end))                
                    # Dump gaitResults dict in Json file and save in pathKinematicsFolder.            
                    with open(pathOutputJsonFile, 'w') as outfile:
                        json.dump(gaitResults, outfile)
                except Exception as e:
                    print(f"Error gait analysis trial {trial_id}: {e}")
                    continue
            else:
                with open(pathOutputJsonFile) as json_file:
                    gaitResults = json.load(json_file)

            # Setup dynamic optimization problem.
            time_window = [
                gaitResults['events']['ipsilateralTime'][0],
                gaitResults['events']['ipsilateralTime'][-1]]
            # Adjust time window to add buffers
            time_window[0] = time_window[0] - buffer_start
            time_window[1] = time_window[1] + buffer_end
            
            # Creating mot files for visualization.
            pathResults = os.path.join(sessionDir, 'OpenSimData', 'Dynamics', trialName_aligned)
            pathTrackedMotionFile = os.path.join(pathResults, 'kinematics_to_track_{}.mot'.format(case_leg))            
            if not os.path.exists(pathTrackedMotionFile) or overwrite_tracked_motion_file:
                gait = gait_analysis(
                    sessionDir, trialName_aligned, leg=leg,
                    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
                    n_gait_cycles=n_gait_cycles)
                coordinateValues = gait.coordinateValues
                time = coordinateValues['time'].to_numpy()
                labels = coordinateValues.columns
                idx_start = (np.abs(time - time_window[0])).argmin()
                idx_end = (np.abs(time - time_window[1])).argmin() + 1
                data = coordinateValues.to_numpy()[idx_start:idx_end, :]
                os.makedirs(pathResults, exist_ok=True)
                numpy_to_storage(labels, data, pathTrackedMotionFile, datatype='IK')

            print('Processing data for dynamic simulation...')
            try:
                settings = processInputsOpenSimAD(
                    baseDir, sessionDir, session_id, trialName_aligned, 
                    motion_type, time_window=time_window)
            except Exception as e:
                print(f"Error setting up dynamic optimization for trial {trial_id}: {e}")
                continue
        
            # Simulation.
            try:
                run_tracking(baseDir, sessionDir, settings, case=case_leg, 
                            solveProblem=solveProblem, analyzeResults=analyzeResults)
                test=1
            except Exception as e:
                print(f"Error during dynamic optimization for trial {trial_id}: {e}")
                continue
            test=1
        
    else:
        if trial in trials_info_alignment:
            suffixOutputFileName = 'aligned'
            trialName_aligned = trial_name + '_' + suffixOutputFileName
        else:
            trialName_aligned = trial_name
        plotResultsOpenSimAD(sessionDir, trialName_aligned, cases=['2_r', '2_l'], mainPlots=True)
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