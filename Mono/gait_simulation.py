# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:35:00 2023

@author: Scott Uhlrich
"""

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
from scipy.spatial.transform import Rotation as R


from gait_analysis import gait_analysis
from utils import get_trial_id, download_trial, numpy_to_storage, get_model_name_from_metadata
from utilsPlotting import plot_dataframe

from utilsTRC import trc_2_dict, dict_to_trc

from utilsProcessing import align_markers_with_ground_3
from utilsOpenSim import runIKTool

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking

from utilsKineticsOpenSimAD import kineticsOpenSimAD 
from utilsKinematics import kinematics


#%% User inputs
# dataDir = 'C:/SharedGdrive/sparseIK/Data/'
dataDir = 'C:/Users/hpl/Documents/MyRepositories/opencap-processing/Data/'


# # walkingTS2 OpenCapSubject4_ts
# trialName = 'walkingTS2_frontal'
# sessionFolder = 'OpenCapSubject4_ts_frontal'
# timeOverride = True
# time_window_override = [.30, 1.36] # walking4
# case = '101'
# single_gait_cycle = True
# leg = 'l'
# manual_align = 0

# # walkingTS2 OpenCapSubject4_ts
# trialName = 'walkingTS2_sagittalRight'
# sessionFolder = 'OpenCapSubject4_ts_sagittalRight'
# timeOverride = True
# time_window_override = [.1 , 1.18] # walking4
# case = '101'
# single_gait_cycle = True
# leg = 'l'
# manual_align = 0

# # walkingTS2 OpenCapSubject4_ts
trialName = 'walking4'
sessionFolder = 'monosquats'
timeOverride = True
time_window_override = [0 , 1.5] # walking4
case = '101'
single_gait_cycle = True
leg = 'l'
manual_align = 0


# OpencapSubject4 'walking4'
# trialName = 'walking4'
# sessionFolder = 'OpenCapSubject4'
# timeOverride = True
# time_window_override = [.1, 1.28] # walking4
# case = '101'
# single_gait_cycle = True
# leg = 'l'
# manual_align = -.4


# walkingTS2 OpenCapSubject4_ts
# trialName = 'walkingTS2'
# sessionFolder = 'OpenCapSubject4_ts'
# timeOverride = True
# time_window_override = [.127, 1.33] # walking4
# case = '101'
# single_gait_cycle = True
# leg = 'l'
# manual_align = -1.3

# Scott walking
# time_window_override = [1, 2.5] 
# case = '102'
# single_gait_cycle = False
# leg = 'l'
# manual_align = 0


sessionDir = os.path.join(dataDir,sessionFolder)

scalar_names = {'gait_speed','stride_length','step_width','cadence',
                'single_support_time','double_support_time'}

# Select how many gait cycles you'd like to analyze. Select -1 for all gait
# cycles detected in the trial.
n_gait_cycles = 1 

# Select lowpass filter frequency for kinematics data.
filter_frequency = 6

# Settings for dynamic simulation.
# motion_type = 'walking_formulation1'
# case = '101'
# legs = ['r', 'l']
runProblem = True
overwrite_aligned_data = True
overwrite_gait_results = False
overwrite_tracked_motion_file = True
processInputs = True
runSimulation = True
solveProblem = True
analyzeResults = True

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
elif case == '101': # for manual override periodic
    buffer_start = 0
    buffer_end = 0
    motion_type = 'walking_formulation1_periodic'
elif case == 102:
    buffer_start = 0
    buffer_end = 0
    motion_type = 'walking_formulation1'
    



if runProblem:
    opensimDir = os.path.join(sessionDir,'OpenSimData')
    
    
    # Download data.
    # try:
    #     trialName, modelName = download_trial(trial_id, sessionDir, session_id=session_id)
    # except Exception as e:
    #     print(f"Error downloading trial {trial_id}: {e}")
    #     continue
    
    # We align all trials.
    # if trial in trials_info_alignment:
    # print("Skipping trial {} because it is an alignment trial.".format(trial))
    # continue
    # Align markers with ground.
    suffixOutputFileName = 'aligned'
    trialName_aligned = trialName + '_' + suffixOutputFileName
    
    # Initiate gait analysis to compute alignment
    gaitResults = {}
    gait = gait_analysis(
        sessionDir, trialName, leg=leg,
        lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
        n_gait_cycles=3,marker_set='mono',single_gait_cycle=single_gait_cycle,
        manual_hs_times = time_window_override)
    
    # assuming straight line walking
    gait_frame = np.mean(gait.compute_gait_frame(),axis=0)
    
    # trc paths
    trc_path = os.path.join(sessionDir,'MarkerData',trialName + '.trc')
    new_trc_path = trc_path[:-4] + '_aligned.trc'
    
    # align the trc
    if overwrite_aligned_data or not os.path.exists(new_trc_path):
        # load trc
        trc_dict = trc_2_dict(trc_path)
        
        # Rotate by the inverse of the average gait frame -- just for overground walking
        euler_world_to_gait =R.from_matrix(gait_frame.T).as_euler('YXZ',degrees=True)
        
        rotationAngles = {ax:val for (ax,val) in zip(['Y','X','Z'],euler_world_to_gait)}
        rotationAngles['Z']+= manual_align
            
        offset_markers = ['r_calc','r_toe','l_calc','l_toe']
        offset = dict_to_trc(trc_dict, new_trc_path, rotationAngles = rotationAngles,
                    offset_markers=offset_markers, offset_axis='y')
    
    # re-run IK
    kinematicsDir = os.path.join(opensimDir,'Kinematics')
    model_name = get_model_name_from_metadata(sessionDir)
    
    if overwrite_aligned_data or not os.path.exists(os.path.join(kinematicsDir,trialName + '_aligned.mot')):
        print('Running inverse kinematics...')
        pathGenericSetupFile = os.path.join(
            baseDir, 'OpenSimPipeline', 
            'InverseKinematics', 'Setup_InverseKinematics_Mono.xml')
        pathScaledModel = os.path.join(opensimDir, 'Model', model_name)        
        runIKTool(pathGenericSetupFile, pathScaledModel, new_trc_path, kinematicsDir)
      
    # Times 
    # TODO CHECK FOR BUFFER SPACE
    # Setup dynamic optimization problem.
    gaitResults['events'] = gait.get_gait_events()
    # Support serialization for json
    gaitResults['events']['ipsilateralTime'] = [float(i) for i in gaitResults['events']['ipsilateralTime'].flatten()]
    gaitResults['events']['contralateralTime'] = [float(i) for i in gaitResults['events']['contralateralTime'].flatten()]
    gaitResults['events']['contralateralIdx'] = [int(i) for i in gaitResults['events']['contralateralIdx'].flatten()]
    gaitResults['events']['ipsilateralIdx'] = [int(i) for i in gaitResults['events']['ipsilateralIdx'].flatten()]    
    time_window = [
                    gaitResults['events']['ipsilateralTime'][0],
                    gaitResults['events']['ipsilateralTime'][-1]]
    # Adjust time window to add buffers
    time_window[0] = time_window[0] - buffer_start
    time_window[1] = time_window[1] + buffer_end
    
    # TODO DELETE
    if timeOverride:
        print('OVERRIDING TIME WINDOW!')
        time_window = time_window_override
    
    
    
    
    
    
    
    # angle = None
    # if trial in trials_manual_alignment:
    #     angle = trials_manual_alignment[trial]['angle']
        
    # select_window = []
    # if trial in trials_select_window:
    #     select_window = trials_select_window[trial]
    
    # # Do if not already done or if overwrite_aligned_data is True.
    # if not os.path.exists(os.path.join(sessionDir, 'OpenSimData', 'Kinematics', trialName_aligned + '.mot')) or overwrite_aligned_data:
    #     print('Aligning markers with ground...')     
    #     try:       
    #         pathTRCFile_out = align_markers_with_ground_3(
    #             sessionDir, trialName,
    #             suffixOutputFileName=suffixOutputFileName,
    #             lowpass_cutoff_frequency_for_marker_values=filter_frequency,
    #             angle=angle, select_window=select_window)
    #         # Run inverse kinematics.
    #         print('Running inverse kinematics...')
    #         pathGenericSetupFile = os.path.join(
    #             baseDir, 'OpenSimPipeline', 
    #             'InverseKinematics', 'Setup_InverseKinematics.xml')
    #         pathScaledModel = os.path.join(sessionDir, 'OpenSimData', 'Model', modelName)        
    #         runIKTool(pathGenericSetupFile, pathScaledModel, pathTRCFile_out, pathKinematicsFolder)
    #     except Exception as e:
    #         print(f"Error alignement trial {trialName}: {e}")
    #         raise Exception('alignment')
    # # else:
    # #     trialName_aligned = trialName
                
    # # Data processing.
    # print('Processing data...')        
    # for leg in legs:
    #     if trial in trials_info_problems.keys():
    #         if leg in trials_info_problems[trial]['leg']:
    #             print('Skipping leg {} for trial {} because it is a problem trial.'.format(leg, trial))
    #             continue
        
    #     # Gait segmentation and analysis.
    #     # Purposefuly save with trialName and not trialName_aligned, since trialName is the trial actually being analysed.
    #     pathOutputJsonFile = os.path.join(pathKinematicsFolder, '{}_kinematic_features_{}.json'.format(trialName, leg))
    #     # Do if not already done.
    #     if not os.path.exists(pathOutputJsonFile) or overwrite_gait_results:
    #         try:
    #             gaitResults = {}
    #             gait = gait_analysis(
    #                 sessionDir, trialName_aligned, leg=leg,
    #                 lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    #                 n_gait_cycles=n_gait_cycles)
    #             # Compute scalars.
    #             gaitResults['scalars'] = gait.compute_scalars(scalar_names)
    #             # Get gait events.
    #             gaitResults['events'] = gait.get_gait_events()
    #             # Support serialization for json
    #             gaitResults['events']['ipsilateralTime'] = [float(i) for i in gaitResults['events']['ipsilateralTime'].flatten()]
    #             gaitResults['events']['contralateralTime'] = [float(i) for i in gaitResults['events']['contralateralTime'].flatten()]
    #             gaitResults['events']['contralateralIdx'] = [int(i) for i in gaitResults['events']['contralateralIdx'].flatten()]
    #             gaitResults['events']['ipsilateralIdx'] = [int(i) for i in gaitResults['events']['ipsilateralIdx'].flatten()]
    #             # Make sure there is a buffer after the last event, otherwise select the previous cycle.
    #             coordinateValues = gait.coordinateValues
    #             time = coordinateValues['time'].to_numpy()
                
    #             select_previous_cycle = False
    #             if trial in trials_select_previous_cycle:
    #                 if leg in trials_select_previous_cycle[trial]['leg']:
    #                     select_previous_cycle = True
                
    #             if time[-1] - gaitResults['events']['ipsilateralTime'][-1] < buffer_end or select_previous_cycle:                    
    #                 gait = gait_analysis(
    #                     sessionDir, trialName_aligned, leg=leg,
    #                     lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    #                     n_gait_cycles=2)
    #                 # Compute scalars.
    #                 gaitResults['scalars'] = gait.compute_scalars(scalar_names, return_all=True)
    #                 # Only extract last value (penultimate gait cycle)
    #                 for scalar_name in scalar_names:
    #                     gaitResults['scalars'][scalar_name]['value'] = gaitResults['scalars'][scalar_name]['value'][-1]
    #                     # If array then only extract last value
    #                     if isinstance(gaitResults['scalars'][scalar_name]['value'], np.ndarray):
    #                         gaitResults['scalars'][scalar_name]['value'] = gaitResults['scalars'][scalar_name]['value'][-1]
    #                 # Get gait events.
    #                 gaitResults['events'] = gait.get_gait_events()
    #                 # Support serialization for json
    #                 gaitResults['events']['ipsilateralTime'] = [float(i) for i in gaitResults['events']['ipsilateralTime'][1,:].flatten()]
    #                 gaitResults['events']['contralateralTime'] = [float(i) for i in gaitResults['events']['contralateralTime'][1,:].flatten()]
    #                 gaitResults['events']['contralateralIdx'] = [int(i) for i in gaitResults['events']['contralateralIdx'][1,:].flatten()]
    #                 gaitResults['events']['ipsilateralIdx'] = [int(i) for i in gaitResults['events']['ipsilateralIdx'][1,:].flatten()]
                    
    #                 print('Buffer after the last event is less than 0.3s for the last gait cycle or bad data, selecting the previous cycle.')
    #                 if time[-1] - gaitResults['events']['ipsilateralTime'][-1] < buffer_end:
    #                     raise ValueError('Buffer after the last event is less than {}s for the selected gait cycle, please check the data.'.format(buffer_end))                
    #             # Dump gaitResults dict in Json file and save in pathKinematicsFolder.            
    #             with open(pathOutputJsonFile, 'w') as outfile:
    #                 json.dump(gaitResults, outfile)
    #         except Exception as e:
    #             print(f"Error gait analysis trial {trial_id}: {e}")
    #             continue
    #     else:
    #         with open(pathOutputJsonFile) as json_file:
    #             gaitResults = json.load(json_file)

    #     # # Temporary check to see if something has changed
    #     # temp1 = gaitResults['events']['ipsilateralTime']
    #     # pathOutputJsonFile_old =  os.path.join(pathKinematicsFolder, 'gaitResults_{}.json'.format(leg))
    #     # with open(pathOutputJsonFile_old) as json_file:
    #     #     gaitResults_old = json.load(json_file)
    #     # temp2 = gaitResults_old['events']['ipsilateralTime']
    #     # # Check that both lists are the same
    #     # if temp1 != temp2:
    #     #     raise ValueError('Something has changed in the gait analysis, please check {}.'.format(session_id))            

    #     # Setup dynamic optimization problem.
    #     time_window = [
    #         gaitResults['events']['ipsilateralTime'][0],
    #         gaitResults['events']['ipsilateralTime'][-1]]
    #     # Adjust time window to add buffers
    #     time_window[0] = time_window[0] - buffer_start
    #     time_window[1] = time_window[1] + buffer_end
        
        # Creating mot files for visualization.
    pathResults = os.path.join(sessionDir, 'OpenSimData', 'Dynamics', trialName_aligned)
    case_leg = '{}_{}'.format(case, leg)
    pathTrackedMotionFile = os.path.join(pathResults, 'kinematics_to_track_{}.mot'.format(case_leg))            
    if not os.path.exists(pathTrackedMotionFile) or overwrite_tracked_motion_file:
        kinematics_forVis = kinematics(
            sessionDir, trialName_aligned, lowpass_cutoff_frequency_for_coordinate_values=filter_frequency)
        coordinateValues = kinematics_forVis.get_coordinate_values()
        time = coordinateValues['time'].to_numpy()
        labels = coordinateValues.columns
        idx_start = (np.abs(time - time_window[0])).argmin()
        idx_end = (np.abs(time - time_window[1])).argmin() + 1
        data = coordinateValues.to_numpy()[idx_start:idx_end, :]
        os.makedirs(pathResults, exist_ok=True)
        numpy_to_storage(labels, data, pathTrackedMotionFile, datatype='IK')

    print('Processing data for dynamic simulation...')
    if processInputs:
        # try:
        settings = processInputsOpenSimAD(
            baseDir, dataDir, sessionFolder, trialName_aligned, 
            motion_type, time_window=time_window)
        # except Exception as e:
        #     print(f"Error setting up dynamic optimization for trial {trialName}: {e}")
        #     # continue

    # Simulation.
    if runSimulation:
        # try:
        run_tracking(baseDir, dataDir, sessionFolder, settings, case=case_leg, 
                    solveProblem=solveProblem, analyzeResults=analyzeResults)
        #     test=1
        # except Exception as e:
        #     print(f"Error during dynamic optimization for trial {trialName}: {e}")
        #     # continue
    test=1

# else:
#     # if trial in trials_info_alignment:
#     suffixOutputFileName = 'aligned'
#     trialName_aligned = trial_name + '_' + suffixOutputFileName
#     # else:
#     #     trialName_aligned = trial_name
#     plotResultsOpenSimAD(sessionDir, trialName_aligned, cases=['2_r', '2_l'], mainPlots=True)
#     test=1