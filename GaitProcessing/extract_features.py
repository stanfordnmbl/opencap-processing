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
sys.path.append("../UtilsDynamicSimulations/OpenSimAD")
baseDir = os.path.join(os.getcwd(), '..')
sys.path.append(baseDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)
import json
import numpy as np

from scipy.integrate import cumtrapz

from utilsProcessing import lowPassFilterDataframe

from data_info import get_data_info, get_data_info_problems, get_data_alignment, get_data_select_previous_cycle
from data_info import get_data_manual_alignment, get_data_case

from utilsKineticsOpenSimAD import kineticsOpenSimAD

# %% Paths.
driveDir = 'C:/MyDriveSym/Projects/ParkerStudy'
dataFolder = os.path.join(driveDir, 'Data')

# %% User-defined variables.
# Select lowpass filter frequency for kinematics data.
filter_frequency = 6

legs = ['l', 'r']
suffixOutputFileName = 'aligned'
coordinates = ['hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle', 'ankle_angle']

# %% Gait segmentation and kinematic analysis.
trials_info = get_data_info(trial_indexes=[i for i in range(0,92)])
# trials_info = get_data_info(trial_indexes=trials_to_run)

trials_info_problems = get_data_info_problems()
trials_info_alignment = get_data_alignment()
# trials_select_previous_cycle = get_data_select_previous_cycle()
trials_case = get_data_case()

for trial in trials_info:
    # Get trial info.
    session_id = trials_info[trial]['sid']
    trial_name = trials_info[trial]['trial']
    print('Processing session {} - trial {}...'.format(session_id, trial_name))
    sessionDir = os.path.join(dataFolder, "{}_{}".format(trial, session_id))

    if trial in trials_info_alignment:        
        trialName_aligned = trial_name + '_' + suffixOutputFileName
    else:
        trialName_aligned = trial_name

    for leg in legs:
        if trial in trials_info_problems.keys():
            if leg in trials_info_problems[trial]['leg']:
                print('Skipping leg {} for trial {} because it is a trial with problems.'.format(leg, trial))
                continue

        case = '2' # default
        if trial in trials_case.keys():
            if leg in trials_case[trial].keys():
                case = trials_case[trial][leg]
                print('Using case {} for trial {} leg {}.'.format(case, trial, leg))
        case_leg = '{}_{}'.format(case, leg)

        # Load gait results (kinematic features).
        pathKinematicsFolder = os.path.join(sessionDir, 'OpenSimData', 'Kinematics')
        pathOutputJsonFile = os.path.join(pathKinematicsFolder, 'gaitResults_{}.json'.format(leg))
        with open(pathOutputJsonFile) as json_file:
            gaitResults = json.load(json_file)

        # Extract time window to later trim simulations.
        time_window = [
            gaitResults['events']['ipsilateralTime'][0],
            gaitResults['events']['ipsilateralTime'][-1]]

        # Retrieve results from the optimal solution using utilsKineticsOpenSimAD.
        opt_sol_obj = kineticsOpenSimAD(sessionDir, trialName_aligned, case_leg)

        # Raw biomechanical data.
        joint_angles = opt_sol_obj.get_coordinate_values()
        joint_moments = opt_sol_obj.get_joint_moments()
        joint_powers = opt_sol_obj.get_joint_powers()
        
        # TODO: do we want to filter?
        # Filter
        joint_angles_filt = lowPassFilterDataframe(joint_angles, filter_frequency)
        joint_moments_filt = lowPassFilterDataframe(joint_moments, filter_frequency)
        joint_powers_filt = lowPassFilterDataframe(joint_powers, filter_frequency)
        
        # Select time window.
        joint_angles_filt_sel = joint_angles_filt.loc[(joint_angles_filt['time'] >= time_window[0]) & (joint_angles_filt['time'] <= time_window[1])]
        joint_moments_filt_sel = joint_moments_filt.loc[(joint_moments_filt['time'] >= time_window[0]) & (joint_moments_filt['time'] <= time_window[1])]
        joint_powers_filt_sel = joint_powers_filt.loc[(joint_powers_filt['time'] >= time_window[0]) & (joint_powers_filt['time'] <= time_window[1])]
        joint_angles_filt_sel = joint_angles_filt_sel.reset_index(drop=True)
        joint_moments_filt_sel = joint_moments_filt_sel.reset_index(drop=True)
        joint_powers_filt_sel = joint_powers_filt_sel.reset_index(drop=True)

        # Stance and swing.
        # Times.
        hs_1_time = gaitResults['events']['ipsilateralTime'][0]
        to_1_time = gaitResults['events']['ipsilateralTime'][1]
        hs_2_time = gaitResults['events']['ipsilateralTime'][2]
        # Find indices in dataframe.
        hs_1_idx = (np.abs(joint_angles_filt_sel['time'] - hs_1_time)).argmin()
        to_1_idx = (np.abs(joint_angles_filt_sel['time'] - to_1_time)).argmin()
        hs_2_idx = (np.abs(joint_angles_filt_sel['time'] - hs_2_time)).argmin()
        
        # Kinetic features
        kinetic_features = {}
        # Foot drop: peak dorsiflexion angle during swing.
        peak_foot_drop_swing = np.max(joint_angles_filt_sel['ankle_angle_' + leg].to_numpy()[to_1_idx:hs_2_idx])
        kinetic_features['peak_foot_drop_swing'] = {}
        kinetic_features['peak_foot_drop_swing']['value'] = peak_foot_drop_swing
        kinetic_features['peak_foot_drop_swing']['units'] = 'deg'
        # Peak knee flexion angle during swing.
        peak_knee_flexion_swing = np.max(joint_angles_filt_sel['knee_angle_' + leg].to_numpy()[to_1_idx:hs_2_idx])
        kinetic_features['peak_knee_flexion_swing'] = {}
        kinetic_features['peak_knee_flexion_swing']['value'] = peak_knee_flexion_swing
        kinetic_features['peak_knee_flexion_swing']['units'] = 'deg'
        # Peak knee flexion angle during stance.
        peak_knee_flexion_stance = np.max(joint_angles_filt_sel['knee_angle_' + leg].to_numpy()[hs_1_idx:to_1_idx])
        kinetic_features['peak_knee_flexion_stance'] = {}
        kinetic_features['peak_knee_flexion_stance']['value'] = peak_knee_flexion_stance
        kinetic_features['peak_knee_flexion_stance']['units'] = 'deg'
        # Peak moments and impulses.
        peak_moment_stance, peak_moment_swing = {}, {}
        peak_power_stance, peak_power_swing = {}, {}
        impulse_stance = {}
        for coordinate in coordinates:
            # Moments.
            moment_stance = joint_moments_filt_sel[coordinate + '_' + leg].to_numpy()[hs_1_idx:to_1_idx]
            moment_swing = joint_moments_filt_sel[coordinate + '_' + leg].to_numpy()[to_1_idx:hs_2_idx]
            # Peak moments.
            peak_moment_stance[coordinate] = np.max(moment_stance)
            kinetic_features['peak_moment_stance_' + coordinate] = {}
            kinetic_features['peak_moment_stance_' + coordinate]['value'] = peak_moment_stance[coordinate]
            kinetic_features['peak_moment_stance_' + coordinate]['units'] = 'Nm'
            peak_moment_swing[coordinate] = np.max(moment_swing)
            kinetic_features['peak_moment_swing_' + coordinate] = {}
            kinetic_features['peak_moment_swing_' + coordinate]['value'] = peak_moment_swing[coordinate]
            kinetic_features['peak_moment_swing_' + coordinate]['units'] = 'Nm'
            # Inpulses.
            time_stance = joint_moments_filt_sel['time'].to_numpy()[hs_1_idx:to_1_idx]
            # TODO: Scott, abs of sum?
            impulse_stance[coordinate] = np.abs(np.sum(cumtrapz(moment_stance, time_stance, initial=0)))
            kinetic_features['impulse_stance_' + coordinate] = {}
            kinetic_features['impulse_stance_' + coordinate]['value'] = impulse_stance[coordinate]
            kinetic_features['impulse_stance_' + coordinate]['units'] = 'Nm*s'
            # Powers.
            power_stance = joint_powers_filt_sel[coordinate + '_' + leg].to_numpy()[hs_1_idx:to_1_idx]
            power_swing = joint_powers_filt_sel[coordinate + '_' + leg].to_numpy()[to_1_idx:hs_2_idx]
            # Peak powers.
            peak_power_stance[coordinate] = np.max(power_stance)
            kinetic_features['peak_power_stance_' + coordinate] = {}
            kinetic_features['peak_power_stance_' + coordinate]['value'] = peak_power_stance[coordinate]
            kinetic_features['peak_power_stance_' + coordinate]['units'] = 'W'
            peak_power_swing[coordinate] = np.max(power_swing)
            kinetic_features['peak_power_swing_' + coordinate] = {}
            kinetic_features['peak_power_swing_' + coordinate]['value'] = peak_power_swing[coordinate]
            kinetic_features['peak_power_swing_' + coordinate]['units'] = 'W'
        
        features = {}
        features.update(kinetic_features)
        features.update(gaitResults['scalars'])

        # Save features in json file.
        pathFeaturesFolder = os.path.join(sessionDir, 'OpenSimData', 'Features')
        os.makedirs(pathFeaturesFolder, exist_ok=True)
        pathOutputJsonFile = os.path.join(pathFeaturesFolder, 'features_{}.json'.format(leg))
        with open(pathOutputJsonFile, 'w') as outfile:
            json.dump(features, outfile)
