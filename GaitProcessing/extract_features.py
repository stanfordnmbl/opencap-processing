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
import copy
import numpy as np
from matplotlib import pyplot as plt

from scipy.integrate import cumtrapz

from utilsProcessing import lowPassFilterDataframe

from data_info import get_data_info, get_data_info_problems, get_data_alignment, get_data_select_previous_cycle
from data_info import get_data_manual_alignment, get_data_case

from utilsKineticsOpenSimAD import kineticsOpenSimAD

# %% Paths.
# driveDir = 'C:/MyDriveSym/Projects/ParkerStudy'
driveDir = 'G:/.shortcut-targets-by-id/1PsjYe9HAdckqeTmAhxFd6F7Oad1qgZNy/ParkerStudy'
dataFolder = os.path.join(driveDir, 'Data')

# %% User-defined variables.
# Select lowpass filter frequency for kinematics data.
filter_frequency = 6

legs = ['l', 'r']
suffixOutputFileName = 'aligned'
coordinates = ['hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle', 'ankle_angle']
coordinates_gaitCycle = ['arm_flex']

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

    # initialize for between-leg comparisons
    resultsBilateral = {}
    features = {}
    
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
        
        # Filter
        joint_angles_filt = joint_angles # these were already filtered prior to simulation
        joint_moments_filt = lowPassFilterDataframe(joint_moments, filter_frequency)
        joint_powers_filt = lowPassFilterDataframe(joint_powers, filter_frequency)
        
        # Select time window.
        joint_angles_filt_sel = joint_angles_filt.loc[(joint_angles_filt['time'] >= time_window[0]) & (joint_angles_filt['time'] <= time_window[1])]
        joint_moments_filt_sel = joint_moments_filt.loc[(joint_moments_filt['time'] >= time_window[0]) & (joint_moments_filt['time'] <= time_window[1])]
        joint_powers_filt_sel = joint_powers_filt.loc[(joint_powers_filt['time'] >= time_window[0]) & (joint_powers_filt['time'] <= time_window[1])]
        joint_angles_filt_sel = joint_angles_filt_sel.reset_index(drop=True)
        joint_moments_filt_sel = joint_moments_filt_sel.reset_index(drop=True)
        joint_powers_filt_sel = joint_powers_filt_sel.reset_index(drop=True)
        
        # for both sides
        resultsBilateral[leg] = {}
        resultsBilateral[leg]['joint_angles'] = copy.copy(joint_angles_filt_sel)
        resultsBilateral[leg]['joint_moments'] = copy.copy(joint_moments_filt_sel)
        resultsBilateral[leg]['joint_powers'] = copy.copy(joint_powers_filt_sel)

        # Stance and swing.
        # Times.
        hs_1_time = gaitResults['events']['ipsilateralTime'][0]
        to_1_time = gaitResults['events']['ipsilateralTime'][1]
        hs_2_time = gaitResults['events']['ipsilateralTime'][2]
        # Find indices in dataframe.
        hs_1_idx = (np.abs(joint_angles_filt_sel['time'] - hs_1_time)).argmin()
        to_1_idx = (np.abs(joint_angles_filt_sel['time'] - to_1_time)).argmin()
        hs_2_idx = (np.abs(joint_angles_filt_sel['time'] - hs_2_time)).argmin()
        
        # Kinetic features (loosely defined...results that use simulation results)
        kinetic_features = {}
        # Foot drop: dorsiflexion angle in mid-swing
        idx_midSwing = np.arange(int(np.round(to_1_idx + 0.4 * (hs_2_idx - to_1_idx))),
                        int(np.round(to_1_idx + 0.6 * (hs_2_idx - to_1_idx))))
        dorsiflexion_angle_midswing = np.mean(joint_angles_filt_sel['ankle_angle_' + leg].to_numpy()[idx_midSwing])
        kinetic_features['dorsiflexion_angle_midswing'] = {}
        kinetic_features['dorsiflexion_angle_midswing']['value'] = dorsiflexion_angle_midswing
        kinetic_features['dorsiflexion_angle_midswing']['units'] = 'deg'
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
        # Shoulder flexion range during stance.
        kinetic_features['arm_flexion_rom'] = {}
        kinetic_features['arm_flexion_rom']['value'] = np.ptp(joint_angles_filt_sel['arm_flex_' + leg])
        kinetic_features['arm_flexion_rom']['units'] = 'deg'
        
        # Peak moments and impulses.
        peak_moment_stance, peak_moment_swing = {}, {}
        peak_power_stance, peak_power_swing = {}, {}
        impulse_stance = {}
        for coordinate in coordinates:
            # Moments.
            moment_stance = joint_moments_filt_sel[coordinate + '_' + leg].to_numpy()[hs_1_idx:to_1_idx]
            moment_swing = joint_moments_filt_sel[coordinate + '_' + leg].to_numpy()[to_1_idx:hs_2_idx]
            # Peak moments.
            peak_moment_stance[coordinate] = [np.min(moment_stance),np.max(moment_stance)]
            kinetic_features['peak_moment_stance_positive_' + coordinate] = {}
            kinetic_features['peak_moment_stance_positive_' + coordinate]['value'] = peak_moment_stance[coordinate][1]
            kinetic_features['peak_moment_stance_positive_' + coordinate]['units'] = 'Nm'
            kinetic_features['peak_moment_stance_negative_' + coordinate] = {}
            kinetic_features['peak_moment_stance_negative_' + coordinate]['value'] = peak_moment_stance[coordinate][0]
            kinetic_features['peak_moment_stance_negative_' + coordinate]['units'] = 'Nm'
            
            peak_moment_swing[coordinate] = [np.min(moment_swing), np.max(moment_swing)]
            kinetic_features['peak_moment_swing_positive_' + coordinate] = {}
            kinetic_features['peak_moment_swing_positive_' + coordinate]['value'] = peak_moment_swing[coordinate][1]
            kinetic_features['peak_moment_swing_positive_' + coordinate]['units'] = 'Nm'
            kinetic_features['peak_moment_swing_negative_' + coordinate] = {}
            kinetic_features['peak_moment_swing_negative_' + coordinate]['value'] = peak_moment_swing[coordinate][0]
            kinetic_features['peak_moment_swing_negative_' + coordinate]['units'] = 'Nm'           
            
            # Inpulses.
            time_stance = joint_moments_filt_sel['time'].to_numpy()[hs_1_idx:to_1_idx]
            dt = np.diff(time_stance)[0]
            impulse_stance[coordinate] = [np.abs(np.mean(moment_stance[moment_stance<0])) * dt*np.sum(moment_stance<0), 
                np.abs(np.mean(moment_stance[moment_stance>0])) * dt*np.sum(moment_stance>0)]
            kinetic_features['impulse_stance_positive_' + coordinate] = {}
            kinetic_features['impulse_stance_positive_' + coordinate]['value'] = impulse_stance[coordinate][1]
            kinetic_features['impulse_stance_positive_' + coordinate]['units'] = 'Nm*s'
            kinetic_features['impulse_stance_negative_' + coordinate] = {}
            kinetic_features['impulse_stance_negative_' + coordinate]['value'] = impulse_stance[coordinate][0]
            kinetic_features['impulse_stance_negative_' + coordinate]['units'] = 'Nm*s'
            
            # Powers.
            power_stance = joint_powers_filt_sel[coordinate + '_' + leg].to_numpy()[hs_1_idx:to_1_idx]
            power_swing = joint_powers_filt_sel[coordinate + '_' + leg].to_numpy()[to_1_idx:hs_2_idx]
            # Peak powers.
            peak_power_stance[coordinate] = [np.min(power_stance), np.max(power_stance)]
            kinetic_features['peak_power_stance_positive_' + coordinate] = {}
            kinetic_features['peak_power_stance_positive_' + coordinate]['value'] = peak_power_stance[coordinate][1]
            kinetic_features['peak_power_stance_positive_' + coordinate]['units'] = 'W'
            kinetic_features['peak_power_stance_negative_' + coordinate] = {}
            kinetic_features['peak_power_stance_negative_' + coordinate]['value'] = peak_power_stance[coordinate][0]
            kinetic_features['peak_power_stance_negative_' + coordinate]['units'] = 'W'
            peak_power_swing[coordinate] = [np.min(power_swing), np.max(power_swing)]
            kinetic_features['peak_power_swing_positive_' + coordinate] = {}
            kinetic_features['peak_power_swing_positive_' + coordinate]['value'] = peak_power_swing[coordinate][1]
            kinetic_features['peak_power_swing_positive_' + coordinate]['units'] = 'W'
            kinetic_features['peak_power_swing_negative_' + coordinate] = {}
            kinetic_features['peak_power_swing_negative_' + coordinate]['value'] = peak_power_swing[coordinate][0]
            kinetic_features['peak_power_swing_negative_' + coordinate]['units'] = 'W'
            
        peak_moment_gaitCycle, impulse_gaitCycle = {}, {}

        for coordinate in coordinates_gaitCycle:
            # Moments.
            moment_gaitCycle = joint_moments_filt_sel[coordinate + '_' + leg].to_numpy()[hs_1_idx:hs_2_idx]
            plt.plot(moment_gaitCycle,label=leg)

            # Peak moments.
            peak_moment_gaitCycle[coordinate] = [np.min(moment_gaitCycle),np.max(moment_gaitCycle)]
            kinetic_features['peak_moment_gaitCycle_positive_' + coordinate] = {}
            kinetic_features['peak_moment_gaitCycle_positive_' + coordinate]['value'] = peak_moment_gaitCycle[coordinate][1]
            kinetic_features['peak_moment_gaitCycle_positive_' + coordinate]['units'] = 'Nm'
            kinetic_features['peak_moment_gaitCycle_negative_' + coordinate] = {}
            kinetic_features['peak_moment_gaitCycle_negative_' + coordinate]['value'] = peak_moment_gaitCycle[coordinate][0]
            kinetic_features['peak_moment_gaitCycle_negative_' + coordinate]['units'] = 'Nm'
            
            # Inpulses.
            time_gaitCycle = joint_moments_filt_sel['time'].to_numpy()[hs_1_idx:hs_2_idx]
            dt = np.diff(time_stance)[0]
            impulse_gaitCycle[coordinate] = [np.abs(np.mean(moment_gaitCycle[moment_gaitCycle<0])) * dt*np.sum(moment_gaitCycle<0), 
                np.abs(np.mean(moment_gaitCycle[moment_gaitCycle>0])) * dt*np.sum(moment_gaitCycle>0)]
            kinetic_features['impulse_gaitCycle_positive_' + coordinate] = {}
            kinetic_features['impulse_gaitCycle_positive_' + coordinate]['value'] = impulse_gaitCycle[coordinate][1]
            kinetic_features['impulse_gaitCycle_positive_' + coordinate]['units'] = 'Nm*s'
            kinetic_features['impulse_gaitCycle_negative_' + coordinate] = {}
            kinetic_features['impulse_gaitCycle_negative_' + coordinate]['value'] = impulse_gaitCycle[coordinate][0]
            kinetic_features['impulse_gaitCycle_negative_' + coordinate]['units'] = 'Nm*s'
        
        features_leg = {}
        features_leg.update(kinetic_features)
        features_leg.update(gaitResults['scalars'])
        features.update({key + '_' + leg: value for key, value in features_leg.items()})

    # %% function that computes weighted correlations between related signals.
    # ie left arm_flex angle from a left gait cycle vs a right arm_flex angle from
    # a right gait cycle
    
    def compute_correlations(df1, df2, cols_to_compare=None, visualize=False):
        if cols_to_compare is None:
            cols_to_compare = df1.columns
    
        # Interpolating both dataframes to have 101 rows for each column
        df1_interpolated = df1.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=100)
        df2_interpolated = df2.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=100)
    
        # Computing the correlation between appropriate columns in both dataframes
        correlations = {}
        total_weighted_correlation = 0
        total_weight = 0
    
        for col1 in df1_interpolated.columns:
            if any(col1.startswith(col_compare) for col_compare in cols_to_compare) and col1.endswith('_r'):
                corresponding_col = col1[:-2] + '_l'
                if corresponding_col in df2_interpolated.columns:
                    signal1 = df1_interpolated[col1]
                    signal2 = df2_interpolated[corresponding_col]
    
                    max_range_signal1 = np.ptp(signal1)
                    max_range_signal2 = np.ptp(signal2)
                    max_range = max(max_range_signal1, max_range_signal2)
    
                    mean_abs_error = np.mean(np.abs(signal1 - signal2)) / max_range
    
                    correlation = signal1.corr(signal2)
                    weight = 1 - mean_abs_error
    
                    weighted_correlation = correlation * weight
                    correlations[col1] = weighted_correlation
    
                    total_weighted_correlation += weighted_correlation
    
                    # Plotting the signals if visualize is True
                    if visualize:
                        plt.figure(figsize=(8, 5))
                        plt.plot(signal1, label='df1')
                        plt.plot(signal2, label='df2')
                        plt.title(f"Comparison between {col1} and {corresponding_col} with weighted correlation {weighted_correlation}")
                        plt.legend()
                        plt.show()
    
        for col2 in df2_interpolated.columns:
            if any(col2.startswith(col_compare) for col_compare in cols_to_compare) and col2.endswith('_r'):
                corresponding_col = col2[:-2] + '_l'
                if corresponding_col in df1_interpolated.columns:
                    signal1 = df1_interpolated[corresponding_col]
                    signal2 = df2_interpolated[col2]
    
                    max_range_signal1 = np.ptp(signal1)
                    max_range_signal2 = np.ptp(signal2)
                    max_range = max(max_range_signal1, max_range_signal2)
    
                    mean_abs_error = np.mean(np.abs(signal1 - signal2)) / max_range
    
                    correlation = signal1.corr(signal2)
                    weight = 1 - mean_abs_error
    
                    weighted_correlation = correlation * weight
                    correlations[corresponding_col] = weighted_correlation
    
                    total_weighted_correlation += weighted_correlation
    
                    # Plotting the signals if visualize is True
                    if visualize:
                        plt.figure(figsize=(8, 5))
                        plt.plot(signal1, label='df1')
                        plt.plot(signal2, label='df2')
                        plt.title(f"Comparison between {corresponding_col} and {col2} with weighted correlation {weighted_correlation}")
                        plt.legend()
                        plt.show()
    
        mean_weighted_correlation = total_weighted_correlation / len(correlations)
        return correlations, mean_weighted_correlation
            
    combined_features = {}
    
    # Compute arm swing asymmetry   
    dofs = ['arm_flex','elbow_flex']
    correlations, meanCorrelation = compute_correlations(resultsBilateral['r']['joint_angles'],
                                       resultsBilateral['l']['joint_angles'],
                                       cols_to_compare=dofs)
    
    combined_features['arm_swing_symmetry'] = {}
    combined_features['arm_swing_symmetry']['value'] = meanCorrelation
    combined_features['arm_swing_symmetry']['units'] = 'unitless'
    
    # add to feature set
    features.update(combined_features)

    # Save features in json file.
    pathFeaturesFolder = os.path.join(sessionDir, 'OpenSimData', 'Features')
    os.makedirs(pathFeaturesFolder, exist_ok=True)
    pathOutputJsonFile = os.path.join(pathFeaturesFolder, 'features.json')
    with open(pathOutputJsonFile, 'w') as outfile:
        json.dump(features, outfile)