'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_gait_analysis.py
    ---------------------------------------------------------------------------
    Copyright 2023 Stanford University and the Authors
    
    Author(s): Scott Uhlrich & Antoine Falisse
    
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
sys.path.append("../ActivityAnalyses")
baseDir = os.path.join(os.getcwd(), '..')
sys.path.append(baseDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)
import json
import copy
import numpy as np
from matplotlib import pyplot as plt
import shutil

from scipy.integrate import trapz

from utilsProcessing import lowPassFilterDataframe

from data_info import get_data_info, get_data_info_problems, get_data_alignment, get_data_select_previous_cycle
from data_info import get_data_manual_alignment, get_data_case

from utilsKineticsOpenSimAD import kineticsOpenSimAD
from utils import get_trial_id, download_trial
from gait_analysis import gait_analysis

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

# Kinematic scalar names
scalar_names = {'gait_speed','stride_length','step_width','cadence',
                'single_support_time','double_support_time','step_length_symmetry',
                'midswing_dorsiflexion_angle','midswing_ankle_heigh_dif'}

# %% Gait segmentation and kinematic analysis.
trials_info = get_data_info(trial_indexes=[i for i in range(0,92)])
# trials_info = get_data_info(trial_indexes=trials_to_run)

trials_info_problems = get_data_info_problems()
trials_info_alignment = get_data_alignment()
# trials_select_previous_cycle = get_data_select_previous_cycle()
trials_case = get_data_case()

for trial in trials_info:
    if trial != 0:
        continue
    # Get trial info.
    session_id = trials_info[trial]['sid']
    trial_name = trials_info[trial]['trial']
    print('Processing session {}_{} - trial {}...'.format(trial, session_id, trial_name))
    sessionDir = os.path.join(dataFolder, "{}_{}".format(trial, session_id))
    pathKinematicsFolder = os.path.join(sessionDir, 'OpenSimData', 'Kinematics')
    pathDynamicsFolder = os.path.join(sessionDir, 'OpenSimData', 'Dynamics')

    if trial in trials_info_alignment:        
        trialName_aligned = trial_name + '_' + suffixOutputFileName
    else:
        trialName_aligned = trial_name

    # initialize feature list
    features = {}
    
    # Load data if not existing 
    if not os.path.exists(os.path.join(pathKinematicsFolder,trialName_aligned + '.mot')):
        trial_id = get_trial_id(session_id,trial_name)    
        download_trial(trial_id,sessionDir,session_id=session_id)
    
    for leg in legs:
        skip_kinetics = False
        # If kinetics exist, get the gait cycle from there
        
        if trial in trials_info_problems.keys():
            if leg in trials_info_problems[trial]['leg']:
                if 'justKinematics' in trials_info_problems[trial] and leg in trials_info_problems[trial]['justKinematics']:
                    skip_kinetics=True
                else:
                    print('Skipping leg {} for trial {} because it is a trial with problems.'.format(leg, trial))
                    continue
            
        ## create gait kinematics class   
        gait_kinematics = gait_analysis(sessionDir, trial_name, leg=leg,
                     lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
                     n_gait_cycles=-1)
        
        case = '2' # default
        if trial in trials_case.keys():
            if leg in trials_case[trial].keys():
                case = trials_case[trial][leg]
                print('Using case {} for trial {} leg {}.'.format(case, trial, leg))
        case_leg = '{}_{}'.format(case, leg)
        
        # load kinetics if they exist
        if not skip_kinetics:
            try:
                # Retrieve results from the optimal solution using utilsKineticsOpenSimAD.
                opt_sol_obj = kineticsOpenSimAD(sessionDir, trialName_aligned, case_leg)
                kinetics_exist = True
                print('Kinetics exist.')
            except:
                kinetics_exist = False
                print('Kinetics do not exist. Using kinematics only.')
        else:
            kinetics_exist = False
        
        if kinetics_exist:
            # pathOutputJsonFile = os.path.join(pathKinematicsFolder, 'gaitResults_{}.json'.format(leg))
            pathOutputJsonFile = os.path.join(pathKinematicsFolder, '{}_kinematic_features_{}.json'.format(trial_name, leg))
            with open(pathOutputJsonFile) as json_file:
                gaitResults = json.load(json_file)
    
            # Extract time window to later trim simulations.
            time_window = [
                gaitResults['events']['ipsilateralTime'][0],
                gaitResults['events']['ipsilateralTime'][-1]]

            # Find gait cycle closest to what was used for dynamics 
            timeOptions = gait_kinematics.gaitEvents['ipsilateralTime'][:,0]
            gaitCycle = np.argmin(np.abs([timeOptions-time_window[0]]))
        else:
            gaitCycle = 0
    
        # Will get kinematic features even if kinetics wasn't run
        # compute scalars
        kinematic_features = gait_kinematics.compute_scalars(scalar_names)        
    
        ## Features that aren't automated in the gait-analysis class
        
        # peak knee flexion angle swing
        to_1_idx = gait_kinematics.gaitEvents['ipsilateralIdx'][:,1]
        hs_2_idx = gait_kinematics.gaitEvents['ipsilateralIdx'][:,2]
        
        peakKFAs,units = gait_kinematics.compute_peak_angle('knee_angle', start_idx = to_1_idx,
                                                 end_idx=hs_2_idx, return_all=True)
        peakKFA = np.mean(peakKFAs[gaitCycle])
        
        kinematic_features['peak_knee_flexion_angle_swing'] = {}
        kinematic_features['peak_knee_flexion_angle_swing']['value'] = peakKFA
        kinematic_features['peak_knee_flexion_angle_swing']['units'] = units
            
        
        # Peak knee flexion during stance
        hs_1_idx = gait_kinematics.gaitEvents['ipsilateralIdx'][:,0]
        midstance_idx = gait_kinematics.gaitEvents['ipsilateralIdx'][:,0] + np.squeeze(np.round(
            .5*np.diff(gait_kinematics.gaitEvents['ipsilateralIdx'][:,(0,2)]))).astype(int)
        
        peakKFAs,units = gait_kinematics.compute_peak_angle('knee_angle', start_idx = hs_1_idx,
                                           end_idx=midstance_idx, return_all=True)
        
        # Average across all strides.
        peakKFA = np.mean(peakKFAs[gaitCycle])

        kinematic_features['peak_knee_flexion_angle_stance'] = {}
        kinematic_features['peak_knee_flexion_angle_stance']['value'] = peakKFA
        kinematic_features['peak_knee_flexion_angle_stance']['units'] = units
        
        
        # shoulder flexion rom
        roms,units = gait_kinematics.compute_rom('arm_flex', 
                        start_idx = gait_kinematics.gaitEvents['ipsilateralIdx'][:,0],
                        end_idx=gait_kinematics.gaitEvents['ipsilateralIdx'][:,2], 
                        return_all=True)
        
        # Average across all strides.
        rom = np.mean(roms[gaitCycle])
        kinematic_features['arm_flexion_rom'] = {}
        kinematic_features['arm_flexion_rom']['value'] = rom
        kinematic_features['arm_flexion_rom']['units'] = units
        
        
        # correlations
        dofs = ['arm_flex','elbow_flex']
        correlations, meanCorrelation = gait_kinematics.compute_correlations(
                                        cols_to_compare=dofs,visualize=False,
                                        return_all=True)    
        kinematic_features['arm_swing_symmetry'] = {}
        kinematic_features['arm_swing_symmetry']['value'] = meanCorrelation[gaitCycle,0]
        kinematic_features['arm_swing_symmetry']['units'] = 'unitless'
        
        features_leg = {}
        features_leg.update(kinematic_features)
        
        # if kinetics wasn't run, skip
        if not kinetics_exist:
            continue
        
        # Kinetic results

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
        
        # difference in vertical ankle positions during midswing
        # make marker names lowercase
        # index by time just to be safe
        idx_midswing_markers = np.argmin(np.abs(opt_sol_obj.marker_dict['time']-
                                 np.mean([to_1_time,hs_2_time])))
        opt_sol_obj.marker_dict['markers'] = {m.lower():d for m,d in opt_sol_obj.marker_dict['markers'].items()}
        if leg == 'r': cont_leg = 'l' 
        else: cont_leg = 'r'
        dAnkle = (opt_sol_obj.marker_dict['markers'][leg + '_ankle_study'] - 
                 opt_sol_obj.marker_dict['markers'][cont_leg + '_ankle_study'])
        kinetic_features['ankle_vertical_pos_diff_midswing'] = {}
        kinetic_features['ankle_vertical_pos_diff_midswing']['value'] = dAnkle[idx_midswing_markers,1]
        kinetic_features['ankle_vertical_pos_diff_midswing']['units'] = 'm'
               
        # Peak moments and impulses.
        peak_moment_stance, peak_moment_swing = {}, {}
        peak_power_stance_pos_moment, peak_power_stance_neg_moment = {}, {}
        peak_power_swing_pos_moment, peak_power_swing_neg_moment = {}, {}
        work_stance, work_swing = {}, {}
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
            impulse_stance[coordinate] = []

            moment_stance_negative = moment_stance[moment_stance < 0]
            if len(moment_stance_negative) > 0:
                impulse_stance[coordinate].append(np.abs(np.mean(moment_stance_negative)) * dt * np.sum(moment_stance_negative))
            else:
                impulse_stance[coordinate].append(0)
            
            moment_stance_positive = moment_stance[moment_stance > 0]
            if len(moment_stance_positive) > 0:
                impulse_stance[coordinate].append(np.abs(np.mean(moment_stance_positive)) * dt * np.sum(moment_stance_positive))
            else:
                impulse_stance[coordinate].append(0)

            kinetic_features['impulse_stance_positive_' + coordinate] = {}
            kinetic_features['impulse_stance_positive_' + coordinate]['value'] = impulse_stance[coordinate][1]
            kinetic_features['impulse_stance_positive_' + coordinate]['units'] = 'Nm*s'
            kinetic_features['impulse_stance_negative_' + coordinate] = {}
            kinetic_features['impulse_stance_negative_' + coordinate]['value'] = impulse_stance[coordinate][0]
            kinetic_features['impulse_stance_negative_' + coordinate]['units'] = 'Nm*s'
            
            # Powers.
            power_stance_pos_moment = (joint_powers_filt_sel[coordinate + '_' + leg].to_numpy()[hs_1_idx:to_1_idx] *
                                       (joint_moments_filt_sel[coordinate + '_' + leg].to_numpy()[hs_1_idx:to_1_idx]>0).astype(float))
            power_stance_neg_moment = (joint_powers_filt_sel[coordinate + '_' + leg].to_numpy()[hs_1_idx:to_1_idx] *
                                       (joint_moments_filt_sel[coordinate + '_' + leg].to_numpy()[hs_1_idx:to_1_idx]<0).astype(float))
            power_swing_pos_moment = (joint_powers_filt_sel[coordinate + '_' + leg].to_numpy()[to_1_idx:hs_2_idx] *
                                       (joint_moments_filt_sel[coordinate + '_' + leg].to_numpy()[to_1_idx:hs_2_idx]>0).astype(float))
            power_swing_neg_moment = (joint_powers_filt_sel[coordinate + '_' + leg].to_numpy()[to_1_idx:hs_2_idx] *
                                       (joint_moments_filt_sel[coordinate + '_' + leg].to_numpy()[to_1_idx:hs_2_idx]<0).astype(float))

            # Peak powers.
            peak_power_stance_pos_moment[coordinate] = [np.min(power_stance_pos_moment), 
                                                        np.max(power_stance_pos_moment)]
            kinetic_features['peak_pos_power_stance_pos_moment_' + coordinate] = {}
            kinetic_features['peak_pos_power_stance_pos_moment_' + coordinate]['value'] = peak_power_stance_pos_moment[coordinate][1]
            kinetic_features['peak_pos_power_stance_pos_moment_' + coordinate]['units'] = 'W'
            kinetic_features['peak_neg_power_stance_pos_moment_' + coordinate] = {}
            kinetic_features['peak_neg_power_stance_pos_moment_' + coordinate]['value'] = peak_power_stance_pos_moment[coordinate][0]
            kinetic_features['peak_neg_power_stance_pos_moment_' + coordinate]['units'] = 'W'
            
            peak_power_stance_neg_moment[coordinate] = [np.min(power_stance_neg_moment), 
                                                        np.max(power_stance_neg_moment)]
            kinetic_features['peak_pos_power_stance_neg_moment_' + coordinate] = {}
            kinetic_features['peak_pos_power_stance_neg_moment_' + coordinate]['value'] = peak_power_stance_neg_moment[coordinate][1]
            kinetic_features['peak_pos_power_stance_neg_moment_' + coordinate]['units'] = 'W'
            kinetic_features['peak_neg_power_stance_neg_moment_' + coordinate] = {}
            kinetic_features['peak_neg_power_stance_neg_moment_' + coordinate]['value'] = peak_power_stance_neg_moment[coordinate][0]
            kinetic_features['peak_neg_power_stance_neg_moment_' + coordinate]['units'] = 'W'

            peak_power_swing_pos_moment[coordinate] = [np.min(power_swing_pos_moment), 
                                                        np.max(power_swing_pos_moment)]
            kinetic_features['peak_pos_power_swing_pos_moment_' + coordinate] = {}
            kinetic_features['peak_pos_power_swing_pos_moment_' + coordinate]['value'] = peak_power_swing_pos_moment[coordinate][1]
            kinetic_features['peak_pos_power_swing_pos_moment_' + coordinate]['units'] = 'W'
            kinetic_features['peak_neg_power_swing_pos_moment_' + coordinate] = {}
            kinetic_features['peak_neg_power_swing_pos_moment_' + coordinate]['value'] = peak_power_swing_pos_moment[coordinate][0]
            kinetic_features['peak_neg_power_swing_pos_moment_' + coordinate]['units'] = 'W'
            
            peak_power_swing_neg_moment[coordinate] = [np.min(power_swing_neg_moment), 
                                                        np.max(power_swing_neg_moment)]
            kinetic_features['peak_pos_power_swing_neg_moment_' + coordinate] = {}
            kinetic_features['peak_pos_power_swing_neg_moment_' + coordinate]['value'] = peak_power_swing_neg_moment[coordinate][1]
            kinetic_features['peak_pos_power_swing_neg_moment_' + coordinate]['units'] = 'W'
            kinetic_features['peak_neg_power_swing_neg_moment_' + coordinate] = {}
            kinetic_features['peak_neg_power_swing_neg_moment_' + coordinate]['value'] = peak_power_swing_neg_moment[coordinate][0]
            kinetic_features['peak_neg_power_swing_neg_moment_' + coordinate]['units'] = 'W'
            
            # Work
            pos_power_pos_moment = np.maximum(power_stance_pos_moment,0)
            neg_power_pos_moment = np.minimum(power_stance_pos_moment,0)
            pos_power_neg_moment = np.maximum(power_stance_neg_moment,0)
            neg_power_neg_moment = np.minimum(power_stance_neg_moment,0)
            
            # Positive work when the moment is either pos or neg
            kinetic_features['positive_work_stance_positive_moment_' + coordinate] = {}
            kinetic_features['positive_work_stance_positive_moment_' + coordinate]['value'] = trapz(pos_power_pos_moment,time_stance)
            kinetic_features['positive_work_stance_positive_moment_' + coordinate]['units'] = 'J'
            kinetic_features['positive_work_stance_negative_moment' + coordinate] = {}
            kinetic_features['positive_work_stance_negative_moment' + coordinate]['value'] = trapz(pos_power_neg_moment,time_stance)
            kinetic_features['positive_work_stance_negative_moment' + coordinate]['units'] = 'J'
            
            # Negative work when the moment is either pos or neg
            kinetic_features['negative_work_stance_positive_moment_' + coordinate] = {}
            kinetic_features['negative_work_stance_positive_moment_' + coordinate]['value'] = trapz(neg_power_pos_moment,time_stance)
            kinetic_features['negative_work_stance_positive_moment_' + coordinate]['units'] = 'J'
            kinetic_features['negative_work_stance_negative_moment' + coordinate] = {}
            kinetic_features['negative_work_stance_negative_moment' + coordinate]['value'] = trapz(neg_power_neg_moment,time_stance)
            kinetic_features['negative_work_stance_negative_moment' + coordinate]['units'] = 'J'

            
        peak_moment_gaitCycle, impulse_gaitCycle = {}, {}

        for coordinate in coordinates_gaitCycle:
            # Moments.
            moment_gaitCycle = joint_moments_filt_sel[coordinate + '_' + leg].to_numpy()[hs_1_idx:hs_2_idx]
            # plt.plot(moment_gaitCycle,label=leg)

            # Peak moments.
            peak_moment_gaitCycle[coordinate] = [np.min(moment_gaitCycle),np.max(moment_gaitCycle)]
            kinetic_features['peak_moment_gaitCycle_positive_' + coordinate] = {}
            kinetic_features['peak_moment_gaitCycle_positive_' + coordinate]['value'] = peak_moment_gaitCycle[coordinate][1]
            kinetic_features['peak_moment_gaitCycle_positive_' + coordinate]['units'] = 'Nm'
            kinetic_features['peak_moment_gaitCycle_negative_' + coordinate] = {}
            kinetic_features['peak_moment_gaitCycle_negative_' + coordinate]['value'] = peak_moment_gaitCycle[coordinate][0]
            kinetic_features['peak_moment_gaitCycle_negative_' + coordinate]['units'] = 'Nm'
            
            # Impulses.
            time_gaitCycle = joint_moments_filt_sel['time'].to_numpy()[hs_1_idx:hs_2_idx]
            dt = np.diff(time_stance)[0]
            impulse_gaitCycle[coordinate] = []

            moment_gaitCycle_negative = moment_gaitCycle[moment_gaitCycle < 0]
            if len(moment_gaitCycle_negative) > 0:
                impulse_gaitCycle[coordinate].append(np.abs(np.mean(moment_gaitCycle_negative)) * dt * np.sum(moment_gaitCycle_negative))
            else:
                impulse_gaitCycle[coordinate].append(0)
            
            moment_gaitCycle_positive = moment_gaitCycle[moment_gaitCycle > 0]
            if len(moment_gaitCycle_positive) > 0:
                impulse_gaitCycle[coordinate].append(np.abs(np.mean(moment_gaitCycle_positive)) * dt * np.sum(moment_gaitCycle_positive))
            else:
                impulse_gaitCycle[coordinate].append(0)

            kinetic_features['impulse_gaitCycle_positive_' + coordinate] = {}
            kinetic_features['impulse_gaitCycle_positive_' + coordinate]['value'] = impulse_gaitCycle[coordinate][1]
            kinetic_features['impulse_gaitCycle_positive_' + coordinate]['units'] = 'Nm*s'
            kinetic_features['impulse_gaitCycle_negative_' + coordinate] = {}
            kinetic_features['impulse_gaitCycle_negative_' + coordinate]['value'] = impulse_gaitCycle[coordinate][0]
            kinetic_features['impulse_gaitCycle_negative_' + coordinate]['units'] = 'Nm*s'
        
        features_leg.update(gaitResults['scalars'])
        features.update({key + '_' + leg: value for key, value in features_leg.items()})

    # Save features in json file.
    pathFeaturesFolder = os.path.join(sessionDir, 'OpenSimData', 'Features')
    os.makedirs(pathFeaturesFolder, exist_ok=True)
    pathOutputJsonFile = os.path.join(pathFeaturesFolder, '{}_features.json'.format(trial_name))
    with open(pathOutputJsonFile, 'w') as outfile:
        json.dump(features, outfile)
