'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_squat_analysis.py
    ---------------------------------------------------------------------------
    Copyright 2024 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Carmichael Ong
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
                
    Please contact us for any questions: https://www.opencap.ai/#contact

    This example shows how to run a kinematic analysis of squat data.
    
'''

import os
import sys
sys.path.append("../")
sys.path.append("../ActivityAnalyses")
import numpy as np

from squat_analysis import squat_analysis
from utils import get_trial_id, download_trial
from utilsPlotting import plot_dataframe_with_shading

# %% Paths.
baseDir = os.path.join(os.getcwd(), '..')
dataFolder = os.path.join(baseDir, 'Data')

# %% User-defined variables.

session_id = '1b380cb9-3c5c-498c-bc30-a1a379ffa04c'
trial_name = 'bossu_squats'

# TODO:
# Peak trunk lean, peak hip adduction, peak knee abduction, knee flexion range of motion, peak ankle eversion
# scalar_names = {'peak_knee_flexion'}

# Select how many repetitions you'd like to analyze. Select -1 for all
# repetitions detected in the trial.
n_repetitions = -1

# Select lowpass filter frequency for kinematics data.
filter_frequency = 4

# %% Gait analysis.
# Get trial id from name.
trial_id = get_trial_id(session_id,trial_name)    

# Set session path.
sessionDir = os.path.join(dataFolder, session_id)

# Download data.
trialName = download_trial(trial_id,sessionDir,session_id=session_id) 

# Init gait analysis.
squat = squat_analysis(
    sessionDir, trialName,
    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    n_repetitions=n_repetitions)
squat_events = squat.get_squat_events()

# Example metrics
ratio_max_knee_flexion_angle_mean, ratio_max_knee_flexion_angle_std, ratio_max_knee_flexion_angle_unit = squat.compute_ratio_peak_angle('knee_angle_r', 'knee_angle_l')
ratio_max_hip_flexion_angle_mean, ratio_max_hip_flexion_angle_std, ratio_max_knee_flexion_angle_unit = squat.compute_ratio_peak_angle('hip_flexion_r', 'hip_flexion_l')

max_knee_flexion_angle_r_mean, max_knee_flexion_angle_r_std, _ = squat.compute_peak_angle('knee_angle_r')
max_knee_flexion_angle_l_mean, max_knee_flexion_angle_l_std, _ = squat.compute_peak_angle('knee_angle_l')
max_knee_flexion_angle_mean_mean = np.mean(np.array([max_knee_flexion_angle_r_mean, max_knee_flexion_angle_l_mean]))
max_knee_flexion_angle_mean_std = np.mean(np.array([max_knee_flexion_angle_r_std, max_knee_flexion_angle_l_std]))

max_hip_flexion_angle_r_mean, max_hip_flexion_angle_r_std, _ = squat.compute_peak_angle('hip_flexion_r')
max_hip_flexion_angle_l_mean, max_hip_flexion_angle_l_std, _ = squat.compute_peak_angle('hip_flexion_l')
max_hip_flexion_angle_mean_mean = np.mean(np.array([max_hip_flexion_angle_r_mean, max_hip_flexion_angle_l_mean]))
max_hip_flexion_angle_mean_std = np.mean(np.array([max_hip_flexion_angle_r_std, max_hip_flexion_angle_l_std]))

max_hip_adduction_angle_r_mean, max_hip_adduction_angle_r_std, _ = squat.compute_peak_angle('hip_adduction_r')
max_hip_adduction_angle_l_mean, max_hip_adduction_angle_l_std, _ = squat.compute_peak_angle('hip_adduction_l')
max_hip_adduction_angle_mean_mean = np.mean(np.array([max_hip_adduction_angle_r_mean, max_hip_adduction_angle_l_mean]))
max_hip_adduction_angle_mean_std = np.mean(np.array([max_hip_adduction_angle_r_std, max_hip_adduction_angle_l_std]))

rom_knee_flexion_angle_r_mean, rom_knee_flexion_angle_r_std, _ = squat.compute_range_of_motion('knee_angle_r')
rom_knee_flexion_angle_l_mean, rom_knee_flexion_angle_l_std, _ = squat.compute_range_of_motion('knee_angle_l')
rom_knee_flexion_angle_mean_mean = np.mean(np.array([rom_knee_flexion_angle_r_mean, rom_knee_flexion_angle_l_mean]))
rom_knee_flexion_angle_mean_std = np.mean(np.array([rom_knee_flexion_angle_r_std, rom_knee_flexion_angle_l_std]))

print('Peak knee flexion angle: {} +/- {} deg'.format(np.round(max_knee_flexion_angle_mean_mean,2), np.round(max_knee_flexion_angle_mean_std,2)))
print('Peak hip flexion angle: {} +/- {} deg'.format(np.round(max_hip_flexion_angle_mean_mean,2), np.round(max_hip_flexion_angle_mean_std,2)))
print('Peak hip adduction angle: {} +/- {} deg'.format(np.round(max_hip_adduction_angle_mean_mean,2), np.round(max_hip_adduction_angle_mean_std,2)))
print('ROM knee flexion angle: {} +/- {} deg'.format(np.round(rom_knee_flexion_angle_mean_mean,2), np.round(rom_knee_flexion_angle_mean_std,2)))

    
# %% Example plots.
squat_joint_kinematics = squat.get_coordinates_segmented_normalized_time()
squat_com_kinematics = squat.get_center_of_mass_segmented_normalized_time() 

plot_dataframe_with_shading(
    [squat_joint_kinematics['mean']],
    [squat_joint_kinematics['sd']],
    xlabel = '% repetition',
    title = 'Joint kinematics (m or deg)')

plot_dataframe_with_shading(
            [squat_com_kinematics['mean']],
            [squat_com_kinematics['sd']],
            xlabel = '% repetition',
            title = 'center of mass kinematics (m)')