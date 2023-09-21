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

    This example shows how to integrate force data from forceplates for a jump.
    
    All data should be expressed in meters and Newtons.
    
    Input data:
    1) OpenCap session identifier and trial name
    2) Force data in a .mot file with 9 columns per leg: (see example data)
    (Fx, Fy, Fz, Tx, Ty, Tz, COPx, COPy, COPz). Column names should be 
    (R_ground_force_x,...R_ground_torque_x,...,R_ground_force_px,...
     L_ground_force_x,........)
    All data should be expressed in meters and Newtons. The 

    
'''

import os
import sys
sys.path.append("../")

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import opensim

from utils import storage_to_numpy, numpy_to_storage, download_kinematics, \
                  cross_corr

from utilsProcessing import lowPassFilter
import utilsKinematics

# %% User-defined variables.

# OpenCap session information
session_id = '9eea5bf0-a550-4fa5-bc69-f5f072765848'
trial_name = 'jump2'
# Specify where to download the data.
data_folder = os.path.join("./../Data", session_id)

# Path and filename for force data. Should be a *.mot file of forces applied
# to right and left foot.
force_dir = os.path.abspath(os.path.join(os.getcwd(),'ExampleData'))
force_file_name = 'jump2_forces_filt50Hz.mot'

# Lowpass filter ground forces and kinematics 
lowpass_filter_frequency = 50
filter_force_data = True
filter_kinematics_data = True

## Transform from force reference frame to OpenCap reference frame.
# We will use 3 reference frames C, R, and L:
# C is OpenCap and is defined by checkerboard. You should always have a black 
# square in the top left corner, and the board should be on its side. 
# The origin (C0) is the top left black-to-black corner; x right, y down, z in. 
# R and L are the frames for the force plate data.

## Position from checkerboard origin to force plate origin expressed in checkerboard.
# You will need to measure this and position the checkerboard consistently. 

# Right force plate
r_R0_from_C0_exp_C0 = [.191, .083, 0]

# Left force plate (same in example data)
r_L0_from_C0_exp_C0 = np.copy(r_R0_from_C0_exp_C0)

## Rotation matrix from force plates to checkerboard

# R from R to C
R_R_C = np.array(((1,0,0),(0,0,-1),(0,-1,0)))

# R from L to C (same in example data)
R_L_C = np.copy(R_R_C)

# %% Functions

def get_columns(list1,list2):
    inds = [i for i, item in enumerate(list2) if item in list1]
    return inds

def downsample(data,time,framerate_in,framerate_out):
    # Calculate the downsampling factor
    downsampling_factor = framerate_in / framerate_out
    
    # Create new indices for downsampling
    original_indices = np.arange(len(data))
    new_indices = np.arange(0, len(data), downsampling_factor)
    
    # Perform downsampling with interpolation
    downsampled_data = np.ndarray((len(new_indices), data.shape[1]))
    for i in range(data.shape[1]):
        downsampled_data[:,i] = np.interp(new_indices, original_indices, data[:,i])
    
    downsampled_time = np.interp(new_indices, original_indices, time)
    
    return downsampled_time, downsampled_data

# %% Load and transform force data
# We will transform the force data into the OpenCap reference frame. We will
# then add a vertical offset to the COP, because the OpenCap data has a vertical
# offset from the checkerboard origin.

# Download kinematics data, initiate kinematic analysis
_,modelName = download_kinematics(session_id, folder=data_folder)
kinematics = utilsKinematics.kinematics(data_folder, trial_name, modelName=modelName, lowpass_cutoff_frequency_for_coordinate_values=10)

# Load force data
forces_structure = storage_to_numpy(os.path.join(force_dir,force_file_name))
force_data = forces_structure.view(np.float64).reshape(forces_structure.shape + (-1,))
force_headers = forces_structure.dtype.names

# %%

# Filter force data
# Note - it is not great to filter COP data directly. In the example GRF data
# we filtered raw forces and moments before computing COP.
if filter_force_data:
    force_data[:,1:] = lowPassFilter(force_data[:,0], force_data[:,1:],
                                 lowpass_filter_frequency, order=4)

# Rotate the forces into C
quantity = ['ground_force_v','ground_torque_','ground_force_p']
directions = ['x','y','z']
for q in quantity:
    for leg,rotMat in zip(['R', 'L'],[R_R_C, R_L_C]):
        force_columns= get_columns([leg + '_' + q + d for d in directions],force_headers)
        # rot = R.from_matrix(rotMat)
        # TODO HARDCODE
        rot = R.from_euler('y',270,degrees=True)
        force_data[:,force_columns] = rot.apply(force_data[:,force_columns])                                      

# 

# Transform the COP into C. Add the translation from force origin to C0, and 
# subtract the offset that was applied to OpenCap data.

#TODO get this from OpenCap data
offset = 0
BW = 86
offset_vector = np.array((0,offset,0)).T

for leg,translation in zip(['R','L'],[r_R0_from_C0_exp_C0, r_L0_from_C0_exp_C0]):
    force_columns = get_columns([leg + '_ground_force_p' + d for d in directions],force_headers)
    force_data[:,force_columns] = force_data[:,force_columns] - offset_vector + translation


## Time synchronize
# Here we will use cross correlation of the summed vertical GRFs vs. COM acceleration
center_of_mass_acc = kinematics.get_center_of_mass_accelerations(lowpass_cutoff_frequency=4)
force_columns = get_columns([leg + '_ground_force_vy' for leg in ['R','L']],force_headers)
forces_for_cross_corr = np.sum(force_data[:,force_columns],axis=1,keepdims=True)

framerate_forces = 1/np.diff(force_data[:2,0])[0]
framerate_kinematics = 1/np.diff(kinematics.time[:2])[0]
time_forces_downsamp, forces_for_cross_corr_downsamp = downsample(forces_for_cross_corr,force_data[:,0],
                                                         framerate_forces,framerate_kinematics)
forces_for_cross_corr_downsamp = lowPassFilter(time_forces_downsamp,
                                               forces_for_cross_corr_downsamp,
                                               4, order=4)

# zero pad the shorter signal
dif_lengths = len(forces_for_cross_corr_downsamp) - len(center_of_mass_acc['y'])
if dif_lengths > 0:
    com_signal = np.pad(center_of_mass_acc['y']*BW + BW*9.8, (int(np.floor(dif_lengths / 2)), 
                                                  int(np.ceil(dif_lengths / 2))), 'constant',constant_values=0)[:,np.newaxis]
    kinematics_pad_length = int(np.floor(dif_lengths / 2))
    force_signal = forces_for_cross_corr_downsamp
else:
    force_signal = np.pad(forces_for_cross_corr_downsamp, (int(np.floor(np.abs(dif_lengths) / 2)), 
                          int(np.ceil(np.abs(dif_lengths) / 2))), 'constant',
                          constant_values=0)
    kinematics_pad_length = 0
    com_signal = center_of_mass_acc['y'][:,np.newaxis]*BW + BW*9.8

# compute the lag between GRFs and forces
_,lag = cross_corr(np.squeeze(com_signal),np.squeeze(force_signal), visualize=True)

force_data_new = np.copy(force_data)
force_data_new[:,0] = force_data[:,0] - (-lag+kinematics_pad_length)/framerate_kinematics

# Plot vertical force and (COM acceleration*m +mg)
plt.figure()
plt.plot(kinematics.time,center_of_mass_acc['y']*BW + BW*9.8,label='COM acceleration')
plt.plot(force_data_new[:,0],forces_for_cross_corr, label = 'vGRF')
plt.legend()

# %% Download data.


# Save force data
root,ext = os.path.splitext(force_file_name)
force_output_path = os.path.join(force_dir,root + '_rotated' + ext)
numpy_to_storage(force_headers, force_data_new, force_output_path, datatype=None)

# %% Run Inverse Dynamics

# 


# %% Load and plot joint moments

test = 1


