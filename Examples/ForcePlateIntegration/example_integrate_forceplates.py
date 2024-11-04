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
sys.path.append("./../../")
script_folder,_ = os.path.split(os.path.abspath(__file__))
                
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import opensim
import requests

import utils as ut
from utilsProcessing import lowPassFilter
from utilsPlotting import plot_dataframe
import utilsKinematics

# %% User-defined variables.

# OpenCap session information from url.
# View example at https://app.opencap.ai/session/9eea5bf0-a550-4fa5-bc69-f5f072765848
session_id = '9eea5bf0-a550-4fa5-bc69-f5f072765848'
trial_name = 'jump2'
# Specify where to download the kinematic data. Make sure there are no spaces
# in this path.
data_folder = os.path.abspath(os.path.join(script_folder,'Data', session_id))

# Path and filename for force data. Should be a *.mot file of forces applied
# to right and left foot.
force_dir = os.path.join(data_folder,'MeasuredForces',trial_name)
force_path = os.path.join(force_dir,'jump2_forces_filt30Hz.mot')

# Download force data from gdrive: this is only for this example. This section
# can be deleted if your force_path points to a local file.
os.makedirs(force_dir,exist_ok=True)
force_gdrive_url = 'https://drive.google.com/uc?id=1Uppzkskn7OiJ5GncNT2sl35deX_U3FTN&export=download'
response = requests.get(force_gdrive_url)
with open(force_path, 'wb') as f:
    f.write(response.content)      

# Lowpass filter ground forces and kinematics 
lowpass_filter_frequency = 30
filter_force_data = True
filter_kinematics_data = True

## Transform from force reference frame to OpenCap reference frame.
# We will use 4 reference frames G, C, R, and L:
# C (checkerboard) and G (ground) are aligned and axes are defined by checkerboard. 
# You should always have a black square in the top left corner, and the board 
# should be on its side.
# x out of board, y up, z away from checkers (left if looking at board). 
# Origin (C0) is the top left black-to-black corner; 
# Origin (G0) defined as opencap as the lowest point in the trial. There is a 
# vertical offset value in the opencap data that defines r_C0_to_G0
# Both kinematic and force data will be output in G.
# R and L are the frames for the force plate data with origins R0 and L0.

## Position from checkerboard origin to force plate origin expressed in C.
# You will need to measure this and position the checkerboard consistently for every
# collection. 

r_C0_to_forceOrigin_exp_C = {'R': [0,-.191,.083],
                             'L': [0,-.191,.083]}


## Rotation from force plates to checkerboard
# You can represent rotations however you like, just make it a 
# scipy.spatial.transform.Rotation object. 
# docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

R_forcePlates_to_C = {'R':R.from_euler('y',-90,degrees=True),
                      'L':R.from_euler('y',-90,degrees=True)}

# Flags
visualize_synchronization = False # visualize kinematic/force synchronization
run_ID = True

# %% Functions

def get_columns(list1,list2):
    inds = [i for i, item in enumerate(list2) if item in list1]
    return inds           

# %% Load and transform force data
# We will transform the force data into the OpenCap reference frame. We will
# then add a vertical offset to the COP, because the OpenCap data has a vertical
# offset from the checkerboard origin.

# Download kinematics data if folder doesn't exist. If incomplete data in folder,
# delete the folder and re-run.
if not os.path.exists(os.path.join(data_folder,'MarkerData')):
    _,model_name = ut.download_kinematics(session_id, folder=data_folder)
else:
    model_name,_ = os.path.splitext(ut.get_model_name_from_metadata(data_folder))

# Initiate kinematic analysis
kinematics = utilsKinematics.kinematics(data_folder, trial_name, 
                                        modelName=model_name, 
                                        lowpass_cutoff_frequency_for_coordinate_values=10)

# Get mass and vertical offset from opencap outputs
opencap_settings = ut.get_main_settings(data_folder,trial_name)
opencap_metadata = ut.import_metadata(os.path.join(data_folder,'sessionMetadata.yaml'))
mass = opencap_metadata['mass_kg']
if 'verticalOffset' in opencap_settings.keys():
    vertical_offset = opencap_settings['verticalOffset']
else:
    vertical_offset = 0

forces_structure = ut.storage_to_numpy(force_path)
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
    for leg in ['R','L']:
        force_columns= get_columns([leg + '_' + q + d for d in directions],force_headers)
        rot = R_forcePlates_to_C[leg]
        # we want r_expFP * R_FP_to_C, but rot.apply does R*r, so need to invert
        force_data[:,force_columns] = rot.inv().apply(force_data[:,force_columns])                                      
       
## Transform COP from force plates to G
r_G0_to_C0_expC = np.array((0,-vertical_offset,0))

for leg in ['R','L']:
    force_columns = get_columns([leg + '_ground_force_p' + d for d in directions],force_headers)
    r_forceOrigin_to_COP_exp_C = force_data[:,force_columns]
    r_G0_to_COP_exp_G = ( r_G0_to_C0_expC + 
                          r_C0_to_forceOrigin_exp_C[leg] + 
                          r_forceOrigin_to_COP_exp_C )
    force_data[:,force_columns] = r_G0_to_COP_exp_G

## Time synchronize
# Here we will use cross correlation of the summed vertical GRFs vs. COM acceleration
center_of_mass_acc = kinematics.get_center_of_mass_accelerations(lowpass_cutoff_frequency=4)
force_columns = get_columns([leg + '_ground_force_vy' for leg in ['R','L']],force_headers)
forces_for_cross_corr = np.sum(force_data[:,force_columns],axis=1,keepdims=True)

framerate_forces = 1/np.diff(force_data[:2,0])[0]
framerate_kinematics = 1/np.diff(kinematics.time[:2])[0]
time_forces_downsamp, forces_for_cross_corr_downsamp = ut.downsample(forces_for_cross_corr,
                                                                     force_data[:,0],
                                                                     framerate_forces,
                                                                     framerate_kinematics)
forces_for_cross_corr_downsamp = lowPassFilter(time_forces_downsamp,
                                               forces_for_cross_corr_downsamp,
                                               4, order=4)

# get body mass from metadata
mass = opencap_metadata['mass_kg']

# zero pad the shorter signal
dif_lengths = len(forces_for_cross_corr_downsamp) - len(center_of_mass_acc['y'])
if dif_lengths > 0:
    com_signal = np.pad(center_of_mass_acc['y']*mass + mass*9.8, 
                        (int(np.floor(dif_lengths / 2)), 
                        int(np.ceil(dif_lengths / 2))), 
                        'constant',constant_values=0)[:,np.newaxis]
    kinematics_pad_length = int(np.floor(dif_lengths / 2))
    force_signal = forces_for_cross_corr_downsamp
else:
    force_signal = np.pad(forces_for_cross_corr_downsamp, 
                          (int(np.floor(np.abs(dif_lengths) / 2)), 
                          int(np.ceil(np.abs(dif_lengths) / 2))), 
                          'constant', constant_values=0)
    kinematics_pad_length = 0
    com_signal = center_of_mass_acc['y'][:,np.newaxis]*mass + mass*9.8

# compute the lag between GRFs and forces
_,lag = ut.cross_corr(np.squeeze(com_signal),np.squeeze(force_signal), 
                      visualize=visualize_synchronization)

force_data_new = np.copy(force_data)
force_data_new[:,0] = force_data[:,0] - (-lag+kinematics_pad_length)/framerate_kinematics

# Plot vertical force and (COM acceleration*m +mg)
if visualize_synchronization:
    plt.figure()
    plt.plot(kinematics.time,center_of_mass_acc['y']*mass + mass*9.8,label='COM acceleration')
    plt.plot(force_data_new[:,0],forces_for_cross_corr, label = 'vGRF')
    plt.legend()

# Force data directories
force_folder = os.path.join(data_folder,'MeasuredForces',trial_name)
os.makedirs(force_folder,exist_ok=True)

# Save force data
_,force_file_name = os.path.split(force_path)
root,ext = os.path.splitext(force_file_name)
force_output_path = os.path.join(force_folder,trial_name + '_syncd_forces.mot')
ut.numpy_to_storage(force_headers, force_data_new, force_output_path, datatype=None)

# Identify time range for ID
time_range = {}
time_range['start'] = np.max([force_data_new[0,0], kinematics.time[0]])
time_range['end'] = np.min([force_data_new[-1,0], kinematics.time[-1]])

# %% Run Inverse Dynamics

# Make folders
opensim_folder = os.path.join(data_folder,'OpenSimData')

id_folder = os.path.join(opensim_folder,'InverseDynamics',trial_name)
os.makedirs(id_folder,exist_ok=True)

model_path = os.path.join(opensim_folder,'Model',model_name + '.osim')
ik_path = os.path.join(opensim_folder,'Kinematics', trial_name + '.mot')
el_path = os.path.join(id_folder,'Setup_ExternalLoads.xml')
id_path = os.path.join(id_folder,'Setup_ID.xml')

# Generic setups
id_path_generic = os.path.join(script_folder,'ID_setup','Setup_ID.xml')
el_path_generic = os.path.join(script_folder,'ID_setup','Setup_ExternalLoads.xml')

if run_ID:
    # External loads
    opensim.Logger.setLevelString('error')
    ELTool = opensim.ExternalLoads(el_path_generic, True)
    ELTool.setDataFileName(force_output_path)
    ELTool.setName(trial_name)
    ELTool.printToXML(el_path)
    
    # ID    
    IDTool = opensim.InverseDynamicsTool(id_path_generic)
    IDTool.setModelFileName(model_path)
    IDTool.setName(trial_name)
    IDTool.setStartTime(time_range['start'])
    IDTool.setEndTime(time_range['end'])      
    IDTool.setExternalLoadsFileName(el_path)
    IDTool.setCoordinatesFileName(ik_path)
    if not filter_kinematics_data:
        freq = -1
    else:
        freq = lowpass_filter_frequency
    IDTool.setLowpassCutoffFrequency(freq)
    IDTool.setResultsDir(id_folder)
    IDTool.setOutputGenForceFileName(trial_name + '.sto')   
    IDTool.printToXML(id_path)   
    print('Running inverse dynamics.')
    IDTool.run()

# %% Load and plot kinematics, forces, inverse dynamics results

# Create force and ID dataframesID results
id_output_path = os.path.join(id_folder,trial_name + '.sto')
id_dataframe = ut.load_storage(id_output_path,outputFormat='dataframe')

force_dataframe = pd.DataFrame(force_data_new, columns=force_headers)

# Select columns to plot
sagittal_dofs = ['ankle_angle','knee_angle','hip_flexion']
kinematics_columns_plot = [s + '_' + leg for s in sagittal_dofs for leg in ['r','l']]
moment_columns_plot = [s + '_moment' for s in kinematics_columns_plot]
force_columns_plot = [leg + '_ground_force_v' + dir for leg in ['R','L'] 
                                                  for dir in ['x','y','z']]

# Make plots
plt.close('all')
plot_dataframe([kinematics.get_coordinate_values()], xlabel='time', ylabel='angle [deg]',
               y = kinematics_columns_plot ,title='Kinematics')

plot_dataframe([id_dataframe], xlabel='time', ylabel='moment [Nm]',
               y = moment_columns_plot ,title='Moments')

plot_dataframe([force_dataframe], xlabel='time', ylabel='force [N]',
               y = force_columns_plot ,title='Ground Forces',xrange=list(time_range.values()))

