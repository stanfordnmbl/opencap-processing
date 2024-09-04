# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:31:58 2023

@author: hpl
"""

# Center mot file on pelvis_tx, tz = 0
import sys
sys.path.append('../')

import os
import utils
import numpy as np
import matplotlib.pyplot as plt
from utilsProcessing import lowPassFilter

# baseDir = 'C:/SharedGdrive/sparseIK/Data/'

dofs_to_zero = ['pelvis_tx','pelvis_tz','pelvis_ty','time']

rewriteMotFiles = True

# timeLags = [
#             [.13,.90],
#             [.50,-1],
#             [.50,-1],
#             [.13,1],
#             [.53,-1],
#             [.53,-1],
#             ]

# secondHalf = [
#             'OpenCapSubject4_ts/OpenSimData/Kinematics/walkingTS2_aligned.mot',
#             'subject4_reference/OpenSimData/Video/HRNet/2-cameras/IK/walkingTS2.mot',
#             'subject4_reference/OpenSimData/Mocap/IK/walkingTS2.mot',
#             'OpenCapSubject4/OpenSimData/Kinematics/walking4_aligned.mot',
#             'subject4_reference/OpenSimData/Video/HRNet/2-cameras/IK/walking4.mot',
#             'subject4_reference/OpenSimData/Mocap/IK/walking4.mot'
#             ]


# # # # Walking
# baseDir = 'C:/Users/hpl/Documents/MyRepositories/wham/output/walking4/case0/walking4_trimmed/OpenSim/' # walking

# timeLags = [# walking
#             [.667,-1], # video walking
#             [0,-1], # 2 cam walking
#             [0,-1], # mocap walking,
#             ]


# secondHalf = [ # walk
#             'IK\walking4_trimmed_5\walking4_trimmed_5.mot',
#             'Reference/OpenSimData/Video/HRNet/2-cameras/IK/walking4.mot',
#             'Reference/OpenSimData/Mocap/IK/walking4.mot',
#             ]

# secondHalf_forces = [ # squat+
#             'Dynamics/walking4/GRF_resultant_walking4_0.mot',
#             'Reference/OpenSimData/Video/HRNet/2-cameras/Dynamics/walking4/forces_resultant.mot',
#             'Reference/ForceData/walking4_forces.mot',
#             # 'IK\squats1_5_wham_result\squats1_5_wham_result.mot'
#             ]

# trim_forces = False

# # # SQuats
baseDir = 'C:/Users/hpl/Documents/MyRepositories/wham/output/squats1/OpenSim/' # squats

timeLags = [ # squat
            [.547,-1], # video squat
            [0,-1], # 2 cam walking,
            [0,-1], # mocap walking
            ]

secondHalf = [ # squat
            'IK\squats1_5\squats1_5.mot',
            # 'IK\squats1_5_wham_result\squats1_5_wham_result.mot',
            'Reference\Video_2cam\IK\squats1.mot',
            'Reference\Mocap\IK\squats1.mot',
            # 'IK\squats1_5_wham_result\squats1_5_wham_result.mot'
            ]

secondHalf_forces = [ # squat+
            'Dynamics/squats1_5_rep1/GRF_resultant_squats1_5_0.mot',
            'Reference/Video_2cam/Dynamics/squats1_1/forces_resultant.mot',
            'Reference/Forces/squats1_forces.mot',
            # 'IK\squats1_5_wham_result\squats1_5_wham_result.mot'
            ]

trim_forces=False

# TODO this is just a filler
# secondHalf_forces = secondHalf


# secondHalf_forces = [
#             'OpenCapSubject4_ts/OpenSimData/Dynamics/walkingTS2_aligned/GRF_resultant_walkingTS2_aligned_101_l.mot',
#             'subject4_reference/OpenSimData/Video/HRNet/2-cameras/Dynamics/walkingTS2/forces_resultant.mot',
#             'subject4_reference/ForceData/walkingTS2_forces.mot',
#             'OpenCapSubject4/OpenSimData/Dynamics/walking4_aligned/GRF_resultant_walking4_aligned_101_l.mot',
#             'subject4_reference/OpenSimData/Video/HRNet/2-cameras/Dynamics/walking4/forces_resultant.mot',
#             'subject4_reference/ForceData/walking4_forces.mot'
#             ]

motFiles = [os.path.join(baseDir,sh) for sh in secondHalf]
forceFiles = [os.path.join(baseDir,sh) for sh in secondHalf_forces]

orange = np.divide([196, 118, 41], 255)
purple = np.divide([92, 45, 141], 255)
orange = np.clip(orange * 1.2, 0, 1)
purple = np.clip(purple * 1.4, 0, 1)

if rewriteMotFiles:
    for motFile,forceFile,timeLag in zip(motFiles,forceFiles,timeLags):
        print('Processing ' + motFile)
        data,header = utils.load_storage(motFile,outputFormat='numpy')
        data_f, header_f = utils.load_storage(forceFile,outputFormat='numpy')
    
        # kinematics
        ind_start = np.argmin(np.abs(data[:,header.index('time')]-timeLag[0]))
        if timeLag[1] != -1:
            ind_end = np.argmin(np.abs(data[:,header.index('time')]-timeLag[1]))
        else:
            ind_end = -1
        data = data[ind_start:ind_end,:] # Cut off the first timeLag seconds
    
        for dof in dofs_to_zero:
            data[:,header.index(dof)] = data[:,header.index(dof)] - data[:,header.index(dof)][0]
            
        # TEMP
        # data[:,header.index('pelvis_ty')] = data[:,header.index('pelvis_ty')] + .87
    
        mot_out = motFile.replace('.mot','_compareToMonocular.mot')
        utils.numpy_to_storage(header, data, mot_out, datatype=None)
    
        # forces
        if trim_forces:
            ind_start= np.argmin(np.abs(data_f[:,header_f.index('time')]-timeLag[0]))
        else:
            ind_start = 0
            
        if timeLag[1] != -1 and trim_forces:
            ind_end = np.argmin(np.abs(data_f[:,header_f.index('time')]-timeLag[1]))
        else:
            ind_end = -1
        data_f = data_f[ind_start:ind_end,:] # Cut off the first timeLag seconds
        
        #start time at 0
        dof='time'
        data_f[:,header_f.index(dof)] = data_f[:,header_f.index(dof)] - data_f[:,header_f.index(dof)][0]
        
        force_out = forceFile.replace('.mot','_compareToMonocular.mot')
        utils.numpy_to_storage(header_f, data_f, force_out, datatype=None)

# Load and trim
walking = {}
walking_TS = {}

trialTypes = ['mono','opencap','mocap']
# Put header_f, forces, header_k, kinematics in a dictionary
for trialType in trialTypes:
    walking[trialType] = {}
    walking_TS[trialType] = {}

for i, (motFile,forceFile) in enumerate(zip(motFiles,forceFiles)):
    trialType = trialTypes[int(np.remainder(i,3))]
    # if i < 3:
    #     walking_TS[trialType]['kinematics'],walking_TS[trialType]['header_k'] = utils.load_storage(motFile.replace('.mot','_compareToMonocular.mot'))
    #     walking_TS[trialType]['forces'],walking_TS[trialType]['header_f'] = utils.load_storage(forceFile.replace('.mot','_compareToMonocular.mot'))
    # else:
    walking[trialType]['kinematics'],walking[trialType]['header_k'] = utils.load_storage(motFile.replace('.mot','_compareToMonocular.mot'))
    walking[trialType]['forces'],walking[trialType]['header_f'] = utils.load_storage(forceFile.replace('.mot','_compareToMonocular.mot'))


def interpolate_columns(new_values, old_values, matrix):
    # Ensure that the sizes match
    if len(old_values) != matrix.shape[0]:
        raise ValueError("Mismatched sizes of old_values, new_values, or matrix dimensions")

    # Interpolate along columns using vectorized operations
    interpolated_matrix = np.zeros((len(new_values),matrix.shape[1]))

    for col in range(matrix.shape[1]):
        interpolated_matrix[:, col] = np.interp(new_values, old_values, matrix[:, col])

    return interpolated_matrix

# Interpolate all the data to match mocap time vector
for trialType in trialTypes:
    mocap_time = walking['mocap']['kinematics'][:,walking['mocap']['header_k'].index('time')]
    walking[trialType]['kinematics'] = interpolate_columns(mocap_time,walking[trialType]['kinematics'][:,walking[trialType]['header_k'].index('time')],walking[trialType]['kinematics'])
    walking[trialType]['forces'] = interpolate_columns(mocap_time,walking[trialType]['forces'][:,walking[trialType]['header_f'].index('time')],walking[trialType]['forces'])
    
    # mocap_time_TS = walking_TS['mocap']['kinematics'][:,walking_TS['mocap']['header_k'].index('time')]
    # walking_TS[trialType]['kinematics'] = interpolate_columns(mocap_time_TS,walking_TS[trialType]['kinematics'][:,walking_TS[trialType]['header_k'].index('time')],walking_TS[trialType]['kinematics'])
    # walking_TS[trialType]['forces'] = interpolate_columns(mocap_time_TS,walking_TS[trialType]['forces'][:,walking_TS[trialType]['header_f'].index('time')],walking_TS[trialType]['forces'])


# compute MAE
dofs_rot = ['knee_angle_r','knee_angle_l','hip_flexion_r','hip_flexion_l','ankle_angle_r','ankle_angle_l','pelvis_rotation','pelvis_list','pelvis_tilt',
        'subtalar_angle_r','subtalar_angle_l','lumbar_extension','lumbar_bending','lumbar_rotation','arm_flex_r','arm_flex_l','arm_add_r','arm_add_l','elbow_flex_r','elbow_flex_l',
        'hip_adduction_r','hip_adduction_l','hip_rotation_r','hip_rotation_l']
dofs_trans = ['pelvis_tx','pelvis_ty','pelvis_tz']

# plot GRFs using header names in headerName_f dict
headerName_f = {'mocap':{'lx':'L_ground_force_vx','ly':'L_ground_force_vy','lz':'L_ground_force_vz','rx':'R_ground_force_vx','ry':'R_ground_force_vy','rz':'R_ground_force_vz'},
                'opencap':{'lx':'ground_force_l_vx','ly':'ground_force_l_vy','lz':'ground_force_l_vz','rx':'ground_force_r_vx','ry':'ground_force_r_vy','rz':'ground_force_r_vz'},
                'mono':{'lx':'ground_force_left_vx','ly':'ground_force_left_vy','lz':'ground_force_left_vz','rx':'ground_force_right_vx','ry':'ground_force_right_vy','rz':'ground_force_right_vz'}}

# Compute mean absolute error of kinematics using dofs_rot in header, comparing 'mono' to 'mocap', and 'opencap' to 'mocap'
for trialType in trialTypes:
    walking[trialType]['MAE_rot'] = {}
    walking_TS[trialType]['MAE_rot'] = {}
    walking[trialType]['MAE_trans'] = {}
    walking_TS[trialType]['MAE_trans'] = {}
    walking[trialType]['MAE_grf'] = {}
    for dof in dofs_rot:
        dofInd = walking[trialType]['header_k'].index(dof)
        walking[trialType]['MAE_rot'][dof] = np.mean(np.abs(walking[trialType]['kinematics'][:,dofInd] - walking['mocap']['kinematics'][:,dofInd]))
        # walking_TS[trialType]['MAE_rot'][dof] = np.mean(np.abs(walking_TS[trialType]['kinematics'][:,dofInd] - walking_TS['mocap']['kinematics'][:,dofInd]))
    for dof in dofs_trans:
        dofInd = walking[trialType]['header_k'].index(dof)
        walking[trialType]['MAE_trans'][dof] = np.mean(np.abs(walking[trialType]['kinematics'][:,dofInd] - walking['mocap']['kinematics'][:,dofInd]))
        # walking_TS[trialType]['MAE_trans'][dof] = np.mean(np.abs(walking_TS[trialType]['kinematics'][:,dofInd] - walking_TS['mocap']['kinematics'][:,dofInd]))
    for k,f in headerName_f[trialType].items():     
        time = walking[trialType]['forces'][:,walking[trialType]['header_f'].index('time')]
        walking[trialType]['MAE_grf'][f] = np.mean(np.abs(lowPassFilter(time, walking[trialType]['forces'][:,walking[trialType]['header_f'].index(f)],6) - 
                                                          lowPassFilter(time, walking['mocap']['forces'][:,walking['mocap']['header_f'].index(headerName_f['mocap'][k])],6)))


    # average the rot and trans MAEs for each trialType for walking and walking_TS
    walking[trialType]['MAE_rot_avg'] = np.mean([walking[trialType]['MAE_rot'][dof] for dof in dofs_rot])
    # walking_TS[trialType]['MAE_rot_avg'] = np.mean([walking_TS[trialType]['MAE_rot'][dof] for dof in dofs_rot])
    walking[trialType]['MAE_trans_avg'] = np.mean([walking[trialType]['MAE_trans'][dof] for dof in dofs_trans])
    # walking_TS[trialType]['MAE_trans_avg'] = np.mean([walking_TS[trialType]['MAE_trans'][dof] for dof in dofs_trans])
    walking[trialType]['MAE_grf_avg'] = np.mean([walking[trialType]['MAE_grf'][dof] for dof in headerName_f[trialType].values()])





# plot GRFs using header names in headerName_f dict
# 3x1 subplot
plt.figure(1,figsize=(15,5))
directions = ['rx','ry','rz']
colors = [purple,orange,'k']
for i,trialType in enumerate(trialTypes):
   
    for j,direction in enumerate(directions):
        plt.subplot(1,3,j+1)
        time = walking[trialType]['forces'][:,walking[trialType]['header_f'].index('time')]
        plt.plot(time,lowPassFilter(time, walking[trialType]['forces'][:,walking[trialType]['header_f'].index(headerName_f[trialType][direction])], 6),color=colors[i],
                 linewidth =3)
        # plt.plot(time,walking_TS[trialType]['forces'][:,walking_TS[trialType]['header_f'].index(headerName_f[trialType][direction])],color=purple)
        plt.ylabel('GRF_' + direction + ' (N)',fontsize=16)
        plt.xlabel('Time (s)',fontsize=16)

# add legend to first subplot with no box and in upper left corner
plt.subplot(1,3,1)
plt.legend(['OpenCap1','OpenCap2','Measured'],loc='upper left')
plt.gca().get_legend().set_frame_on(False)



# remove right and top spines, increase axis label sizes and xlabel and ylabel sizes
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=14)

# tight
plt.tight_layout()

    


