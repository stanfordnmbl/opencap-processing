# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:21:40 2022

@author: antoi
"""

def get_default_setup(motion_type):

    setups = {}    
    setups['running'] = {
        'weights': {
            'positionTrackingTerm': 100,
            'velocityTrackingTerm': 10,
            'accelerationTrackingTerm': 50,
            'activationTerm': 10,
            'armExcitationTerm': 0.001,
            'lumbarExcitationTerm': 0.001,
            'jointAccelerationTerm': 0.001,
            'activationDtTerm': 0.001,
            'forceDtTerm': 0.001},            
        'coordinates_toTrack': {
            'pelvis_tilt': {"weight": 10},
            'pelvis_list': {"weight": 10},
            'pelvis_rotation': {"weight": 10},
            'pelvis_tx': {"weight": 10},
            'pelvis_ty': {"weight": 10},
            'pelvis_tz': {"weight": 10}, 
            'hip_flexion_l': {"weight": 20},
            'hip_adduction_l': {"weight": 10},
            'hip_rotation_l': {"weight": 1},
            'hip_flexion_r': {"weight": 20},
            'hip_adduction_r': {"weight": 10},
            'hip_rotation_r': {"weight": 1},
            'knee_angle_l': {"weight": 10},
            'knee_angle_r': {"weight": 10},
            'ankle_angle_l': {"weight": 10},
            'ankle_angle_r': {"weight": 10},
            'subtalar_angle_l': {"weight": 10},
            'subtalar_angle_r': {"weight": 10},
            'lumbar_extension': {"weight": 10},
            'lumbar_bending': {"weight": 10},
            'lumbar_rotation': {"weight": 10},
            'arm_flex_l': {"weight": 10},
            'arm_add_l': {"weight": 10},
            'arm_rot_l': {"weight": 10},
            'arm_flex_r': {"weight": 10},
            'arm_add_r': {"weight": 10},
            'arm_rot_r': {"weight": 10},
            'elbow_flex_l': {"weight": 10},
            'elbow_flex_r': {"weight": 10},
            'pro_sup_l': {"weight": 10},
            'pro_sup_r': {"weight": 10}},
        'coordinate_constraints': {
            'pelvis_tx': {"env_bound": 0.1}},
        'ignorePassiveFiberForce': True}
    
    setups['walking'] = {
        'weights': {
            'positionTrackingTerm': 10,
            'velocityTrackingTerm': 10,
            'accelerationTrackingTerm': 50,
            'activationTerm': 1,
            'armExcitationTerm': 0.001,
            'lumbarExcitationTerm': 0.001,
            'jointAccelerationTerm': 0.001,
            'activationDtTerm': 0.001,
            'forceDtTerm': 0.001},            
        'coordinates_toTrack': {
            'pelvis_tilt': {"weight": 10},
            'pelvis_list': {"weight": 1},
            'pelvis_rotation': {"weight": 1},
            'pelvis_tx': {"weight": 1},
            'pelvis_ty': {"weight": 1},
            'pelvis_tz': {"weight": 1}, 
            'hip_flexion_l': {"weight": 10},
            'hip_adduction_l': {"weight": 1},
            'hip_rotation_l': {"weight": 1},
            'hip_flexion_r': {"weight": 10},
            'hip_adduction_r': {"weight": 1},
            'hip_rotation_r': {"weight": 1},
            'knee_angle_l': {"weight": 10},
            'knee_angle_r': {"weight": 10},
            'ankle_angle_l': {"weight": 10},
            'ankle_angle_r': {"weight": 10},
            'subtalar_angle_l': {"weight": 1},
            'subtalar_angle_r': {"weight": 1},
            'lumbar_extension': {"weight": 10},
            'lumbar_bending': {"weight": 1},
            'lumbar_rotation': {"weight": 1},
            'arm_flex_l': {"weight": 1},
            'arm_add_l': {"weight": 1},
            'arm_rot_l': {"weight": 1},
            'arm_flex_r': {"weight": 1},
            'arm_add_r': {"weight": 1},
            'arm_rot_r': {"weight": 1},
            'elbow_flex_l': {"weight": 1},
            'elbow_flex_r': {"weight": 1},
            'pro_sup_l': {"weight": 1},
            'pro_sup_r': {"weight": 1}},            
        'coordinate_constraints': {
            'pelvis_ty': {"env_bound": 0.1},
            'pelvis_tx': {"env_bound": 0.1}},
        'enableLimitTorques': True}
    
    
    setups['drop_jump'] = {
        'weights': {
            'positionTrackingTerm': 50,
            'velocityTrackingTerm': 10,
            'accelerationTrackingTerm': 50,
            'activationTerm': 1,
            'armExcitationTerm': 0.001,
            'lumbarExcitationTerm': 0.001,
            'jointAccelerationTerm': 0.001,
            'activationDtTerm': 0.001,
            'forceDtTerm': 0.001},            
        'coordinates_toTrack': {
            'pelvis_tilt': {"weight": 10},
            'pelvis_list': {"weight": 1},
            'pelvis_rotation': {"weight": 1},
            'pelvis_tx': {"weight": 1},
            'pelvis_ty': {"weight": 10},
            'pelvis_tz': {"weight": 1}, 
            'hip_flexion_l': {"weight": 10},
            'hip_adduction_l': {"weight": 1},
            'hip_rotation_l': {"weight": 1},
            'hip_flexion_r': {"weight": 10},
            'hip_adduction_r': {"weight": 1},
            'hip_rotation_r': {"weight": 1},
            'knee_angle_l': {"weight": 10},
            'knee_angle_r': {"weight": 10},
            'ankle_angle_l': {"weight": 10},
            'ankle_angle_r': {"weight": 10},
            'subtalar_angle_l': {"weight": 1},
            'subtalar_angle_r': {"weight": 1},
            'lumbar_extension': {"weight": 10},
            'lumbar_bending': {"weight": 1},
            'lumbar_rotation': {"weight": 1},
            'arm_flex_l': {"weight": 50},
            'arm_add_l': {"weight": 50},
            'arm_rot_l': {"weight": 50},
            'arm_flex_r': {"weight": 50},
            'arm_add_r': {"weight": 50},
            'arm_rot_r': {"weight": 50},
            'elbow_flex_l': {"weight": 50},
            'elbow_flex_r': {"weight": 50},
            'pro_sup_l': {"weight": 50},
            'pro_sup_r': {"weight": 50}},            
        'coordinate_constraints': {
            'pelvis_ty': {"env_bound": 0.02},
            'pelvis_tx': {"env_bound": 0.02},
            'pelvis_tz': {"env_bound": 0.02}},
        'ignorePassiveFiberForce': True}
    
    # setups['sit_to_stand'] = {
    #     'weights': {
    #         'positionTrackingTerm': 50,
    #         'velocityTrackingTerm': 10,
    #         'accelerationTrackingTerm': 50,
    #         'activationTerm': 100,
    #         'armExcitationTerm': 0.001,
    #         'lumbarExcitationTerm': 0.001,
    #         'jointAccelerationTerm': 0.001,
    #         'activationDtTerm': 0.001,
    #         'forceDtTerm': 0.001,
    #         'reserveActuatorTerm': 0.001,
    #         # 'vGRFRatioTerm': 0.1
    #         },            
    #     'coordinates_toTrack': {
    #         'pelvis_tilt': {"weight": 100},
    #         'pelvis_list': {"weight": 10},
    #         'pelvis_rotation': {"weight": 1},
    #         'pelvis_tx': {"weight": 100},
    #         'pelvis_ty': {"weight": 10},
    #         'pelvis_tz': {"weight": 100}, 
    #         'hip_flexion_l': {"weight": 100},
    #         'hip_adduction_l': {"weight": 20},
    #         'hip_rotation_l': {"weight": 1},
    #         'hip_flexion_r': {"weight": 100},
    #         'hip_adduction_r': {"weight": 20},
    #         'hip_rotation_r': {"weight": 1},
    #         'knee_angle_l': {"weight": 100},
    #         'knee_angle_r': {"weight": 100},
    #         'ankle_angle_l': {"weight": 100},
    #         'ankle_angle_r': {"weight": 100},
    #         'subtalar_angle_l': {"weight": 20},
    #         'subtalar_angle_r': {"weight": 20},
    #         'lumbar_extension': {"weight": 100},
    #         'lumbar_bending': {"weight": 20},
    #         'lumbar_rotation': {"weight": 20},
    #         'arm_flex_l': {"weight": 50},
    #         'arm_add_l': {"weight": 10},
    #         'arm_rot_l': {"weight": 10},
    #         'arm_flex_r': {"weight": 50},
    #         'arm_add_r': {"weight": 10},
    #         'arm_rot_r': {"weight": 10},
    #         'elbow_flex_l': {"weight": 10},
    #         'elbow_flex_r': {"weight": 10},
    #         'pro_sup_l': {"weight": 10},
    #         'pro_sup_r': {"weight": 10}},            
    #     'coordinate_constraints': {
    #         'pelvis_ty': {"env_bound": 0.1},
    #         'pelvis_tx': {"env_bound": 0.1}},       
    #     'withReserveActuators': True,
    #     'reserveActuatorJoints': {
    #         'hip_rotation_l': 30, 'hip_rotation_r': 30},
    #     'ignorePassiveFiberForce': True}
    
    setups['sit_to_stand'] = {
        'weights': {
            'positionTrackingTerm': 50,
            'velocityTrackingTerm': 10,
            'accelerationTrackingTerm': 50,
            'activationTerm': 100,
            'armExcitationTerm': 0.001,
            'lumbarExcitationTerm': 0.001,
            'jointAccelerationTerm': 0.001,
            'activationDtTerm': 0.001,
            'forceDtTerm': 0.001,
            'reserveActuatorTerm': 0.001,
            # 'vGRFRatioTerm': 0.1
            },            
        'coordinates_toTrack': {
            'pelvis_tilt': {"weight": 100},
            'pelvis_list': {"weight": 10},
            'pelvis_rotation': {"weight": 1},
            'pelvis_tx': {"weight": 100},
            'pelvis_ty': {"weight": 10},
            'pelvis_tz': {"weight": 100}, 
            'hip_flexion_l': {"weight": 100},
            'hip_adduction_l': {"weight": 20},
            'hip_rotation_l': {"weight": 1},
            'hip_flexion_r': {"weight": 100},
            'hip_adduction_r': {"weight": 20},
            'hip_rotation_r': {"weight": 1},
            'knee_angle_l': {"weight": 100},
            'knee_angle_r': {"weight": 100},
            'ankle_angle_l': {"weight": 100},
            'ankle_angle_r': {"weight": 100},
            'subtalar_angle_l': {"weight": 20},
            'subtalar_angle_r': {"weight": 20},
            'lumbar_extension': {"weight": 100},
            'lumbar_bending': {"weight": 20},
            'lumbar_rotation': {"weight": 20},
            'arm_flex_l': {"weight": 50},
            'arm_add_l': {"weight": 10},
            'arm_rot_l': {"weight": 10},
            'arm_flex_r': {"weight": 50},
            'arm_add_r': {"weight": 10},
            'arm_rot_r': {"weight": 10},
            'elbow_flex_l': {"weight": 10},
            'elbow_flex_r': {"weight": 10},
            'pro_sup_l': {"weight": 10},
            'pro_sup_r': {"weight": 10}},            
        'coordinate_constraints': {
            'pelvis_ty': {"env_bound": 0.1},
            'pelvis_tx': {"env_bound": 0.1}},       
        'withReserveActuators': True,
        'reserveActuatorJoints': {
            'hip_rotation_l': 30, 'hip_rotation_r': 30},
        'periodicConstraints': {'Qs': ['lowerLimbJoints']},
        'ignorePassiveFiberForce': True}
    
    setups['squats'] = {
        'weights': {
            'positionTrackingTerm': 50,
            'velocityTrackingTerm': 10,
            'accelerationTrackingTerm': 50,
            'activationTerm': 100,
            'armExcitationTerm': 0.001,
            'lumbarExcitationTerm': 0.001,
            'jointAccelerationTerm': 0.001,
            'activationDtTerm': 0.001,
            'forceDtTerm': 0.001,
            'reserveActuatorTerm': 0.001},            
        'coordinates_toTrack': {
            'pelvis_tilt': {"weight": 100},
            'pelvis_list': {"weight": 10},
            'pelvis_rotation': {"weight": 1},
            'pelvis_tx': {"weight": 100},
            'pelvis_ty': {"weight": 10},
            'pelvis_tz': {"weight": 100}, 
            'hip_flexion_l': {"weight": 100},
            'hip_adduction_l': {"weight": 20},
            'hip_rotation_l': {"weight": 1},
            'hip_flexion_r': {"weight": 100},
            'hip_adduction_r': {"weight": 20},
            'hip_rotation_r': {"weight": 1},
            'knee_angle_l': {"weight": 100},
            'knee_angle_r': {"weight": 100},
            'ankle_angle_l': {"weight": 100},
            'ankle_angle_r': {"weight": 100},
            'subtalar_angle_l': {"weight": 20},
            'subtalar_angle_r': {"weight": 20},
            'lumbar_extension': {"weight": 100},
            'lumbar_bending': {"weight": 20},
            'lumbar_rotation': {"weight": 20},
            'arm_flex_l': {"weight": 50},
            'arm_add_l': {"weight": 10},
            'arm_rot_l': {"weight": 10},
            'arm_flex_r': {"weight": 50},
            'arm_add_r': {"weight": 10},
            'arm_rot_r': {"weight": 10},
            'elbow_flex_l': {"weight": 10},
            'elbow_flex_r': {"weight": 10},
            'pro_sup_l': {"weight": 10},
            'pro_sup_r': {"weight": 10}},            
        'coordinate_constraints': {
            'pelvis_ty': {"env_bound": 0.1},
            'pelvis_tx': {"env_bound": 0.1}},
        'withReserveActuators': True,
        'reserveActuatorJoints': {
            'hip_rotation_l': 30, 'hip_rotation_r': 30},
        'periodicConstraints': {'Qs': ['lowerLimbJoints'],
                                'Qds': ['lowerLimbJoints'],
                                'muscles': ['all'],
                                'lumbar': ['all']},
        'ignorePassiveFiberForce': True}
        
    setups['jumping'] = {
        'weights': {
            'positionTrackingTerm': 100,
            'velocityTrackingTerm': 10,
            'accelerationTrackingTerm': 50,
            'activationTerm': 1,
            'armExcitationTerm': 0.001,
            'lumbarExcitationTerm': 0.001,
            'jointAccelerationTerm': 0.001,
            'activationDtTerm': 0.001,
            'forceDtTerm': 0.001},            
        'coordinates_toTrack': {
            'pelvis_tilt': {"weight": 10},
            'pelvis_list': {"weight": 10},
            'pelvis_rotation': {"weight": 10},
            'pelvis_tx': {"weight": 10},
            'pelvis_ty': {"weight": 100},
            'pelvis_tz': {"weight": 10}, 
            'hip_flexion_l': {"weight": 20},
            'hip_adduction_l': {"weight": 10},
            'hip_rotation_l': {"weight": 10},
            'hip_flexion_r': {"weight": 20},
            'hip_adduction_r': {"weight": 10},
            'hip_rotation_r': {"weight": 10},
            'knee_angle_l': {"weight": 10},
            'knee_angle_r': {"weight": 10},
            'ankle_angle_l': {"weight": 10},
            'ankle_angle_r': {"weight": 10},
            'subtalar_angle_l': {"weight": 10},
            'subtalar_angle_r': {"weight": 10},
            'lumbar_extension': {"weight": 10},
            'lumbar_bending': {"weight": 10},
            'lumbar_rotation': {"weight": 10},
            'arm_flex_l': {"weight": 100},
            'arm_add_l': {"weight": 100},
            'arm_rot_l': {"weight": 100},
            'arm_flex_r': {"weight": 100},
            'arm_add_r': {"weight": 100},
            'arm_rot_r': {"weight": 100},
            'elbow_flex_l': {"weight": 100},
            'elbow_flex_r': {"weight": 100},
            'pro_sup_l': {"weight": 100},
            'pro_sup_r': {"weight": 100}},
        'coordinate_constraints': {
            'pelvis_tx': {"env_bound": 0.1},
            'pelvis_ty': {"env_bound": 0.1}},
        'ignorePassiveFiberForce': True,
        }

    return setups[motion_type]

def get_trial_setup(settings, motion_type, trialName):
    
    if motion_type == 'running':        
        settings['trials'], settings['trials'][trialName] = {}, {}
        settings['trials'][trialName]['filter_coordinates_toTrack'] = True
        settings['trials'][trialName]['cutoff_freq_coord'] = 12
        settings['trials'][trialName]['filter_Qds_toTracks'] = True
        settings['trials'][trialName]['cutoff_freq_Qds'] = 12
        settings['trials'][trialName]['filter_Qdds_toTracks'] = True
        settings['trials'][trialName]['cutoff_freq_Qdds'] = 12
        settings['trials'][trialName]['splineQds'] = True
        settings['trials'][trialName]['meshDensity'] = 100
        settings['trials'][trialName]['yCalcnToes'] = True
        
    elif motion_type == 'jumping':  
        settings['trials'], settings['trials'][trialName] = {}, {}
        settings['trials'][trialName]['filter_coordinates_toTrack'] = True
        settings['trials'][trialName]['cutoff_freq_coord'] = 20
        settings['trials'][trialName]['filter_Qds_toTracks'] = True
        settings['trials'][trialName]['cutoff_freq_Qds'] = 20
        settings['trials'][trialName]['filter_Qdds_toTracks'] = True
        settings['trials'][trialName]['cutoff_freq_Qdds'] = 20
        settings['trials'][trialName]['splineQds'] = True
        settings['trials'][trialName]['meshDensity'] = 50
        settings['trials'][trialName]['yCalcnToes'] = True
        
    elif motion_type == 'walking':  
        settings['trials'], settings['trials'][trialName] = {}, {}
        settings['trials'][trialName]['filter_coordinates_toTrack'] = True
        settings['trials'][trialName]['cutoff_freq_coord'] = 6
        settings['trials'][trialName]['meshDensity'] = 100
        
    elif motion_type == 'drop_jump':  
        settings['trials'], settings['trials'][trialName] = {}, {}
        settings['trials'][trialName]['filter_coordinates_toTrack'] = True
        settings['trials'][trialName]['cutoff_freq_coord'] = 30
        settings['trials'][trialName]['filter_Qds_toTracks'] = True
        settings['trials'][trialName]['cutoff_freq_Qds'] = 30
        settings['trials'][trialName]['filter_Qdds_toTracks'] = True
        settings['trials'][trialName]['cutoff_freq_Qdds'] = 30
        settings['trials'][trialName]['splineQds'] = True
        settings['trials'][trialName]['meshDensity'] = 100
        
    elif motion_type == 'sit_to_stand':  
        settings['trials'], settings['trials'][trialName] = {}, {}
        settings['trials'][trialName]['filter_coordinates_toTrack'] = True
        settings['trials'][trialName]['cutoff_freq_coord'] = 4
        settings['trials'][trialName]['filter_Qds_toTracks'] = True
        settings['trials'][trialName]['cutoff_freq_Qds'] = 4
        settings['trials'][trialName]['filter_Qdds_toTracks'] = True
        settings['trials'][trialName]['cutoff_freq_Qdds'] = 4
        settings['trials'][trialName]['splineQds'] = True
        # settings['trials'][trialName]['isSTSs_yCalcn_vGRF'] = True
        settings['trials'][trialName]['yCalcnThresholds'] = 0.015
        # settings['trials'][trialName]['stsThresholds'] = 0
        settings['trials'][trialName]['meshDensity'] = 50
        
    elif motion_type == 'squats':  
        settings['trials'], settings['trials'][trialName] = {}, {}
        settings['trials'][trialName]['filter_coordinates_toTrack'] = True
        settings['trials'][trialName]['cutoff_freq_coord'] = 4
        settings['trials'][trialName]['filter_Qds_toTracks'] = True
        settings['trials'][trialName]['cutoff_freq_Qds'] = 4
        settings['trials'][trialName]['filter_Qdds_toTracks'] = True
        settings['trials'][trialName]['cutoff_freq_Qdds'] = 4
        settings['trials'][trialName]['splineQds'] = True
        settings['trials'][trialName]['isSquat'] = True
        settings['trials'][trialName]['squatThreshold'] = 5
        settings['trials'][trialName]['meshDensity'] = 50
        
    return settings
