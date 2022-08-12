'''
    This script formulates and solves the direct collocation problem underlying
    the tracking simulation of marker trajectories / coordinates.
'''

# TODO eventually remove margins in poly and bounds check

# %% Packages.
import os
import casadi as ca
import numpy as np
# import sys
# sys.path.append("../..") # utilities in child directory
# sys.path.append("../../opensimPipeline/JointReaction") # utilities in child directory
import yaml
import scipy.interpolate as interpolate

# %% Settings.
plotPolynomials = False
plotGuessVsBounds = False
tol = 4
d = 3
dimensions = ['x', 'y', 'z']
nContactSpheres = 6

def run_tracking(
        baseDir, dataDir, subject, settings, subject_demo, case='default', 
        data_type='Video', poseDetector='mmpose_0.8', cameraSetup='2-cameras',
        augmenter='separateLowerUpperBody_OpenPose', solveProblem=True, 
        analyzeResults=True, writeGUI=True, visualizeTracking=False, 
        visualizeResultsAgainstBounds=False, printModel=False, computeKAM=False,
        computeMCF=False, rep=None, processResults=False,
        collection_type='default', treadmill_speed=0):
    import copy
    
    # Cost function weights.
    weights = {
        'activationTerm': settings['weights']['activationTerm'],
        'jointAccelerationTerm': settings['weights']['jointAccelerationTerm'],
        'armExcitationTerm': settings['weights']['armExcitationTerm'],
        'activationDtTerm': settings['weights']['activationDtTerm'],
        'forceDtTerm': settings['weights']['forceDtTerm'],
        'positionTrackingTerm': settings['weights']['positionTrackingTerm'],
        'velocityTrackingTerm': settings['weights']['velocityTrackingTerm']}
    
    # Model info.
    OpenSimModel = 'LaiArnoldModified2017_poly_withArms_weldHand'
    if 'OpenSimModel' in settings:  
        OpenSimModel = settings['OpenSimModel']
    model_type = 'rajagopal2016'
    if 'model_type' in settings:  
        model_type = settings['model_type']
    model_full_name = OpenSimModel + "_scaled_adjusted"
    withMTP = True
    if 'withMTP' in settings:  
        withMTP = settings['withMTP']
    withActiveMTP = False
    if 'withActiveMTP' in settings:  
        withActiveMTP = settings['withActiveMTP']
        weights['mtpExcitationTerm'] = settings['weights']['w_mtpExcitationTerm']
    if withActiveMTP:
        raise ValueError("Not supported yet: withActiveMTP")   
    withArms = True  
    if "withArms" in settings:
         withArms = settings['withArms']
    withLumbarCoordinateActuators = True
    if "withLumbarCoordinateActuators" in settings:
         withLumbarCoordinateActuators = (
             settings['withLumbarCoordinateActuators'])
    if withLumbarCoordinateActuators:   
         weights['lumbarExcitationTerm'] = 1
         if 'lumbarExcitationTerm' in settings['weights']:
            weights['lumbarExcitationTerm'] = (
                settings['weights']['lumbarExcitationTerm'])
    withKA = False
    if 'withKA' in settings:  
        withKA = settings['withKA']
    scaleIsometricMuscleForce = 1
    if 'scaleIsometricMuscleForce' in settings: 
        scaleIsometricMuscleForce = settings['scaleIsometricMuscleForce']
    # dissipation = 2
    # if 'dissipation' in settings: 
    #     dissipation = settings['dissipation']
    # stiffness_spheres = 1 # actually means 1e6
    # if 'stiffness_spheres' in settings: 
    #     stiffness_spheres = settings['stiffness_spheres']
    # specificContactLocations = False
    # if 'specificContactLocations' in settings: 
    #     specificContactLocations = settings['specificContactLocations']
    # lower_transition_speed = False
    # if 'lower_transition_speed' in settings: 
    #     lower_transition_speed = settings['lower_transition_speed']
    reserveActuators = False
    if 'reserveActuators' in settings: 
        reserveActuators = settings['reserveActuators']
    if reserveActuators:
        reserveActuatorJoints = settings['reserveActuatorJoints']
        weights['reserveActuatorTerm'] = settings['weights']['reserveActuatorTerm']
    ignorePassiveFiberForce = False
    if 'ignorePassiveFiberForce' in settings: 
        ignorePassiveFiberForce = settings['ignorePassiveFiberForce']
        
    # Trials info.    
    trials = settings['trials']
    timeIntervals, timeElapsed, tgridf, N = {}, {}, {}, {}
    isSquats, squatThresholds, squatContacts = {}, {}, {}
    filter_coordinates_toTracks, cutoff_freq_coords, splineQds = {}, {}, {}
    filter_Qds_toTracks, cutoff_freq_Qds = {}, {}
    filter_Qdds_toTracks, cutoff_freq_Qdds = {}, {}
    isSTSs, stsThresholds, isSTSs_yCalcn, yCalcnThresholds = {}, {}, {}, {}
    isSTSs_yCalcn_vGRF, offset_vGRF_ratio = {}, {}
    yCalcnToes, yCalcnToesThresholds = {}, {}
    treadmill = {}
    for trial in trials:
        timeIntervals[trial] = trials[trial]['timeInterval']
        timeElapsed[trial] = timeIntervals[trial][1] - timeIntervals[trial][0]
        if 'N' in trials[trial]:
            N[trial] = trials[trial]['N']
        else:
            meshDensity = 100 # default is N=100 for t=1s
            if 'meshDensity' in trials[trial]:
                meshDensity = trials[trial]['meshDensity']
            N[trial] = int(round(timeElapsed[trial] * meshDensity, 2))
        tgrid = np.linspace(timeIntervals[trial][0], 
                            timeIntervals[trial][1], N[trial]+1)
        tgridf[trial] = np.zeros((1, N[trial]+1))
        tgridf[trial][:,:] = tgrid.T
        
        isSquats[trial] = False
        if 'isSquat' in trials[trial]:
            isSquats[trial] = trials[trial]['isSquat']
        if isSquats[trial]:
            squatThresholds[trial] = 5
            if 'squatThreshold' in trials[trial]:
                squatThresholds[trial] = trials[trial]['squatThreshold']
            squatContacts[trial] = 1 # heel sphere in contact
            if 'squatContacts' in trials[trial]:
                squatContacts[trial] = trials[trial]['squatContacts']
                
        isSTSs[trial] = False
        if 'isSTS' in trials[trial]:
            isSTSs[trial] = trials[trial]['isSTS']
        # if isSTSs[trial]:
        #     stsThresholds[trial] = 5
        #     if 'stsThreshold' in trials[trial]:
        #         stsThresholds[trial] = trials[trial]['stsThreshold']
                
        isSTSs_yCalcn[trial] = False
        if 'isSTSs_yCalcn' in trials[trial]:
            isSTSs_yCalcn[trial] = trials[trial]['isSTSs_yCalcn']           
                
        isSTSs_yCalcn_vGRF[trial] = False
        if 'isSTSs_yCalcn_vGRF' in trials[trial]:
            isSTSs_yCalcn_vGRF[trial] = trials[trial]['isSTSs_yCalcn_vGRF']
            
        if isSTSs_yCalcn[trial] or isSTSs_yCalcn_vGRF[trial]:
            yCalcnThresholds[trial] = 0.015
            if 'yCalcnThresholds' in trials[trial]:
                yCalcnThresholds[trial] = trials[trial]['yCalcnThresholds'] 
            weights['vGRFRatioTerm'] = 1
            if 'vGRFRatioTerm' in settings['weights']:
                weights['vGRFRatioTerm'] = settings['weights']['vGRFRatioTerm']
            offset_vGRF_ratio[trial] = 0
            if 'offset_vGRF_ratio' in trials[trial]:
                offset_vGRF_ratio[trial] = trials[trial]['offset_vGRF_ratio']
            stsThresholds[trial] = 0
            if 'stsThresholds' in trials[trial]:
                stsThresholds[trial] = trials[trial]['stsThresholds']
                
        yCalcnToes[trial] = False
        if 'yCalcnToes' in trials[trial]:
            yCalcnToes[trial] = trials[trial]['yCalcnToes']
            
        if yCalcnToes[trial] :
            yCalcnToesThresholds[trial] = 0.015
            if 'yCalcnToesThresholds' in trials[trial]:
                yCalcnToesThresholds[trial] = trials[trial]['yCalcnToesThresholds'] 
                
        filter_coordinates_toTracks[trial] = True
        if 'filter_coordinates_toTrack' in trials[trial]:
            filter_coordinates_toTracks[trial] = (
                trials[trial]['filter_coordinates_toTrack'])
        if filter_coordinates_toTracks[trial]:
            cutoff_freq_coords[trial] = 30
            if 'cutoff_freq_coord' in trials[trial]:
                cutoff_freq_coords[trial] = trials[trial]['cutoff_freq_coord']
                
        filter_Qds_toTracks[trial] = False
        if 'filter_Qds_toTracks' in trials[trial]:
            filter_Qds_toTracks[trial] = (
                trials[trial]['filter_Qds_toTracks'])
        if filter_Qds_toTracks[trial]:
            cutoff_freq_Qds[trial] = 30
            if 'cutoff_freq_Qds' in trials[trial]:
                cutoff_freq_Qds[trial] = trials[trial]['cutoff_freq_Qds']
                
        filter_Qdds_toTracks[trial] = False
        if 'filter_Qdds_toTracks' in trials[trial]:
            filter_Qdds_toTracks[trial] = (
                trials[trial]['filter_Qdds_toTracks'])
        if filter_Qdds_toTracks[trial]:
            cutoff_freq_Qdds[trial] = 30
            if 'cutoff_freq_Qdds' in trials[trial]:
                cutoff_freq_Qdds[trial] = trials[trial]['cutoff_freq_Qdds']
                
        splineQds[trial] = False
        if 'splineQds' in trials[trial]:
            splineQds[trial] = trials[trial]['splineQds']
            
        treadmill[trial] = False
        if treadmill_speed != 0:
            treadmill[trial] = True           
    
    # Filter info.          
    filter_bounds_guess_acceleration = False
    if 'filter_bounds_guess_acceleration' in settings:
        filter_bounds_guess_acceleration = (
            settings['filter_bounds_guess_acceleration'])
    
    # Guess info.
    type_guess = "dataDriven"
    if 'type_guess' in settings:
        type_guess = settings['type_guess']    
    guess_zeroAcceleration = False
    if 'guess_zeroAcceleration' in settings:
        guess_zeroAcceleration = settings['guess_zeroAcceleration']
    guess_zeroMTP = False
    if 'guess_zeroMTP' in settings:
        guess_zeroMTP = settings['guess_zeroMTP']
        
    # TODO: Bounds info
    bounds_update = True
    if 'bounds_update' in settings:
        bounds_update = settings['bounds_update']
    if bounds_update:
        bounds_update_index = 2
        if 'bounds_update_index' in settings:
            bounds_update_index = settings['bounds_update_index']
            
    lb_activation = 0.01
    if 'lb_activation' in settings:
        lb_activation = settings['lb_activation']
    
    # Problem info.    
    tracking_data = 'coordinates'
    if 'tracking_data' in settings:
        tracking_data = settings['tracking_data']
    if tracking_data == "coordinates":
        coordinates_toTrack = settings['coordinates_toTrack']
    offset_ty = True
    if 'offset_ty' in settings:
        offset_ty = settings['offset_ty']
    enableLimitTorques = False
    if 'enableLimitTorques' in settings:
        enableLimitTorques = settings['enableLimitTorques']    
    periodicConstraints = False
    if 'periodicConstraints' in settings:
        periodicConstraints = settings['periodicConstraints']
    trackQdds = True
    if 'trackQdds' in settings:
        trackQdds = settings['trackQdds']
    if trackQdds:
        weights['accelerationTrackingTerm'] = 1
        if 'accelerationTrackingTerm' in settings['weights']:
            weights['accelerationTrackingTerm'] = (
                settings['weights']['accelerationTrackingTerm'])    
    trackGRF = False
    if 'trackGRF' in settings:
        trackGRF = settings['trackGRF']
    if trackGRF:
        weights['grfTrackingTerm'] = 1
        if 'grfTrackingTerm' in settings['weights']:
            weights['grfTrackingTerm'] = settings['weights']['grfTrackingTerm']    
    trackGRM = False
    if 'trackGRM' in settings:
        trackGRM = settings['trackGRM']
    if trackGRM:
        weights['grmTrackingTerm'] = 1
        if 'grmTrackingTerm' in settings['weights']:
            weights['grmTrackingTerm'] = settings['weights']['grmTrackingTerm']
    powerDiff = 2
    if 'powerDiff' in settings:     
        powerDiff = settings['powerDiff']
    powActivations = 2
    if 'powActivations' in settings:
        powActivations = settings['powActivations']  
    volumeScaling = False
    if 'volumeScaling' in settings:
        volumeScaling = settings['volumeScaling']    
    coordinate_constraints = {}
    if 'coordinate_constraints' in settings:
        coordinate_constraints = settings['coordinate_constraints']
    optimizeContacts = False
    if 'optimizeContacts' in settings:     
        optimizeContacts = settings['optimizeContacts']
    if optimizeContacts:
        parameter_to_optimize = 'option1'
        if 'parameter_to_optimize' in settings:     
            parameter_to_optimize = settings['parameter_to_optimize']

    # %% Paths and dirs.
    # dataDir = copy.deepcopy(dataDir_main)
    # for s_dir in settings['pathData']: 
    #     dataDir = os.path.join(dataDir, s_dir)
    osDir = os.path.join(dataDir, subject, 'OpenSimData') 
    if data_type == 'Video':
        if collection_type == 'bigData':
            pathOSData = os.path.join(osDir, poseDetector, 
                                      cameraSetup, OpenSimModel)
        elif collection_type == 'default_OpenCap':
            pathOSData = os.path.join(osDir, poseDetector, cameraSetup)
        elif collection_type == 'default':
            pathOSData = osDir
            
        else:
            pathOSData = os.path.join(osDir, 'Video', poseDetector, 
                                      cameraSetup, augmenter)
    else:
        pathOSData = os.path.join(osDir, "Mocap")
    pathOSDataMocap = os.path.join(osDir, "Mocap")
    if collection_type == 'bigData' or collection_type == 'default_OpenCap' or collection_type == 'default':
        pathModelFolder = os.path.join(pathOSData, 'Model') 
    else:
        pathModelFolder = os.path.join(pathOSData, 'Model', OpenSimModel)
    pathModelFile = os.path.join(pathModelFolder,
                                 model_full_name + ".osim")
    nameMA = "default"
    if not withMTP:
        nameMA = "default_weldMTP"     
    pathSession = os.path.join(dataDir, subject)
    # pathMarkerData = os.path.join(pathSession, "MarkerData")
    # pathTRCFolder = os.path.join(pathMarkerData, 'PostAugmentation')
    pathExternalFunctionFolder = os.path.join(pathModelFolder,
                                              'ExternalFunction')
    if collection_type == 'bigData' or collection_type == 'default_OpenCap':
        pathIKFolder = os.path.join(pathOSData, 'IK')
    if collection_type == 'default':
        pathIKFolder = os.path.join(pathOSData, 'Kinematics')
    else:
        pathIKFolder = os.path.join(pathOSData, 'IK', OpenSimModel)
    pathGRFFolder = os.path.join(pathSession, 'ForceData')
    pathEMGFolder = os.path.join(pathSession, 'EMGData')
    pathIDFolder = os.path.join(pathOSDataMocap, 'ID', OpenSimModel)
    pathSOFolder = os.path.join(pathOSDataMocap, 'SO', OpenSimModel)
    pathIKFolderMocap = os.path.join(pathOSDataMocap, 'IK', OpenSimModel)
    pathJRAfromSOFolderMocap = os.path.join(pathOSDataMocap, 'JRA', OpenSimModel)
    pathModelFileMocap = os.path.join(pathOSDataMocap, 'Model', OpenSimModel,
                                      model_full_name + ".osim")  
    trials_list = [trial for trial in trials]
    listToStr = '_'.join([str(elem) for elem in trials_list])
    if collection_type == 'bigData' or collection_type == 'default_OpenCap':
        pathResults = os.path.join(pathOSData, 'DC', listToStr)
        if not rep is None:
            pathResults = os.path.join(pathOSData, 'DC', listToStr + '_rep' + str(rep))
    elif collection_type == 'default':
        pathResults = os.path.join(pathOSData, 'Dynamics', listToStr)
        if not rep is None:
            pathResults = os.path.join(pathOSData, 'Dynamics', listToStr + '_rep' + str(rep)) 
    else:
        pathResults = os.path.join(pathOSData, 'DC', OpenSimModel, listToStr)
        if not rep is None:
            pathResults = os.path.join(pathOSData, 'DC', OpenSimModel, listToStr + '_rep' + str(rep))        
    os.makedirs(pathResults, exist_ok=True)
    pathSettings = os.path.join(pathResults, 'Setup_{}.yaml'.format(case))
    if not processResults:
        with open(pathSettings, 'w') as file:
            yaml.dump(settings, file)
            
    # %% Uncomment to skip trial if existing solution     
    # temp_path = os.path.join(pathResults, 'w_opt_{}.npy'.format(case))
    # exist_result = os.path.exists(temp_path)
    # if exist_result:
    #     return
    
    # %% Muscles.
    if model_type == 'gait2392':
        muscles = [
            'glut_med1_r', 'glut_med2_r', 'glut_med3_r', 'glut_min1_r', 
            'glut_min2_r', 'glut_min3_r', 'semimem_r', 'semiten_r',
            'bifemlh_r', 'bifemsh_r', 'sar_r', 'add_long_r', 'add_brev_r',
            'add_mag1_r', 'add_mag2_r', 'add_mag3_r', 'tfl_r', 'pect_r',
            'grac_r', 'glut_max1_r', 'glut_max2_r', 'glut_max3_r',
            'iliacus_r', 'psoas_r', 'quad_fem_r', 'gem_r', 'peri_r',
            'rect_fem_r', 'vas_med_r', 'vas_int_r', 'vas_lat_r',
            'med_gas_r', 'lat_gas_r', 'soleus_r', 'tib_post_r',
            'flex_dig_r', 'flex_hal_r', 'tib_ant_r', 'per_brev_r',
            'per_long_r', 'per_tert_r', 'ext_dig_r', 'ext_hal_r', 'ercspn_r', 
            'intobl_r', 'extobl_r', 'ercspn_l', 'intobl_l', 'extobl_l']
        rightSideMuscles = muscles[:-3]
    elif model_type == 'rajagopal2016':
        muscles = [
            'addbrev_r', 'addlong_r', 'addmagDist_r', 'addmagIsch_r', 
            'addmagMid_r', 'addmagProx_r', 'bflh_r', 'bfsh_r',
            'edl_r', 'ehl_r', 'fdl_r', 'fhl_r', 'gaslat_r',
            'gasmed_r', 'glmax1_r', 'glmax2_r', 'glmax3_r', 'glmed1_r',
            'glmed2_r', 'glmed3_r', 'glmin1_r', 'glmin2_r',
            'glmin3_r', 'grac_r', 'iliacus_r', 'perbrev_r', 'perlong_r',
            'piri_r', 'psoas_r', 'recfem_r', 'sart_r',
            'semimem_r', 'semiten_r', 'soleus_r', 'tfl_r',
            'tibant_r', 'tibpost_r', 'vasint_r', 'vaslat_r', 'vasmed_r']
        rightSideMuscles = muscles        
    leftSideMuscles = [muscle[:-1] + 'l' for muscle in rightSideMuscles]
    bothSidesMuscles = leftSideMuscles + rightSideMuscles
    NMuscles = len(bothSidesMuscles)
    nSideMuscles = len(rightSideMuscles)
    
    # Data for muscle model.
    from muscleDataOpenSimAD import getMTParameters
    loadMTParameters = True
    if not os.path.exists(os.path.join(pathModelFolder, 
                                       model_full_name + '_mtParameters.npy')):
        loadMTParameters = False
    righSideMtParameters = getMTParameters(pathModelFile, rightSideMuscles,
                                           loadMTParameters, pathModelFolder,
                                           model_full_name)
    leftSideMtParameters = getMTParameters(pathModelFile, leftSideMuscles,
                                           loadMTParameters, pathModelFolder,
                                           model_full_name)
    mtParameters = np.concatenate((leftSideMtParameters, 
                                   righSideMtParameters), axis=1)
    mtParameters[0,:] = mtParameters[0,:] * scaleIsometricMuscleForce    
    from muscleDataOpenSimAD import tendonCompliance
    sideTendonCompliance = tendonCompliance(nSideMuscles)
    tendonCompliance = np.concatenate((sideTendonCompliance, 
                                       sideTendonCompliance), axis=1)    
    from muscleDataOpenSimAD import tendonShift
    sideTendonShift = tendonShift(nSideMuscles)
    tendonShift = np.concatenate((sideTendonShift, sideTendonShift), axis=1)    
    from functionCasADiOpenSimAD import hillEquilibrium
    if model_type == 'gait2392':
        from muscleDataOpenSimAD import specificTension_3D
        sideSpecificTension = specificTension_3D(rightSideMuscles)
        specificTension = np.concatenate((sideSpecificTension, 
                                          sideSpecificTension), axis=1)
        f_hillEquilibrium = hillEquilibrium(
            mtParameters, tendonCompliance, tendonShift, specificTension,
            ignorePassiveFiberForce=ignorePassiveFiberForce)
    else:
        specificTension = 0.5*np.ones((1, NMuscles))
        f_hillEquilibrium = hillEquilibrium(
            mtParameters, tendonCompliance, tendonShift, specificTension,
            ignorePassiveFiberForce=ignorePassiveFiberForce)
        
    # Time constants for activation dynamics.
    activationTimeConstant = 0.015
    deactivationTimeConstant = 0.06
    
    # Individual muscle weights.
    w_muscles = np.ones((NMuscles,1))
    if 'muscle_weights' in settings:
        for count, c_muscle in enumerate(bothSidesMuscles):
            if c_muscle in settings['muscle_weights']:
                w_muscles[count, 0] = (
                        settings['muscle_weights'][c_muscle]['weight'])
           
    # Muscle volume scaling.
    if volumeScaling:
        muscleVolume = np.multiply(mtParameters[0, :], mtParameters[1, :])
        s_muscleVolume = muscleVolume / np.sum(muscleVolume)
    else:
        s_muscleVolume = np.ones((NMuscles,))
    s_muscleVolume = np.reshape(s_muscleVolume, (NMuscles, 1))
    
    # %% Coordinates.
    from utilsOpenSimAD import getIndices
    # All coordinates.
    joints = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
              'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
              'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
              'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
              'knee_angle_l', 'knee_adduction_l',
              'knee_angle_r', 'knee_adduction_r',
              'ankle_angle_l', 'ankle_angle_r',
              'subtalar_angle_l', 'subtalar_angle_r', 
              'mtp_angle_l', 'mtp_angle_r', 
              'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    # Mtp coordinates.
    mtpJoints = ['mtp_angle_l', 'mtp_angle_r']
    if not withMTP:
        for joint in mtpJoints:
            joints.remove(joint)
    # KA coordinates.
    kaJoints = ['knee_adduction_l', 'knee_adduction_r']
    if not withKA:
        for joint in kaJoints:
            joints.remove(joint)
    # Lower-limb joints
    lowerLimbJoints = copy.deepcopy(joints)
    # Arm coordinates.
    armJoints = ['arm_flex_l', 'arm_add_l', 'arm_rot_l',
                 'arm_flex_r', 'arm_add_r', 'arm_rot_r',
                 'elbow_flex_l', 'elbow_flex_r', 'pro_sup_l', 'pro_sup_r']
    if withArms:
        for joint in armJoints:
            joints.append(joint)
    # Total count.
    nJoints = len(joints)
    
    # Translational coordinates.
    translationalJoints = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    
    # Rotational coordinates.
    rotationalJoints = copy.deepcopy(joints)
    for joint in translationalJoints:
        rotationalJoints.remove(joint)
    idxRotationalJoints = getIndices(joints, rotationalJoints)
    
    # Ground pelvis coordinates.
    groundPelvisJoints = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                          'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    # idxGroundPelvisJoints = getIndices(joints, groundPelvisJoints)
    
    # Helper indices.
    if withMTP:        
        nMtpJoints = len(mtpJoints)
        # idxMtpJoints = getIndices(joints, mtpJoints)    
    if withArms:        
        nArmJoints = len(armJoints)
        # idxArmJoints = getIndices(joints, armJoints)
    
    # Lumbar coordinates (for torque actuators).
    if not model_type == 'gait2392':        
        lumbarJoints = ['lumbar_extension', 'lumbar_bending',
                        'lumbar_rotation']
        nLumbarJoints = len(lumbarJoints)
        # idxLumbarJoints = getIndices(joints, lumbarJoints)
    
    # Coordinates with passive torques.
    # We here hard code the list to replicate previous results. 
    passiveTorqueJoints = [
        'hip_flexion_r', 'hip_flexion_l', 'hip_adduction_r', 
        'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l',              
        'knee_angle_r', 'knee_angle_l', 
        'ankle_angle_r', 'ankle_angle_l', 
        'subtalar_angle_r', 'subtalar_angle_l',
        'mtp_angle_r', 'mtp_angle_l',
        'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    if not withMTP:
        for joint in mtpJoints:
            passiveTorqueJoints.remove(joint)
    nPassiveTorqueJoints = len(passiveTorqueJoints)
        
    if periodicConstraints:        
        if 'Qs' in periodicConstraints:
            if 'lowerLimbJoints' in periodicConstraints['Qs']:
                idxPeriodicQs = getIndices(joints, lowerLimbJoints)
            else:
                idxPeriodicQs = []
                for joint in periodicConstraints['Qs']:
                    idxPeriodicQs.append(joints.index(joint))
        if 'Qds' in periodicConstraints:
            if 'lowerLimbJoints' in periodicConstraints['Qds']:
                idxPeriodicQds = getIndices(joints, lowerLimbJoints)
            else:
                idxPeriodicQds = []
                for joint in periodicConstraints['Qds']:
                    idxPeriodicQds.append(joints.index(joint))
        if 'muscles' in periodicConstraints:
            if 'all' in periodicConstraints['muscles']:
                idxPeriodicMuscles = getIndices(bothSidesMuscles, 
                                                     bothSidesMuscles)
            else:
                idxPeriodicMuscles = []
                for c_m in periodicConstraints['Qds']:
                    idxPeriodicMuscles.append(bothSidesMuscles.index(c_m))
        if 'lumbar' in periodicConstraints:
            if 'all' in periodicConstraints['lumbar']:
                idxPeriodicLumbar = getIndices(lumbarJoints, 
                                                    lumbarJoints)
            else:
                idxPeriodicLumbar = []
                for c_m in periodicConstraints['lumbar']:
                    idxPeriodicLumbar.append(lumbarJoints.index(c_m))
        
    # Muscle-driven coordinates.
    muscleDrivenJoints = [
        'hip_flexion_l', 'hip_flexion_r', 'hip_adduction_l', 
        'hip_adduction_r', 'hip_rotation_l', 'hip_rotation_r',              
        'knee_angle_l', 'knee_angle_r', 
        'ankle_angle_l', 'ankle_angle_r', 
        'subtalar_angle_l', 'subtalar_angle_r',
        'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    if not model_type == 'gait2392':
        for joint in lumbarJoints:
            muscleDrivenJoints.remove(joint)
        
    # %% Torque actuator activation dynamics
    if withArms or withActiveMTP or withLumbarCoordinateActuators:
        from functionCasADiOpenSimAD import armActivationDynamics
        if withArms:
            f_armActivationDynamics = armActivationDynamics(nArmJoints)
        if withActiveMTP:
            f_mtpActivationDynamics = armActivationDynamics(nMtpJoints)
        if withLumbarCoordinateActuators:
            f_lumbarActivationDynamics = armActivationDynamics(nLumbarJoints)
    
    # %% Metabolic energy.        
    """
    maximalIsometricForce = mtParameters[0, :]
    optimalFiberLength = mtParameters[1, :]
    muscleVolume = np.multiply(maximalIsometricForce, optimalFiberLength)
    muscleMass = np.divide(np.multiply(muscleVolume, 1059.7), 
                            np.multiply(specificTension[0, :].T, 1e6))
    from muscleDataOpenSimAD import slowTwitchRatio_3D
    sideSlowTwitchRatio = slowTwitchRatio_3D(rightSideMuscles)
    slowTwitchRatio = (np.concatenate((sideSlowTwitchRatio, 
                                      sideSlowTwitchRatio), axis=1))[0, :].T
    smoothingConstant = 10
    from functionCasADiOpenSimAD import metabolicsBhargava
    f_metabolicsBhargava = metabolicsBhargava(slowTwitchRatio, 
                                              maximalIsometricForce,
                                              muscleMass, 
                                              smoothingConstant)
    """
    
    # %% Passive/limit torques.
    from functionCasADiOpenSimAD import limitTorque, passiveTorque
    from muscleDataOpenSimAD import passiveJointTorqueData_3D    
    damping = 0.1
    f_passiveTorque = {}
    for joint in passiveTorqueJoints:
        f_passiveTorque[joint] = limitTorque(
            passiveJointTorqueData_3D(joint, model_type)[0],
            passiveJointTorqueData_3D(joint, model_type)[1], damping)    
    if withMTP:
        stiffnessMtp = 25
        dampingMtp = 2 # based on new results from pred sim study.
        f_linearPassiveMtpTorque = passiveTorque(stiffnessMtp, dampingMtp)        
    if withArms:
        stiffnessArm = 0
        dampingArm = 0.1
        f_linearPassiveArmTorque = passiveTorque(stiffnessArm, dampingArm)
    
    # %% External functions.       
    
    # if optimizeContacts:
    #     F1_c = ca.external('F', os.path.join(
    #         pathExternalFunctionFolder, 'nominal_c1.dll'))
    #     # NOutput_F1_c = 54
    #     F2_c = ca.external('F', os.path.join(
    #         pathExternalFunctionFolder, 'nominal_c2.dll'))
    #     NOutput_F2_c = nJoints
    #     if trackGRM:
    #         NOutput_F2_c += 6          
    #     if analyzeResults:
    #         F1 = ca.external('F', os.path.join(
    #             pathExternalFunctionFolder, 'nominal_c2_pp.dll'))
    #     F1_ref = ca.external('F', os.path.join(
    #         pathExternalFunctionFolder, 'nominal_pp.dll'))         
            
        F, F_map = {}, {}
        for trial in trials:
            F[trial] = ca.external(
                'F', os.path.join(pathExternalFunctionFolder, 'F.dll'))
            F_map[trial] = np.load(
                os.path.join(pathExternalFunctionFolder, 'F_map.npy'), 
                allow_pickle=True).item()

    # Example of how to call F with numerical values.
    # vec1 = np.ones((nJoints*2, 1))
    # vec2 = np.ones((nJoints, 1))
    # vec3 = np.concatenate((vec1,vec2))
    # res1 = F(vec3).full()
    
    # Indices outputs external function.
    for trial in trials:
        if isSquats[trial]:
            idx_vGRF_heel = [F_map[trial]['GRFs']['Sphere_0'][1],
                             F_map[trial]['GRFs']['Sphere_6'][1]]
        if isSTSs_yCalcn[trial] or isSTSs_yCalcn_vGRF[trial]:
            idx_yCalcn = [F_map[trial]['body_origins']['calcn_l'][1],
                          F_map[trial]['body_origins']['calcn_r'][1]]
            if isSTSs_yCalcn_vGRF[trial]:
                idx_vGRF = []
                for contactSphere in range(2*nContactSpheres):
                    idx_vGRF.append(F_map[trial]['GRFs'][
                        'Sphere_{}'.format(contactSphere)][1])
                idx_vGRF_heel_l = [idx_vGRF[0+nContactSpheres], 
                                   idx_vGRF[3+nContactSpheres]]
                idx_vGRF_front_l = [idx_vGRF[1+nContactSpheres], 
                                    idx_vGRF[2+nContactSpheres], 
                                    idx_vGRF[4+nContactSpheres], 
                                    idx_vGRF[5+nContactSpheres]]
                idx_vGRF_heel_r = [idx_vGRF[0], idx_vGRF[3]]
                idx_vGRF_front_r = [idx_vGRF[1], idx_vGRF[2], 
                                    idx_vGRF[4], idx_vGRF[5]]
                idx_vGRF_heel_lr = [idx_vGRF[0+nContactSpheres], idx_vGRF[0]]
        if yCalcnToes[trial]:            
            idx_yCalcnToes = [F_map[trial]['body_origins']['calcn_l'][1],
                              F_map[trial]['body_origins']['calcn_r'][1],
                              F_map[trial]['body_origins']['toes_l'][1],
                              F_map[trial]['body_origins']['toes_r'][1]]
    
    # For analysis
    spheres = ['s{}'.format(i) for i in range(1, nContactSpheres+1)]    
    idxGR = {}
    idxGR["GRF"] = {}
    idxGR["GRF"]["all"] = {}
    idxGR["COP"] = {}     
    idxGR["GRM"] = {}
    idxGR["GRM"]["all"] = {}
    for sphere in spheres:
        idxGR["GRF"][sphere] = {}
        idxGR["COP"][sphere] = {}        
    sides_all = ['right', 'left']
    for c_side, side in enumerate(sides_all):
        idxGR['GRF']["all"][side[0]] = list(F_map[trial]['GRFs'][side])
        idxGR['GRM']["all"][side[0]] = list(F_map[trial]['GRMs'][side])
        for c_sphere, sphere in enumerate(spheres):
            idxGR['GRF'][sphere][side[0]] = list(F_map[trial]['GRFs'][
                'Sphere_{}'.format(c_sphere + c_side*len(spheres))])
            idxGR['COP'][sphere][side[0]] = list(F_map[trial]['COPs'][
                'Sphere_{}'.format(c_sphere + c_side*len(spheres))])
            
    # Helper lists to map order of joints defined here and in F.
    idxGroundPelvisJointsinF = [F_map[trial]['residuals'][joint] 
                                for joint in groundPelvisJoints]    
    idxJoints4F = [joints.index(joint) 
                   for joint in list(F_map[trial]['residuals'].keys())]
        
    # %% Polynomials
    from functionCasADiOpenSimAD import polynomialApproximation
    leftPolynomialJoints = [
        'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l',
        'knee_adduction_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', 
        'lumbar_extension', 'lumbar_bending', 'lumbar_rotation'] 
    rightPolynomialJoints = [
        'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r',
        'knee_adduction_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
        'lumbar_extension', 'lumbar_bending', 'lumbar_rotation'] 
    if not withKA:
        leftPolynomialJoints.remove('knee_adduction_l')
        rightPolynomialJoints.remove('knee_adduction_r') 
    if not withMTP:
        leftPolynomialJoints.remove('mtp_angle_l')
        rightPolynomialJoints.remove('mtp_angle_r')
    if model_type == 'rajagopal2016':
        leftPolynomialJoints.remove('lumbar_extension')
        leftPolynomialJoints.remove('lumbar_bending')
        leftPolynomialJoints.remove('lumbar_rotation')
        rightPolynomialJoints.remove('lumbar_extension')
        rightPolynomialJoints.remove('lumbar_bending')
        rightPolynomialJoints.remove('lumbar_rotation')
        
    pathGenericTemplates = os.path.join(baseDir, "opensimPipeline") 
    pathDummyMotion = os.path.join(pathGenericTemplates, "MuscleAnalysis", 
                                   'DummyMotion_{}.mot'.format(model_type))
    loadPolynomialData = True
    if (not os.path.exists(os.path.join(
            pathModelFolder, model_full_name + '_polynomial_r_' + 
            nameMA +'.npy'))
            or not os.path.exists(os.path.join(
            pathModelFolder, model_full_name + '_polynomial_l_' + 
            nameMA +'.npy'))):
        loadPolynomialData = False
    
    from muscleDataOpenSimAD import getPolynomialData
    polynomialData = {}
    polynomialData['r'] = getPolynomialData(
        loadPolynomialData, pathModelFolder, model_full_name, pathDummyMotion, 
        rightPolynomialJoints, muscles, nameMA, side='r')
    polynomialData['l'] = getPolynomialData(
        loadPolynomialData, pathModelFolder, model_full_name, pathDummyMotion, 
        leftPolynomialJoints, leftSideMuscles, nameMA, side='l')
     
    if loadPolynomialData:
        polynomialData['r'] = polynomialData['r'].item()
        polynomialData['l'] = polynomialData['l'].item()
    
    sides = ['r', 'l']
    for side in sides:
        for c_pol in polynomialData[side]:
            assert (np.max(polynomialData[side][c_pol]['coefficients']) < 1), (
                "coeffs {}".format(side))
        
    NPolynomials = len(leftPolynomialJoints)
    f_polynomial = {}
    f_polynomial['r'] = polynomialApproximation(
        muscles, polynomialData['r'], NPolynomials)
    f_polynomial['l'] = polynomialApproximation(
        leftSideMuscles, polynomialData['l'], NPolynomials)
    
    leftPolynomialJointIndices = getIndices(joints, 
                                                 leftPolynomialJoints)
    rightPolynomialJointIndices = getIndices(joints, 
                                                  rightPolynomialJoints)
    if model_type == 'gait2392':
        leftPolynomialMuscleIndices = (
            list(range(nSideMuscles-3)) + 
            list(range(nSideMuscles, nSideMuscles+3)))
    elif model_type == 'rajagopal2016':
        leftPolynomialMuscleIndices = (
            list(range(nSideMuscles)) + 
            list(range(nSideMuscles, nSideMuscles)))
    rightPolynomialMuscleIndices = list(range(nSideMuscles))
    from utilsOpenSimAD import getMomentArmIndices
    momentArmIndices = getMomentArmIndices(rightSideMuscles,
                                           leftPolynomialJoints,
                                           rightPolynomialJoints, 
                                           polynomialData['r'])
    if model_type == 'gait2392':
        trunkMomentArmPolynomialIndices = (
            list(range(nSideMuscles, nSideMuscles+3)) + 
            list(range(nSideMuscles-3, nSideMuscles)))
    
    # Test polynomials
    if plotPolynomials:
        from polynomialsOpenSimAD import testPolynomials
        if model_type == 'gait2392':
            raise ValueError("Test polynomials gait2392 not supported")
        elif model_type == 'rajagopal2016':
            path_data4PolynomialFitting = os.path.join(
                pathModelFolder, 
                'data4PolynomialFitting_{}.npy'.format(model_full_name))
            data4PolynomialFitting = np.load(path_data4PolynomialFitting, 
                                             allow_pickle=True).item()            
            testPolynomials(
                data4PolynomialFitting, rightPolynomialJoints,
                muscles, f_polynomial['r'], polynomialData['r'], 
                momentArmIndices)
            testPolynomials(
                data4PolynomialFitting, leftPolynomialJoints,
                leftSideMuscles, f_polynomial['l'], polynomialData['l'], 
                momentArmIndices)
            
    if data_type == 'Video':
        
        if bounds_update:
            if bounds_update_index == 1:
                polynomial_bounds = {
                        'hip_flexion_l': {'max': 120, 'min': -30},
                        'hip_flexion_r': {'max': 120, 'min': -30},
                        'hip_adduction_l': {'max': 20, 'min': -40},
                        'hip_adduction_r': {'max': 20, 'min': -40},
                        'hip_rotation_l': {'max': 30, 'min': -40},
                        'hip_rotation_r': {'max': 30, 'min': -40},
                        'knee_angle_l': {'max': 125, 'min': 0},
                        'knee_angle_r': {'max': 125, 'min': 0},
                        'knee_adduction_l': {'max': 20, 'min': -30},
                        'knee_adduction_r': {'max': 20, 'min': -30},
                        'ankle_angle_l': {'max': 50, 'min': -50},
                        'ankle_angle_r': {'max': 50, 'min': -50},
                        'subtalar_angle_l': {'max': 35, 'min': -35},
                        'subtalar_angle_r': {'max': 35, 'min': -35},
                        'mtp_angle_l': {'max': 5, 'min': -45},
                        'mtp_angle_r': {'max': 5, 'min': -45}}                
            elif bounds_update_index == 2:                
                polynomial_bounds = {
                        'hip_flexion_l': {'max': 120, 'min': -30},
                        'hip_flexion_r': {'max': 120, 'min': -30},
                        'hip_adduction_l': {'max': 20, 'min': -50},
                        'hip_adduction_r': {'max': 20, 'min': -50},
                        'hip_rotation_l': {'max': 35, 'min': -40},
                        'hip_rotation_r': {'max': 35, 'min': -40},
                        'knee_angle_l': {'max': 138, 'min': 0},
                        'knee_angle_r': {'max': 138, 'min': 0},
                        'knee_adduction_l': {'max': 20, 'min': -30},
                        'knee_adduction_r': {'max': 20, 'min': -30},
                        'ankle_angle_l': {'max': 50, 'min': -50},
                        'ankle_angle_r': {'max': 50, 'min': -50},
                        'subtalar_angle_l': {'max': 35, 'min': -35},
                        'subtalar_angle_r': {'max': 35, 'min': -35},
                        'mtp_angle_l': {'max': 5, 'min': -45},
                        'mtp_angle_r': {'max': 5, 'min': -45}}                
            else:
                raise ValueError('bounds_update_index non known')                
        else:
            # This is what used to be used as ocp bounds, not matching what was
            # used for fitting the coefficients (big data 107)
            polynomial_bounds = {
                    'hip_flexion_l': {'max': 120, 'min': -30},
                    'hip_flexion_r': {'max': 120, 'min': -30},
                    'hip_adduction_l': {'max': 20, 'min': -35},
                    'hip_adduction_r': {'max': 20, 'min': -35},
                    'hip_rotation_l': {'max': 25, 'min': -30},
                    'hip_rotation_r': {'max': 25, 'min': -30},
                    'knee_angle_l': {'max': 120, 'min': 0},
                    'knee_angle_r': {'max': 120, 'min': 0},
                    'knee_adduction_l': {'max': 20, 'min': -30},
                    'knee_adduction_r': {'max': 20, 'min': -30},
                    'ankle_angle_l': {'max': 50, 'min': -40},
                    'ankle_angle_r': {'max': 50, 'min': -40},
                    'subtalar_angle_l': {'max': 20, 'min': -35},
                    'subtalar_angle_r': {'max': 20, 'min': -35},
                    'mtp_angle_l': {'max': 5, 'min': -45},
                    'mtp_angle_r': {'max': 5, 'min': -45}}            
    elif data_type == 'Mocap':
        polynomial_bounds = {
                'hip_flexion_l': {'max': 120, 'min': -30},
                'hip_flexion_r': {'max': 120, 'min': -30},
                'hip_adduction_l': {'max': 20, 'min': -40},
                'hip_adduction_r': {'max': 20, 'min': -40},
                'hip_rotation_l': {'max': 30, 'min': -40},
                'hip_rotation_r': {'max': 30, 'min': -40},
                'knee_angle_l': {'max': 125, 'min': 0},
                'knee_angle_r': {'max': 125, 'min': 0},
                'knee_adduction_l': {'max': 20, 'min': -30},
                'knee_adduction_r': {'max': 20, 'min': -30},
                'ankle_angle_l': {'max': 50, 'min': -50},
                'ankle_angle_r': {'max': 50, 'min': -50},
                'subtalar_angle_l': {'max': 35, 'min': -35},
                'subtalar_angle_r': {'max': 35, 'min': -35},
                'mtp_angle_l': {'max': 5, 'min': -45},
                'mtp_angle_r': {'max': 5, 'min': -45}}
        
    # %% Contacts.
    # NContactParameters = 0
    # if optimizeContacts:        
    #     # Indices in external function (c1)
    #     foot_segments = ['calcn_l', 'calcn_r', 'toes_l', 'toes_r']
    #     outs_c1 = {'angular_velocity': 3, 'linear_velocity': 3,
    #               'position': 3, 'rotation': 9, 'translation': 3}
    #     idxF_c1 = {}
    #     idx_acc = 0
    #     for segment in foot_segments:
    #         idxF_c1[segment] = {}
    #         for out_c1 in outs_c1:
    #             idxF_c1[segment][out_c1] = list(
    #                 range(idx_acc, idx_acc+outs_c1[out_c1]))
    #             idx_acc += outs_c1[out_c1]       
        
    #     # Default values
    #     normal = np.array([[0, 1, 0]])
    #     transitionVelocity = 0.2
    #     staticFriction = 0.8
    #     dynamicFriction = 0.8
    #     viscousFriction = 0.5
    #     stiffness = 1e6
    #     dissipation = 2.0
        
    #     # Contact model.
    #     from functionCasADiOpenSimAD import smoothSphereHalfSpaceForce
    #     f_contactForce = smoothSphereHalfSpaceForce(
    #         transitionVelocity, staticFriction, dynamicFriction,
    #         viscousFriction, normal)
    
    #     if parameter_to_optimize == "option1":
    #         NContactParameters = nContactSpheres*3       
    #         contactParameterNames = []
    #         for s in range(nContactSpheres):
    #             contactParameterNames.append("s{}_loc_x".format(s+1))
    #             contactParameterNames.append("s{}_loc_z".format(s+1))
    #         for s in range(nContactSpheres):
    #             contactParameterNames.append("s{}_radius".format(s+1))
                
    #     if trackGRM:
    #         idxGRM = list(range(nJoints, nJoints+6))
            
    #     # Test implementation.
    #     vecF = -np.ones((99, 1))
    #     resF = (F1_ref(vecF)).full()        
    #     vecF_c1 = -np.ones((66, 1))
    #     resF_c1 = (F1_c(vecF_c1)).full()        
    #     RBG, posB, TBG, lVelB, aVelB = {}, {}, {}, {}, {}
    #     for segment in foot_segments:
    #         RBG_flat = resF_c1[idxF_c1[segment]['rotation']]
    #         RBG[segment] = np.reshape(RBG_flat, (3,3))
    #         posB[segment] = resF_c1[idxF_c1[segment]['position']]
    #         TBG[segment] = resF_c1[idxF_c1[segment]['translation']]
    #         lVelB[segment] = resF_c1[idxF_c1[segment]['linear_velocity']]
    #         aVelB[segment] = resF_c1[idxF_c1[segment]['angular_velocity']]        
    #     if nContactSpheres == 6:
    #         # Default values
    #         radius = 0.032
    #         locSpheres = np.array(
    #             [[0.00215773306688716053,  -0.01, -0.00434269152195360195],
    #              [0.16841223157345971972,  -0.01, -0.03258850869005603529],
    #              [0.15095065283989317351,  -0.01,  0.05860493716970469752],
    #              [0.07517351958454182581,  -0.01,  0.02992219727974926649],
    #              [0.06809743951165971032,  -0.01, -0.02129214951175864221],
    #              [0.05107307963374478621,  -0.01, 0.07020500618327656095]])            
    #         locSphere_inB = {}
    #         for n_c in range(nContactSpheres):
    #             locSphere_inB[str(n_c+1)] = {}
    #             locSphere_inB[str(n_c+1)]['r'] = locSpheres[n_c, :].T
    #             locSphere_inB[str(n_c+1)]['l'] = (
    #                 locSpheres[n_c, :].T * np.array([1,1,-1]))
    #         appliedF = {}
    #         side_contacts = ['r', 'l']
    #         contactSegments = ['calcn', 'calcn', 'calcn', 'calcn', 
    #                            'toes', 'toes']
    #         for n_c in range(nContactSpheres):
    #             c_n = str(n_c+1)
    #             appliedF[c_n] = {}
    #             for s_c in side_contacts:
    #                 c_s = contactSegments[n_c] + '_' + s_c                    
    #                 appliedF[c_n][s_c] = f_contactForce(
    #                     dissipation, stiffness, radius,
    #                     locSphere_inB[c_n][s_c], posB[c_s], lVelB[c_s],
    #                     aVelB[c_s], RBG[c_s], TBG[c_s])                    
    #         in_c2 = ca.DM(1, nContactSpheres*3*3)            
    #         count = 0
    #         for n_c in range(nContactSpheres):
    #             c_n = str(n_c+1)
    #             for s_c in side_contacts:
    #                 in_c2[0, count*3:count*3+3] = appliedF[c_n][s_c]
    #                 count += 1
    #             in_c2[0, nContactSpheres*2*3+n_c*2] = (
    #                 locSphere_inB[c_n]['r'][0,])
    #             in_c2[0, nContactSpheres*2*3+n_c*2+1] = (
    #                 locSphere_inB[c_n]['r'][2,])             
    #             in_c2[0, nContactSpheres*2*4+n_c] = radius
    #         resF_c2 = (F2_c(ca.vertcat(vecF,in_c2.T))).full()        
    #     diffRes = resF[:nJoints] - resF_c2[:nJoints]
    #     assert np.alltrue(np.abs(diffRes) < 10**(-8)), (
    #         "Error contact implementation")
    #     idxGRM_pp = idxGR["GRM"]["all"]['r'] + idxGR["GRM"]["all"]['l']
    #     diffRes2 = resF[idxGRM_pp] - resF_c2[idxGRM]
    #     assert np.alltrue(np.abs(diffRes2) < 10**(-8)), (
    #         "Error contact implementation 2")
    # else:
    #     if trackGRF or trackGRM:
    #         idxGRF = idxGR["GRF"]["all"]['r'] + idxGR["GRF"]["all"]['l']
    #         idxGRM = idxGR["GRM"]["all"]['r'] + idxGR["GRM"]["all"]['l']
            
    # %% Kinematic data
    from utilsOpenSimAD import getIK, filterDataFrame
    from utilsOpenSimAD import interpolateDataFrame, selectDataFrame
    Qs_toTrack = {}
    Qs_toTrack_sel = {}
    for trial in trials:
        pathIK = os.path.join(pathIKFolder, trial + '.mot')        
        # Extract joint positions from walking trial.
        Qs_fromIK = getIK(pathIK, joints)
        if filter_coordinates_toTracks[trial]:
            Qs_fromIK_filter = filterDataFrame(
                Qs_fromIK, cutoff_frequency=cutoff_freq_coords[trial])
        else:
            Qs_fromIK_filter = Qs_fromIK           
        # Interpolation.
        Qs_fromIK_interp = interpolateDataFrame(
            Qs_fromIK_filter, timeIntervals[trial][0], timeIntervals[trial][1],
            N[trial])
        Qs_toTrack[trial] = copy.deepcopy(Qs_fromIK_interp)
        # We do not want to down-sample before differentiating the splines.
        Qs_fromIK_sel = selectDataFrame(
            Qs_fromIK_filter, timeIntervals[trial][0], timeIntervals[trial][1])
        Qs_toTrack_sel[trial] = copy.deepcopy(Qs_fromIK_sel) 
    nEl_toTrack = len(coordinates_toTrack)
    # Individual coordinate weigths.
    w_dataToTrack = np.ones((nEl_toTrack,1))
    coordinates_toTrack_list = []
    for count, coord in enumerate(coordinates_toTrack):
        coordinates_toTrack_list.append(coord)
        if 'weight' in coordinates_toTrack[coord]:
            w_dataToTrack[count, 0] = coordinates_toTrack[coord]['weight']      
    idx_coordinates_toTrack = getIndices(joints, coordinates_toTrack_list)
    
    # %% GRF data     
    from utilsOpenSimAD import getGRFAll, getGRFPeaks 
    GRF_toTrack = {}
    GRF_peaks = {}
    for trial in trials:
        # Experimental data do not have the videoAndMocap. It is a bit hacky,
        # but we here 
        trial_exp =  trial
        if 'videoAndMocap' in trial:
            trial_exp = trial[:-14]
        pathGRFFile = os.path.join(pathGRFFolder, trial_exp + '_forces.mot')
        if os.path.exists(pathGRFFile):
            GRF_toTrack[trial] = getGRFAll(
                pathGRFFile, timeIntervals[trial], N[trial])            
            # Select peak from non-interpolated data.
            GRF_peaks[trial] = getGRFPeaks(GRF_toTrack[trial], 
                                           timeIntervals[trial])
            
    # %% ID data
    from utilsOpenSimAD import getID        
    ID_toTrack = {}
    for trial in trials: 
        # Experimental data do not have the videoAndMocap. It is a bit hacky,
        # but we here 
        trial_exp =  trial
        if 'videoAndMocap' in trial:
            trial_exp = trial[:-14]
        pathIDFile = os.path.join(pathIDFolder, trial_exp + '.sto')
        if os.path.exists(pathIDFile):
            ID_temp = getID(pathIDFile, joints)
            ID_toTrack[trial] = interpolateDataFrame(
                ID_temp, timeIntervals[trial][0], timeIntervals[trial][1],
                N[trial]).to_numpy()[:,1::].T
            
    # %% EMG data
    from utilsOpenSimAD import getEMG        
    EMG_ref = {}
    for trial in trials: 
        # Experimental data do not have the videoAndMocap. It is a bit hacky,
        # but we here 
        trial_exp =  trial
        if 'videoAndMocap' in trial:
            trial_exp = trial[:-14]
        pathEMGFile = os.path.join(pathEMGFolder, trial_exp + '_EMG.sto')
        if os.path.exists(pathEMGFile):
            EMG_temp = getEMG(pathEMGFile, bothSidesMuscles)
            EMG_ref[trial] = interpolateDataFrame(
                EMG_temp, timeIntervals[trial][0], timeIntervals[trial][1],
                N[trial]).to_numpy()[:,1::].T
            
    # %% SO data
    from utilsOpenSimAD import getFromStorage
    SO_toTrack = {}
    for trial in trials: 
        # Experimental data do not have the videoAndMocap. It is a bit hacky,
        # but we here 
        trial_exp =  trial
        if 'videoAndMocap' in trial:
            trial_exp = trial[:-14]
        pathSOFile = os.path.join(
            pathSOFolder, trial_exp + '_StaticOptimization_activation.sto')
        if os.path.exists(pathSOFile):
            SO_temp = getFromStorage(pathSOFile, bothSidesMuscles)
            SO_toTrack[trial] = interpolateDataFrame(
                SO_temp, timeIntervals[trial][0], timeIntervals[trial][1],
                N[trial]).to_numpy()[:,1::].T
            
    # %% Mocap-based joint positions, velocities, and accelerations.
    Qs_mocap_ref = {}
    Qds_mocap_ref = {}
    Qdds_mocap_ref = {}
    for trial in trials:
        # Experimental data do not have the videoAndMocap. It is a bit hacky,
        # but we here 
        trial_exp =  trial
        if 'videoAndMocap' in trial:
            trial_exp = trial[:-14]
        pathIKFileMocap = os.path.join(pathIKFolderMocap, trial_exp + '.mot')
        if os.path.exists(pathIKFileMocap):
            # Extract joint positions from walking trial.
            c_Ik_mocap = getIK(pathIKFileMocap, joints)
            if filter_coordinates_toTracks[trial]:
                c_Ik_mocap_filter = filterDataFrame(
                    c_Ik_mocap, cutoff_frequency=cutoff_freq_coords[trial])
            else:
                c_Ik_mocap_filter = c_Ik_mocap
            c_Ik_mocap_sel = selectDataFrame(
                c_Ik_mocap_filter, timeIntervals[trial][0],
                timeIntervals[trial][1])
            Qs_mocap_ref[trial] = interpolateDataFrame(
                c_Ik_mocap_filter, timeIntervals[trial][0], 
                timeIntervals[trial][1],
                N[trial]).to_numpy()[:,1::].T
            Qs_mocap_spline = c_Ik_mocap_sel.copy()
            Qds_mocap_spline = c_Ik_mocap_sel.copy()
            Qdds_mocap_spline = c_Ik_mocap_sel.copy()
            for joint in joints:
                spline = interpolate.InterpolatedUnivariateSpline(
                    c_Ik_mocap_sel['time'], 
                    c_Ik_mocap_sel[joint], k=3)
                Qs_mocap_spline[joint] = spline(
                    c_Ik_mocap_sel['time'])
                splineD1 = spline.derivative(n=1)
                Qds_mocap_spline[joint] = splineD1(
                    c_Ik_mocap_sel['time'])
                splineD2 = spline.derivative(n=2)
                Qdds_mocap_spline[joint] = splineD2(
                    c_Ik_mocap_sel['time'])
            
            # Filtering
            if filter_Qds_toTracks[trial]:
                Qds_mocap_spline_filter = filterDataFrame(
                    Qds_mocap_spline,
                    cutoff_frequency=cutoff_freq_Qds[trial])
            else:
                Qds_mocap_spline_filter = Qds_mocap_spline                
            if filter_Qdds_toTracks[trial]:
                Qdds_mocap_spline_filter = filterDataFrame(
                    Qdds_mocap_spline,
                    cutoff_frequency=cutoff_freq_Qdds[trial])
            else:
                Qdds_mocap_spline_filter = Qdds_mocap_spline
                
            # Instead of splining Qs twice to get Qdds, spline Qds, which can
            # be filtered or not.
            if splineQds[trial]:
                Qdds_mocap_spline2 = c_Ik_mocap_sel.copy()
                for joint in joints:
                    spline = interpolate.InterpolatedUnivariateSpline(
                        Qds_mocap_spline_filter['time'], 
                        Qds_mocap_spline_filter[joint], k=3)
                    splineD1 = spline.derivative(n=1)
                    Qdds_mocap_spline2[joint] = splineD1(
                        Qds_mocap_spline_filter['time'])
                    
                if filter_Qdds_toTracks[trial]:
                    Qdds_mocap_spline_filter = filterDataFrame(
                        Qdds_mocap_spline2, 
                        cutoff_frequency=cutoff_freq_Qdds[trial])
                else:
                    Qdds_mocap_spline_filter = Qdds_mocap_spline2
            
            # Interpolation
            Qds_mocap_ref[trial] = interpolateDataFrame(
                Qds_mocap_spline_filter, timeIntervals[trial][0], 
                timeIntervals[trial][1], N[trial]).to_numpy()[:,1::].T
            Qdds_mocap_ref[trial] = interpolateDataFrame(
                Qdds_mocap_spline_filter, timeIntervals[trial][0], 
                timeIntervals[trial][1], N[trial]).to_numpy()[:,1::].T            
   
    # %% Other helper CasADi functions
    from functionCasADiOpenSimAD import normSumSqr
    from functionCasADiOpenSimAD import normSumWeightedPow
    from functionCasADiOpenSimAD import diffTorques
    from functionCasADiOpenSimAD import normSumWeightedSqrDiff
    f_NMusclesSum2 = normSumSqr(NMuscles)
    f_NMusclesSumWeightedPow = normSumWeightedPow(NMuscles, powActivations)
    f_nJointsSum2 = normSumSqr(nJoints)
    # f_NQsToTrackSum2 = normSumSqr(nEl_toTrack)
    f_NQsToTrackWSum2 = normSumWeightedSqrDiff(nEl_toTrack)
    if withActiveMTP:
        f_nMtpJointsSum2 = normSumSqr(nMtpJoints)
    if withArms:
        f_nArmJointsSum2 = normSumSqr(nArmJoints)
    if withLumbarCoordinateActuators:
        f_nLumbarJointsSum2 = normSumSqr(nLumbarJoints)
    if trackGRF or trackGRM:
        f_NGRToTrackSum2 = normSumSqr(6)        
    # f_nPassiveTorqueJointsSum2 = normSumSqr(nPassiveTorqueJoints)
    f_diffTorques = diffTorques()
    if powerDiff == 2:
        f_normSumSqrDiff_tracking = normSumWeightedSqrDiff(nEl_toTrack)
    else:
       from functionCasADiOpenSimAD import normSumWeightedPowDiff
       f_normSumSqrDiff_tracking = normSumWeightedPowDiff(powerDiff, 
                                                          nEl_toTrack)     
        
    # %% Bounds
    from boundsOpenSimAD import bounds_tracking
    # Pre-allocations
    uw, lw, scaling = {}, {}, {}    
    uw['A'], lw['A'], scaling['A'] = {}, {}, {}
    uw['Ak'], lw['Ak'], scaling['Ak'] = {}, {}, {}
    uw['Aj'], lw['Aj'], scaling['Aj'] = {}, {}, {}    
    uw['F'], lw['F'], scaling['F'] = {}, {}, {}
    uw['Fk'], lw['Fk'], scaling['Fk'] = {}, {}, {}
    uw['Fj'], lw['Fj'], scaling['Fj'] = {}, {}, {}    
    uw['Qs'], lw['Qs'], scaling['Qs'] = {}, {}, {}
    uw['Qsk'], lw['Qsk'], scaling['Qsk'] = {}, {}, {}
    uw['Qsj'], lw['Qsj'], scaling['Qsj'] = {}, {}, {}    
    uw['Qds'], lw['Qds'], scaling['Qds'] = {}, {}, {}
    uw['Qdsk'], lw['Qdsk'], scaling['Qdsk'] = {}, {}, {}
    uw['Qdsj'], lw['Qdsj'], scaling['Qdsj'] = {}, {}, {}    
    if withArms:
        uw['ArmA'], lw['ArmA'], scaling['ArmA'] = {}, {}, {}
        uw['ArmAk'], lw['ArmAk'], scaling['ArmAk'] = {}, {}, {}
        uw['ArmAj'], lw['ArmAj'], scaling['ArmAj'] = {}, {}, {}
        uw['ArmE'], lw['ArmE'], scaling['ArmE'] = {}, {}, {}
        uw['ArmEk'], lw['ArmEk'], scaling['ArmEk'] = {}, {}, {}  
    if withActiveMTP:
        uw['MtpA'], lw['MtpA'], scaling['MtpA'] = {}, {}, {}
        uw['MtpAk'], lw['MtpAk'], scaling['MtpAk'] = {}, {}, {}
        uw['MtpAj'], lw['MtpAj'], scaling['MtpAj'] = {}, {}, {}
        uw['MtpE'], lw['MtpE'], scaling['MtpE'] = {}, {}, {}
        uw['MtpEk'], lw['MtpEk'], scaling['MtpEk'] = {}, {}, {}
    if withLumbarCoordinateActuators:
        uw['LumbarA'], lw['LumbarA'], scaling['LumbarA'] = {}, {}, {}
        uw['LumbarAk'], lw['LumbarAk'], scaling['LumbarAk'] = {}, {}, {}
        uw['LumbarAj'], lw['LumbarAj'], scaling['LumbarAj'] = {}, {}, {}
        uw['LumbarE'], lw['LumbarE'], scaling['LumbarE'] = {}, {}, {}
        uw['LumbarEk'], lw['LumbarEk'], scaling['LumbarEk'] = {}, {}, {}  
    uw['ADt'], lw['ADt'], scaling['ADt'] = {}, {}, {}
    uw['ADtk'], lw['ADtk'], scaling['ADtk'] = {}, {}, {}  
           
    uw['FDt'], lw['FDt'], scaling['FDt'] = {}, {}, {}
    uw['FDtk'], lw['FDtk'], scaling['FDtk'] = {}, {}, {}    
    uw['Qdds'], lw['Qdds'], scaling['Qdds'] = {}, {}, {}
    uw['Qddsk'], lw['Qddsk'], scaling['Qddsk'] = {}, {}, {}    
    if offset_ty:
        uw['Offset'], lw['Offset'], scaling['Offset'] = {}, {}, {}
        uw['Offsetk'], lw['Offsetk'], scaling['Offsetk'] = {}, {}, {}    
    if optimizeContacts:
        uw['ContactParameters'], lw['ContactParameters'], scaling['ContactParameters_v'], scaling['ContactParameters_r'] = {}, {}, {}, {}    
    if GRF_toTrack:
        scaling['GRF'] = {}
        scaling['GRM'] = {}
    if reserveActuators:
        uw['rAct'], lw['rAct'], scaling['rAct'] = {}, {}, {}
        uw['rActk'], lw['rActk']= {}, {}
        
    # Loop over trials 
    for trial in trials:
        bounds = bounds_tracking(Qs_toTrack[trial], joints, rightSideMuscles)   
        
        ###########################################################################
        # States
        ###########################################################################
        # Muscle activations
        uw['A'][trial], lw['A'][trial], scaling['A'][trial] = bounds.getBoundsActivation(lb_activation=lb_activation)
        uw['Ak'][trial] = ca.vec(uw['A'][trial].to_numpy().T * np.ones((1, N[trial]+1))).full()
        lw['Ak'][trial] = ca.vec(lw['A'][trial].to_numpy().T * np.ones((1, N[trial]+1))).full()
        uw['Aj'][trial] = ca.vec(uw['A'][trial].to_numpy().T * np.ones((1, d*N[trial]))).full()
        lw['Aj'][trial] = ca.vec(lw['A'][trial].to_numpy().T * np.ones((1, d*N[trial]))).full()
        # Muscle forces
        uw['F'][trial], lw['F'][trial], scaling['F'][trial] = bounds.getBoundsForce()
        uw['Fk'][trial] = ca.vec(uw['F'][trial].to_numpy().T * np.ones((1, N[trial]+1))).full()
        lw['Fk'][trial] = ca.vec(lw['F'][trial].to_numpy().T * np.ones((1, N[trial]+1))).full()
        uw['Fj'][trial] = ca.vec(uw['F'][trial].to_numpy().T * np.ones((1, d*N[trial]))).full()
        lw['Fj'][trial] = ca.vec(lw['F'][trial].to_numpy().T * np.ones((1, d*N[trial]))).full()
        # Joint positions
        # uw['Qs'][trial], lw['Qs'][trial], scaling['Qs'][trial] =  bounds.getBoundsPosition()
        if bounds_update:
            if bounds_update_index == 1:
                uw['Qs'][trial], lw['Qs'][trial], scaling['Qs'][trial] =  bounds.getBoundsPosition_fixed_update1(data_type=data_type)    
            elif bounds_update_index == 2:
                uw['Qs'][trial], lw['Qs'][trial], scaling['Qs'][trial] =  bounds.getBoundsPosition_fixed_update2(data_type=data_type)       
            else:
                raise ValueError('Unknown bounds_index')
        else:
            uw['Qs'][trial], lw['Qs'][trial], scaling['Qs'][trial] =  bounds.getBoundsPosition_fixed(data_type=data_type)
        uw['Qsk'][trial] = ca.vec(uw['Qs'][trial].to_numpy().T * np.ones((1, N[trial]+1))).full()
        lw['Qsk'][trial] = ca.vec(lw['Qs'][trial].to_numpy().T * np.ones((1, N[trial]+1))).full()
        uw['Qsj'][trial] = ca.vec(uw['Qs'][trial].to_numpy().T * np.ones((1, d*N[trial]))).full()
        lw['Qsj'][trial] = ca.vec(lw['Qs'][trial].to_numpy().T * np.ones((1, d*N[trial]))).full()
        # Joint velocities
        uw['Qds'][trial], lw['Qds'][trial], scaling['Qds'][trial] = bounds.getBoundsVelocity()
        uw['Qdsk'][trial] = ca.vec(uw['Qds'][trial].to_numpy().T*np.ones((1, N[trial]+1))).full()
        lw['Qdsk'][trial] = ca.vec(lw['Qds'][trial].to_numpy().T*np.ones((1, N[trial]+1))).full()
        uw['Qdsj'][trial] = ca.vec(uw['Qds'][trial].to_numpy().T*np.ones((1, d*N[trial]))).full()
        lw['Qdsj'][trial] = ca.vec(lw['Qds'][trial].to_numpy().T*np.ones((1, d*N[trial]))).full()    
        if withArms:
            # Arm activations
            uw['ArmA'][trial], lw['ArmA'][trial], scaling['ArmA'][trial] = bounds.getBoundsArmActivation(armJoints)
            uw['ArmAk'][trial] = ca.vec(uw['ArmA'][trial].to_numpy().T * np.ones((1, N[trial]+1))).full()
            lw['ArmAk'][trial] = ca.vec(lw['ArmA'][trial].to_numpy().T * np.ones((1, N[trial]+1))).full()
            uw['ArmAj'][trial] = ca.vec(uw['ArmA'][trial].to_numpy().T * np.ones((1, d*N[trial]))).full()
            lw['ArmAj'][trial] = ca.vec(lw['ArmA'][trial].to_numpy().T * np.ones((1, d*N[trial]))).full()
        if withActiveMTP:
            # MTP activations
            uw['MtpA'][trial], lw['MtpA'][trial], scaling['MtpA'][trial] = bounds.getBoundsMtpActivation(mtpJoints)
            uw['MtpAk'][trial] = ca.vec(uw['MtpA'][trial].to_numpy().T * np.ones((1, N[trial]+1))).full()
            lw['MtpAk'][trial] = ca.vec(lw['MtpA'][trial].to_numpy().T * np.ones((1, N[trial]+1))).full()
            uw['MtpAj'][trial] = ca.vec(uw['MtpA'][trial].to_numpy().T * np.ones((1, d*N[trial]))).full()
            lw['MtpAj'][trial] = ca.vec(lw['MtpA'][trial].to_numpy().T * np.ones((1, d*N[trial]))).full()
        if withLumbarCoordinateActuators:
            # Lumbar activations
            uw['LumbarA'][trial], lw['LumbarA'][trial], scaling['LumbarA'][trial] = bounds.getBoundsLumbarActivation(lumbarJoints)
            uw['LumbarAk'][trial] = ca.vec(uw['LumbarA'][trial].to_numpy().T * np.ones((1, N[trial]+1))).full()
            lw['LumbarAk'][trial] = ca.vec(lw['LumbarA'][trial].to_numpy().T * np.ones((1, N[trial]+1))).full()
            uw['LumbarAj'][trial] = ca.vec(uw['LumbarA'][trial].to_numpy().T * np.ones((1, d*N[trial]))).full()
            lw['LumbarAj'][trial] = ca.vec(lw['LumbarA'][trial].to_numpy().T * np.ones((1, d*N[trial]))).full()
        
        ###########################################################################
        # Controls
        ###########################################################################
        # Muscle activation derivative
        uw['ADt'][trial], lw['ADt'][trial], scaling['ADt'][trial] = bounds.getBoundsActivationDerivative()
        uw['ADtk'][trial] = ca.vec(uw['ADt'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
        lw['ADtk'][trial] = ca.vec(lw['ADt'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
        if withArms:
            # Arm excitations
            uw['ArmE'][trial], lw['ArmE'][trial], scaling['ArmE'][trial] = bounds.getBoundsArmExcitation(armJoints)
            uw['ArmEk'][trial] = ca.vec(uw['ArmE'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
            lw['ArmEk'][trial] = ca.vec(lw['ArmE'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
        if withActiveMTP:
            # MTP excitations
            uw['MtpE'][trial], lw['MtpE'][trial], scaling['MtpE'][trial] = bounds.getBoundsMtpExcitation(mtpJoints)
            uw['MtpEk'][trial] = ca.vec(uw['MtpE'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
            lw['MtpEk'][trial] = ca.vec(lw['MtpE'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
        if withLumbarCoordinateActuators:
            # Arm excitations
            uw['LumbarE'][trial], lw['LumbarE'][trial], scaling['LumbarE'][trial] = bounds.getBoundsLumbarExcitation(lumbarJoints)
            uw['LumbarEk'][trial] = ca.vec(uw['LumbarE'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
            lw['LumbarEk'][trial] = ca.vec(lw['LumbarE'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
        # Muscle force derivatives        
        uw['FDt'][trial], lw['FDt'][trial], scaling['FDt'][trial] = bounds.getBoundsForceDerivative()
        uw['FDtk'][trial] = ca.vec(uw['FDt'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
        lw['FDtk'][trial] = ca.vec(lw['FDt'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
        # Joint velocity derivatives (accelerations)
        if filter_bounds_guess_acceleration:
            uw['Qdds'][trial], lw['Qdds'][trial], scaling['Qdds'][trial] = bounds.getBoundsAccelerationFiltered()
        else:
            uw['Qdds'][trial], lw['Qdds'][trial], scaling['Qdds'][trial] = bounds.getBoundsAcceleration()
        uw['Qddsk'][trial] = ca.vec(uw['Qdds'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
        lw['Qddsk'][trial] = ca.vec(lw['Qdds'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
        # Reserve actuators
        if reserveActuators:
            uw['rAct'][trial], lw['rAct'][trial], scaling['rAct'][trial] = {}, {}, {}
            uw['rActk'][trial], lw['rActk'][trial], = {}, {}
            for c_j in reserveActuatorJoints:
                uw['rAct'][trial][c_j], lw['rAct'][trial][c_j], scaling['rAct'][trial][c_j] = bounds.getBoundsReserveActuators(c_j,
                                                                                                                               reserveActuatorJoints[c_j])
                uw['rActk'][trial][c_j] = ca.vec(uw['rAct'][trial][c_j].to_numpy().T * np.ones((1, N[trial]))).full()
                lw['rActk'][trial][c_j] = ca.vec(lw['rAct'][trial][c_j].to_numpy().T * np.ones((1, N[trial]))).full()
                    
        #######################################################################
        # Static parameters
        #######################################################################
        if offset_ty:
            if tracking_data == "coordinates":
                # scaling['Offset'][trial] = scaling['Qs'][trial].iloc[0]["pelvis_ty"]
                # TODO
                scaling['Offset'][trial] = 1.
            elif tracking_data == "markers":
                scaling['Offset'][trial] = (
                    scaling['Marker'].iloc[0][scaling['Marker'].columns[0]])           
            uw['Offset'][trial], lw['Offset'][trial] = bounds.getBoundsOffset(scaling['Offset'][trial])
            uw['Offsetk'][trial] = uw['Offset'][trial].to_numpy()
            lw['Offsetk'][trial] = lw['Offset'][trial].to_numpy()
            
        if optimizeContacts:
            (uw['ContactParameters'][trial], lw['ContactParameters'][trial], 
             scaling['ContactParameters_v'][trial], 
             scaling['ContactParameters_r'][trial]) = (
                 bounds.getBoundsContactParameters(nContactSpheres, 
                                                   parameter_to_optimize))

        if GRF_toTrack:
        # if trackGRF:            
            _, _, scaling['GRF'][trial] = bounds.getBoundsGR(
                GRF_toTrack[trial]['df_interp']['forces']['all'], 
                GRF_toTrack[trial]['headers']['forces']['all'])
        # if trackGRM:
            _, _, scaling['GRM'][trial] = bounds.getBoundsGR(
                GRF_toTrack[trial]['df_interp']['torques_G']['all'], 
                GRF_toTrack[trial]['headers']['torques']['all'])   
    
    # %% Guesses
    if type_guess == "dataDriven":         
        from guessesOpenSimAD import dataDrivenGuess_tracking
    elif type_guess == "quasiRandom": 
        from guessesOpenSimAD import quasiRandomGuess
    # Pre-allocation
    w0 = {}
    w0['A'], w0['Aj'] = {}, {}
    w0['F'], w0['Fj'] = {}, {}
    w0['Qs'], w0['Qsj'] = {}, {}
    w0['Qds'], w0['Qdsj'] = {}, {}
    if withArms:
        w0['ArmA'], w0['ArmAj'], w0['ArmE'] = {}, {}, {}
    if withActiveMTP:
        w0['MtpA'], w0['MtpAj'],w0['MtpE'] = {}, {}, {}
    if withLumbarCoordinateActuators:
        w0['LumbarA'], w0['LumbarAj'], w0['LumbarE'] = {}, {}, {}
    w0['ADt'] = {}
    w0['FDt'] = {}
    w0['Qdds'] = {}
    if offset_ty:
        w0['Offset'] = {}
    if optimizeContacts:
        w0['ContactParameters'] = {}
    if reserveActuators:
        w0['rAct'] = {}        
    # Loop over trials 
    for trial in trials:
        if type_guess == "dataDriven":         
            guess = dataDrivenGuess_tracking(Qs_toTrack[trial], N[trial], d, joints, 
                                             bothSidesMuscles)    
        elif type_guess == "quasiRandom": 
            guess = quasiRandomGuess(N[trial], d, joints, bothSidesMuscles,
                                     timeElapsed, Qs_toTrack[trial])
            
        ###########################################################################
        # States
        ###########################################################################
        # Muscle activations
        w0['A'][trial] = guess.getGuessActivation(scaling['A'][trial])
        w0['Aj'][trial] = guess.getGuessActivationCol()
        # Muscle forces
        w0['F'][trial] = guess.getGuessForce(scaling['F'][trial])
        w0['Fj'][trial] = guess.getGuessForceCol()
        # Joint positions
        w0['Qs'][trial] = guess.getGuessPosition(scaling['Qs'][trial], guess_zeroMTP)
        w0['Qsj'][trial] = guess.getGuessPositionCol()
        # Joint velocities
        w0['Qds'][trial] = guess.getGuessVelocity(scaling['Qds'][trial], guess_zeroMTP)
        w0['Qdsj'][trial] = guess.getGuessVelocityCol()    
        if withArms:
            w0['ArmA'][trial] = guess.getGuessTorqueActuatorActivation(armJoints)   
            w0['ArmAj'][trial] = guess.getGuessTorqueActuatorActivationCol(armJoints)
        if withActiveMTP:
            w0['MtpA'][trial] = guess.getGuessTorqueActuatorActivation(mtpJoints)
            w0['MtpAj'][trial] = guess.getGuessTorqueActuatorActivationCol(mtpJoints)
        if withLumbarCoordinateActuators:
            w0['LumbarA'][trial] = guess.getGuessTorqueActuatorActivation(lumbarJoints)   
            w0['LumbarAj'][trial] = guess.getGuessTorqueActuatorActivationCol(lumbarJoints)
        
        ###########################################################################
        # Controls
        ###########################################################################
        # Muscle activation derivative
        w0['ADt'][trial] = guess.getGuessActivationDerivative(scaling['ADt'][trial])
        if withArms:
            # Arm activations
            w0['ArmE'][trial] = guess.getGuessTorqueActuatorExcitation(armJoints)
        if withActiveMTP:
            # MTP activations
            w0['MtpE'][trial] = guess.getGuessTorqueActuatorExcitation(mtpJoints)
        if withLumbarCoordinateActuators:
            # Lumbar activations
            w0['LumbarE'][trial] = guess.getGuessTorqueActuatorExcitation(lumbarJoints)
        # Muscle force derivatives   
        w0['FDt'][trial] = guess.getGuessForceDerivative(scaling['FDt'][trial])
        # Joint velocity derivatives (accelerations)
        if filter_bounds_guess_acceleration:
            w0['Qdds'][trial] = (
                guess.getGuessAccelerationFiltered(scaling['Qdds'][trial], 
                                                   guess_zeroAcceleration,
                                                   guess_zeroMTP))
        else:
            w0['Qdds'][trial] = (
                guess.getGuessAcceleration(scaling['Qdds'][trial], 
                                           guess_zeroAcceleration,
                                           guess_zeroMTP))
        # Reserve actuators
        if reserveActuators:
            w0['rAct'][trial] = {}
            for c_j in reserveActuatorJoints:
                w0['rAct'][trial][c_j] = guess.getGuessReserveActuators(c_j)            
            
        ###########################################################################
        # Static parameters
        ###########################################################################
        if offset_ty:
            w0['Offset'][trial] = guess.getGuessOffset(scaling['Offset'][trial])
            
        if optimizeContacts:
            from guessesOpenSimAD import contactParameterGuess
            guessParameters = contactParameterGuess()
            w0['ContactParameters'][trial] = (
                guessParameters.getGuessContactParameters(
                    nContactSpheres, parameter_to_optimize, 
                    scaling['ContactParameters_v'][trial], scaling['ContactParameters_r'][trial]))
            
    # %% Tracking data
    if tracking_data == "coordinates":
        from utilsOpenSimAD import scaleDataFrame, selectFromDataFrame
        
        dataToTrack_sc = {}
        dataToTrack_nsc = {}
        dataToTrack_dot_nsc = {}
        dataToTrack_dot_sc = {}
        dataToTrack_dotdot_nsc = {}
        dataToTrack_dotdot_sc = {}
        refData_dot_nsc = {}
        refData_dotdot_nsc = {}
        for trial in trials:
            dataToTrack_sc[trial] = scaleDataFrame(
                Qs_toTrack[trial], scaling['Qs'][trial], 
                coordinates_toTrack_list).to_numpy()[:,1::].T
             
            dataToTrack_nsc[trial] = selectFromDataFrame(
                Qs_toTrack[trial], 
                coordinates_toTrack_list).to_numpy()[:,1::].T
        
            Qs_spline = Qs_toTrack_sel[trial].copy()
            Qds_spline = Qs_toTrack_sel[trial].copy()
            Qdds_spline = Qs_toTrack_sel[trial].copy()
            for joint in joints:
                spline = interpolate.InterpolatedUnivariateSpline(
                    Qs_toTrack_sel[trial]['time'], 
                    Qs_toTrack_sel[trial][joint], k=3)
                Qs_spline[joint] = spline(
                    Qs_toTrack_sel[trial]['time'])
                splineD1 = spline.derivative(n=1)
                Qds_spline[joint] = splineD1(
                    Qs_toTrack_sel[trial]['time'])                
                
                splineD2 = spline.derivative(n=2)
                Qdds_spline[joint] = splineD2(
                    Qs_toTrack_sel[trial]['time'])
                
            # Filtering
            if filter_Qds_toTracks[trial]:
                Qds_spline_filter = filterDataFrame(
                    Qds_spline, cutoff_frequency=cutoff_freq_Qds[trial])
            else:
                Qds_spline_filter = Qds_spline
                
            if filter_Qdds_toTracks[trial]:
                Qdds_spline_filter = filterDataFrame(
                    Qdds_spline, cutoff_frequency=cutoff_freq_Qdds[trial])
            else:
                Qdds_spline_filter = Qdds_spline
                
            # Instead of splining Qs twice to get Qdds, spline Qds, which can
            # be filtered or not.
            if splineQds[trial]:
                Qdds_spline2 = Qs_toTrack_sel[trial].copy()
                for joint in joints:
                    spline = interpolate.InterpolatedUnivariateSpline(
                        Qds_spline_filter['time'], 
                        Qds_spline_filter[joint], k=3)
                    splineD1 = spline.derivative(n=1)
                    Qdds_spline2[joint] = splineD1(
                        Qds_spline_filter['time'])
                    
                if filter_Qdds_toTracks[trial]:
                    Qdds_spline_filter = filterDataFrame(
                        Qdds_spline2, cutoff_frequency=cutoff_freq_Qdds[trial])
                else:
                    Qdds_spline_filter = Qdds_spline2
                
            # Interpolation
            Qds_spline_interp = interpolateDataFrame(
                Qds_spline_filter, timeIntervals[trial][0], 
                timeIntervals[trial][1], N[trial])
            Qdds_spline_interp = interpolateDataFrame(
                Qdds_spline_filter, timeIntervals[trial][0], 
                timeIntervals[trial][1], N[trial])
                
            dataToTrack_dot_nsc[trial] = selectFromDataFrame(
                Qds_spline_interp, 
                coordinates_toTrack_list).to_numpy()[:,1::].T
            dataToTrack_dot_sc[trial] = scaleDataFrame(
                Qds_spline_interp, scaling['Qds'][trial], 
                coordinates_toTrack_list).to_numpy()[:,1::].T
            
            dataToTrack_dotdot_nsc[trial] = selectFromDataFrame(
                Qdds_spline_interp, 
                coordinates_toTrack_list).to_numpy()[:,1::].T
            dataToTrack_dotdot_sc[trial] = scaleDataFrame(
                Qdds_spline_interp, scaling['Qdds'][trial], 
                coordinates_toTrack_list).to_numpy()[:,1::].T
            
            refData_dot_nsc[trial] = selectFromDataFrame(
                Qds_spline_interp, 
                joints).to_numpy()[:,1::].T            
            refData_dotdot_nsc[trial] = selectFromDataFrame(
                Qdds_spline_interp, 
                joints).to_numpy()[:,1::].T
            
        # Scanity check: ensure that the Qs to track are within the bounds
        # used to define the polynomials.
        from utilsOpenSimAD import checkQsWithinPolynomialBounds
        successCheckPoly = checkQsWithinPolynomialBounds(dataToTrack_nsc, 
                                      polynomial_bounds,
                                      coordinates_toTrack_list,
                                      trials)
        # Exit if problem with poly
        if not successCheckPoly:
            print('Problem poly range')
            return
        
        # if 'translational' in coordinates_toTrack:            
        #     dataToTrack_tr_sc = scaleDataFrame(
        #         Qs_toTrack[trial], scaling['Qs'][trial], 
        #         coordinates_toTrack_tr).to_numpy()[:,1::].T
        #     dataToTrack_tr_nsc = selectFromDataFarme(
        #         Qs_toTrack[trial], 
        #         coordinates_toTrack_tr).to_numpy()[:,1::].T
        
    if GRF_toTrack:
        grfToTrack_sc = {}
        grfToTrack_nsc = {}
        for trial in trials:
            grfToTrack_sc[trial] = scaleDataFrame(
                GRF_toTrack[trial]['df_interp']['forces']['all'], scaling['GRF'][trial], 
                GRF_toTrack[trial]['headers']['forces']['all']).to_numpy()[:,1::].T
            grfToTrack_nsc[trial] = selectFromDataFrame(
                GRF_toTrack[trial]['df_interp']['forces']['all'], 
                GRF_toTrack[trial]['headers']['forces']['all']).to_numpy()[:,1::].T
            
        grmToTrack_sc = {}
        grmToTrack_nsc = {}
        for trial in trials:            
            grmToTrack_sc[trial] = scaleDataFrame(
                GRF_toTrack[trial]['df_interp']['torques_G']['all'], scaling['GRM'][trial], 
                GRF_toTrack[trial]['headers']['torques']['all']).to_numpy()[:,1::].T
            grmToTrack_nsc[trial] = selectFromDataFrame(
                GRF_toTrack[trial]['df_interp']['torques_G']['all'], 
                GRF_toTrack[trial]['headers']['torques']['all']).to_numpy()[:,1::].T
            
    # %% Update bounds if enveloppe constraints
    if coordinate_constraints:
        from utilsOpenSimAD import getColfromk
        if 'pelvis_ty' in coordinate_constraints:
            pelvis_ty_sc = {}
        for trial in trials:
            ubQsk_vec = uw['Qs'][trial].to_numpy().T * np.ones((1, N[trial]+1))
            lbQsk_vec = lw['Qs'][trial].to_numpy().T * np.ones((1, N[trial]+1))            
            ubQsj_vec = uw['Qs'][trial].to_numpy().T * np.ones((1, d*N[trial]))
            lbQsj_vec = lw['Qs'][trial].to_numpy().T * np.ones((1, d*N[trial]))           
            for cons in coordinate_constraints:            
                if cons == 'pelvis_ty':
                    pelvis_ty_sc[trial] = scaleDataFrame(Qs_toTrack[trial], scaling['Qs'][trial], [cons]).to_numpy()[:,1::].T
                    # If there is an offset as part of the design variables,
                    # the constraint is handled as a constraint and not as a
                    # bound.
                    if not offset_ty:
                        ubQsk_vec[joints.index(cons),:-1] = coordinate_constraints[cons]['env_bound'] / scaling['Qs'][trial].iloc[0][cons] + pelvis_ty_sc[trial]        
                        lbQsk_vec[joints.index(cons),:-1] = -coordinate_constraints[cons]['env_bound'] / scaling['Qs'][trial].iloc[0][cons] + pelvis_ty_sc[trial]                    
                        c_sc_j = getColfromk(pelvis_ty_sc[trial], d, N[trial])
                        ubQsj_vec[joints.index(cons),:] = coordinate_constraints[cons]['env_bound'] / scaling['Qs'][trial].iloc[0][cons] + c_sc_j
                        lbQsj_vec[joints.index(cons),:] = -coordinate_constraints[cons]['env_bound'] / scaling['Qs'][trial].iloc[0][cons] + c_sc_j                        
                else:            
                    c_sc = scaleDataFrame(Qs_toTrack[trial], scaling['Qs'][trial], [cons]).to_numpy()[:,1::].T                
                    ubQsk_vec[joints.index(cons),:-1] = coordinate_constraints[cons]['env_bound'] / scaling['Qs'][trial].iloc[0][cons] + c_sc        
                    lbQsk_vec[joints.index(cons),:-1] = -coordinate_constraints[cons]['env_bound'] / scaling['Qs'][trial].iloc[0][cons] + c_sc               
                    # c_sc_j = getColfromk(c_sc, d, N[trial])
                    # ubQsj_vec[joints.index(cons),:] = coordinate_constraints[cons]['env_bound'] / scaling['Qs'][trial].iloc[0][cons] + c_sc_j
                    # lbQsj_vec[joints.index(cons),:] = -coordinate_constraints[cons]['env_bound'] / scaling['Qs'][trial].iloc[0][cons] + c_sc_j
            uw['Qsk'][trial] = ca.vec(ubQsk_vec).full()
            lw['Qsk'][trial] = ca.vec(lbQsk_vec).full()
            # uw['Qsj'][trial] = ca.vec(ubQsj_vec).full()
            # lw['Qsj'][trial] = ca.vec(lbQsj_vec).full()
        
    # %% Formulate OCP.
    # Collocation matrices.
    tau = ca.collocation_points(d,'radau')
    [C,D] = ca.collocation_interpolators(tau)
    # Missing matrix B, add manually.
    if d == 3:  
        B = [0, 0.376403062700467, 0.512485826188421, 0.111111111111111]
    elif d == 2:
        B = [0, 0.75, 0.25]
    
    if solveProblem:            
        # Initialize cost function.
        J = 0            
        # Initialize opti instance.
        opti = ca.Opti()            
        # Static parameters.
        # if optimizeContacts:
        #     p = opti.variable(NContactParameters)
        #     opti.subject_to(opti.bounded(lw['ContactParameters'][trial].T, p,
        #                                  uw['ContactParameters'][trial].T))
        #     opti.set_initial(p, w0['ContactParameters'][trial])
        #     # assert np.alltrue(lw['ContactParameters'][trial] <= w0['ContactParameters'][trial]), "lw contact parameters"
        #     # assert np.alltrue(uw['ContactParameters'][trial] >= w0['ContactParameters'][trial]), "uw contact parameters"
        if offset_ty:
            # Offset pelvis_ty.
            offset = opti.variable(1)
            opti.subject_to(opti.bounded(lw['Offsetk'][trial], offset,
                                         uw['Offsetk'][trial]))
            opti.set_initial(offset, w0['Offset'][trial])
        else:
            offset = 0            
        # Initialize variables.    
        a, a_col, nF, nF_col = {}, {}, {}, {}
        Qs, Qs_col, Qds, Qds_col = {}, {}, {}, {}
        if withArms:
            aArm, aArm_col, eArm = {}, {}, {}
        if withActiveMTP:
            aMtp, aMtp_col, eMtp = {}, {}, {}
        if withLumbarCoordinateActuators:
            aLumbar, aLumbar_col, eLumbar = {}, {}, {}            
        aDt, nFDt, Qdds = {}, {}, {}
        if reserveActuators:
            rAct = {}        
        # Loop over trials.
        for trial in trials:        
            # Time step    
            h = timeElapsed[trial] / N[trial]
            # States
            # Muscle activation at mesh points
            a[trial] = opti.variable(NMuscles, N[trial]+1)
            opti.subject_to(opti.bounded(lw['Ak'][trial], ca.vec(a[trial]), uw['Ak'][trial]))
            opti.set_initial(a[trial], w0['A'][trial].to_numpy().T)
            assert np.alltrue(lw['Ak'][trial] <= ca.vec(w0['A'][trial].to_numpy().T).full()), "lw Muscle activation"
            assert np.alltrue(uw['Ak'][trial] >= ca.vec(w0['A'][trial].to_numpy().T).full()), "uw Muscle activation"
            # Muscle activation at collocation points
            a_col[trial] = opti.variable(NMuscles, d*N[trial])
            opti.subject_to(opti.bounded(lw['Aj'][trial], ca.vec(a_col[trial]), uw['Aj'][trial]))
            opti.set_initial(a_col[trial], w0['Aj'][trial].to_numpy().T)
            assert np.alltrue(lw['Aj'][trial] <= ca.vec(w0['Aj'][trial].to_numpy().T).full()), "lw Muscle activation col"
            assert np.alltrue(uw['Aj'][trial] >= ca.vec(w0['Aj'][trial].to_numpy().T).full()), "uw Muscle activation col"
            # Muscle force at mesh points
            nF[trial] = opti.variable(NMuscles, N[trial]+1)
            opti.subject_to(opti.bounded(lw['Fk'][trial], ca.vec(nF[trial]), uw['Fk'][trial]))
            opti.set_initial(nF[trial], w0['F'][trial].to_numpy().T)
            assert np.alltrue(lw['Fk'][trial] <= ca.vec(w0['F'][trial].to_numpy().T).full()), "lw Muscle force"
            assert np.alltrue(uw['Fk'][trial] >= ca.vec(w0['F'][trial].to_numpy().T).full()), "uw Muscle force"
            # Muscle force at collocation points
            nF_col[trial] = opti.variable(NMuscles, d*N[trial])
            opti.subject_to(opti.bounded(lw['Fj'][trial], ca.vec(nF_col[trial]), uw['Fj'][trial]))
            opti.set_initial(nF_col[trial], w0['Fj'][trial].to_numpy().T)
            assert np.alltrue(lw['Fj'][trial] <= ca.vec(w0['Fj'][trial].to_numpy().T).full()), "lw Muscle force col"
            assert np.alltrue(uw['Fj'][trial] >= ca.vec(w0['Fj'][trial].to_numpy().T).full()), "uw Muscle force col"
            # Joint position at mesh points
            Qs[trial] = opti.variable(nJoints, N[trial]+1)
            opti.subject_to(opti.bounded(lw['Qsk'][trial], ca.vec(Qs[trial]), uw['Qsk'][trial]))
            guessQsEnd = np.concatenate(
                (w0['Qs'][trial].to_numpy().T, np.reshape(
                    w0['Qs'][trial].to_numpy().T[:,-1], 
                    (w0['Qs'][trial].to_numpy().T.shape[0], 1))), axis=1)
            opti.set_initial(Qs[trial], guessQsEnd)
            # TODO 5 is for s11
            assert np.alltrue(lw['Qsk'][trial] - 5*np.pi/180 <= ca.vec(guessQsEnd).full()), "lw Joint position"
            assert np.alltrue(uw['Qsk'][trial] + 5*np.pi/180 >= ca.vec(guessQsEnd).full()), "uw Joint position"
            # Joint position at collocation points
            Qs_col[trial] = opti.variable(nJoints, d*N[trial])
            opti.subject_to(opti.bounded(lw['Qsj'][trial], ca.vec(Qs_col[trial]), uw['Qsj'][trial]))
            opti.set_initial(Qs_col[trial], w0['Qsj'][trial].to_numpy().T)
            # Small margin to account for filtering.
            assert np.alltrue(lw['Qsj'][trial] - 5*np.pi/180 <= ca.vec(w0['Qsj'][trial].to_numpy().T).full()), "lw Joint position col"
            assert np.alltrue(uw['Qsj'][trial] + 5*np.pi/180 >= ca.vec(w0['Qsj'][trial].to_numpy().T).full()), "uw Joint position col"
            # Joint velocity at mesh points
            Qds[trial] = opti.variable(nJoints, N[trial]+1)
            opti.subject_to(opti.bounded(lw['Qdsk'][trial], ca.vec(Qds[trial]), uw['Qdsk'][trial]))
            guessQdsEnd = np.concatenate(
                (w0['Qds'][trial].to_numpy().T, np.reshape(
                    w0['Qds'][trial].to_numpy().T[:,-1], 
                    (w0['Qds'][trial].to_numpy().T.shape[0], 1))), axis=1)
            opti.set_initial(Qds[trial], guessQdsEnd)
            assert np.alltrue(lw['Qdsk'][trial] <= ca.vec(guessQdsEnd).full()), "lw Joint velocity"
            assert np.alltrue(uw['Qdsk'][trial] >= ca.vec(guessQdsEnd).full()), "uw Joint velocity"        
            # Joint velocity at collocation points
            Qds_col[trial] = opti.variable(nJoints, d*N[trial])
            opti.subject_to(opti.bounded(lw['Qdsj'][trial], ca.vec(Qds_col[trial]), uw['Qdsj'][trial]))
            opti.set_initial(Qds_col[trial], w0['Qdsj'][trial].to_numpy().T)
            assert np.alltrue(lw['Qdsj'][trial] <= ca.vec(w0['Qdsj'][trial].to_numpy().T).full()), "lw Joint velocity col"
            assert np.alltrue(uw['Qdsj'][trial] >= ca.vec(w0['Qdsj'][trial].to_numpy().T).full()), "uw Joint velocity col"
            if withArms:
                # Arm activation at mesh points
                aArm[trial] = opti.variable(nArmJoints, N[trial]+1)
                opti.subject_to(opti.bounded(lw['ArmAk'][trial], ca.vec(aArm[trial]), uw['ArmAk'][trial]))
                opti.set_initial(aArm[trial], w0['ArmA'][trial].to_numpy().T)
                assert np.alltrue(lw['ArmAk'][trial] <= ca.vec(w0['ArmA'][trial].to_numpy().T).full()), "lw Arm activation"
                assert np.alltrue(uw['ArmAk'][trial] >= ca.vec(w0['ArmA'][trial].to_numpy().T).full()), "uw Arm activation"
                # Arm activation at collocation points
                aArm_col[trial] = opti.variable(nArmJoints, d*N[trial])
                opti.subject_to(opti.bounded(lw['ArmAj'][trial], ca.vec(aArm_col[trial]), uw['ArmAj'][trial]))
                opti.set_initial(aArm_col[trial], w0['ArmAj'][trial].to_numpy().T)
                assert np.alltrue(lw['ArmAj'][trial] <= ca.vec(w0['ArmAj'][trial].to_numpy().T).full()), "lw Arm activation col"
                assert np.alltrue(uw['ArmAj'][trial] >= ca.vec(w0['ArmAj'][trial].to_numpy().T).full()), "uw Arm activation col"
            if withActiveMTP:
                # Mtp activation at mesh points
                aMtp[trial] = opti.variable(nMtpJoints, N[trial]+1)
                opti.subject_to(opti.bounded(lw['MtpAk'][trial], ca.vec(aMtp[trial]), uw['MtpAk'][trial]))
                opti.set_initial(aMtp[trial], w0['MtpA'][trial].to_numpy().T)
                assert np.alltrue(lw['MtpAk'][trial] <= ca.vec(w0['MtpA'][trial].to_numpy().T).full()), "lw Mtp activation"
                assert np.alltrue(uw['MtpAk'][trial] >= ca.vec(w0['MtpA'][trial].to_numpy().T).full()), "uw Mtp activation"
                # Mtp activation at collocation points
                aMtp_col[trial] = opti.variable(nMtpJoints, d*N[trial])
                opti.subject_to(opti.bounded(lw['MtpAj'][trial], ca.vec(aMtp_col[trial]), uw['MtpAj'][trial]))
                opti.set_initial(aMtp_col[trial], w0['MtpAj'][trial].to_numpy().T)
                assert np.alltrue(lw['MtpAj'][trial] <= ca.vec(w0['MtpAj'][trial].to_numpy().T).full()), "lw Mtp activation col"
                assert np.alltrue(uw['MtpAj'][trial] >= ca.vec(w0['MtpAj'][trial].to_numpy().T).full()), "uw Mtp activation col"
            if withLumbarCoordinateActuators:
                # Lumbar activation at mesh points
                aLumbar[trial] = opti.variable(nLumbarJoints, N[trial]+1)
                opti.subject_to(opti.bounded(lw['LumbarAk'][trial], ca.vec(aLumbar[trial]), uw['LumbarAk'][trial]))
                opti.set_initial(aLumbar[trial], w0['LumbarA'][trial].to_numpy().T)
                assert np.alltrue(lw['LumbarAk'][trial] <= ca.vec(w0['LumbarA'][trial].to_numpy().T).full()), "lw Lumbar activation"
                assert np.alltrue(uw['LumbarAk'][trial] >= ca.vec(w0['LumbarA'][trial].to_numpy().T).full()), "uw Lumbar activation"
                # Lumbar activation at collocation points
                aLumbar_col[trial] = opti.variable(nLumbarJoints, d*N[trial])
                opti.subject_to(opti.bounded(lw['LumbarAj'][trial], ca.vec(aLumbar_col[trial]), uw['LumbarAj'][trial]))
                opti.set_initial(aLumbar_col[trial], w0['LumbarAj'][trial].to_numpy().T)
                assert np.alltrue(lw['LumbarAj'][trial] <= ca.vec(w0['LumbarAj'][trial].to_numpy().T).full()), "lw Lumbar activation col"
                assert np.alltrue(uw['LumbarAj'][trial] >= ca.vec(w0['LumbarAj'][trial].to_numpy().T).full()), "uw Lumbar activation col"
            # Controls
            # Muscle activation derivative at mesh points
            aDt[trial] = opti.variable(NMuscles, N[trial])
            opti.subject_to(opti.bounded(lw['ADtk'][trial], ca.vec(aDt[trial]), uw['ADtk'][trial]))
            opti.set_initial(aDt[trial], w0['ADt'][trial].to_numpy().T)
            assert np.alltrue(lw['ADtk'][trial] <= ca.vec(w0['ADt'][trial].to_numpy().T).full()), "lw Muscle activation derivative"
            assert np.alltrue(uw['ADtk'][trial] >= ca.vec(w0['ADt'][trial].to_numpy().T).full()), "uw Muscle activation derivative"
            if withArms:
                # Arm excitation at mesh points
                eArm[trial] = opti.variable(nArmJoints, N[trial])
                opti.subject_to(opti.bounded(lw['ArmEk'][trial], ca.vec(eArm[trial]), uw['ArmEk'][trial]))
                opti.set_initial(eArm[trial], w0['ArmE'][trial].to_numpy().T)
                assert np.alltrue(lw['ArmEk'][trial] <= ca.vec(w0['ArmE'][trial].to_numpy().T).full()), "lw Arm excitation"
                assert np.alltrue(uw['ArmEk'][trial] >= ca.vec(w0['ArmE'][trial].to_numpy().T).full()), "uw Arm excitation"
            if withActiveMTP:
                # Mtp excitation at mesh points
                eMtp[trial] = opti.variable(nMtpJoints, N[trial])
                opti.subject_to(opti.bounded(lw['MtpEk'][trial], ca.vec(eMtp[trial]), uw['MtpEk'][trial]))
                opti.set_initial(eMtp[trial], w0['MtpE'][trial].to_numpy().T)
                assert np.alltrue(lw['MtpEk'][trial] <= ca.vec(w0['MtpE'][trial].to_numpy().T).full()), "lw Mtp excitation"
                assert np.alltrue(uw['MtpEk'][trial] >= ca.vec(w0['MtpE'][trial].to_numpy().T).full()), "uw Mtp excitation"
            if withLumbarCoordinateActuators:
                # Lumbar excitation at mesh points
                eLumbar[trial] = opti.variable(nLumbarJoints, N[trial])
                opti.subject_to(opti.bounded(lw['LumbarEk'][trial], ca.vec(eLumbar[trial]), uw['LumbarEk'][trial]))
                opti.set_initial(eLumbar[trial], w0['LumbarE'][trial].to_numpy().T)
                assert np.alltrue(lw['LumbarEk'][trial] <= ca.vec(w0['LumbarE'][trial].to_numpy().T).full()), "lw Lumbar excitation"
                assert np.alltrue(uw['LumbarEk'][trial] >= ca.vec(w0['LumbarE'][trial].to_numpy().T).full()), "uw Lumbar excitation"
            # Muscle force derivative at mesh points
            nFDt[trial] = opti.variable(NMuscles, N[trial])
            opti.subject_to(opti.bounded(lw['FDtk'][trial], ca.vec(nFDt[trial]), uw['FDtk'][trial]))
            opti.set_initial(nFDt[trial], w0['FDt'][trial].to_numpy().T)
            assert np.alltrue(lw['FDtk'][trial] <= ca.vec(w0['FDt'][trial].to_numpy().T).full()), "lw Muscle force derivative"
            assert np.alltrue(uw['FDtk'][trial] >= ca.vec(w0['FDt'][trial].to_numpy().T).full()), "uw Muscle force derivative"
            # Joint velocity derivative (acceleration) at mesh points
            Qdds[trial] = opti.variable(nJoints, N[trial])
            opti.subject_to(opti.bounded(lw['Qddsk'][trial], ca.vec(Qdds[trial]), uw['Qddsk'][trial]))
            opti.set_initial(Qdds[trial], w0['Qdds'][trial].to_numpy().T)
            assert np.alltrue(lw['Qddsk'][trial] <= ca.vec(w0['Qdds'][trial].to_numpy().T).full()), "lw Joint velocity derivative"
            assert np.alltrue(uw['Qddsk'][trial] >= ca.vec(w0['Qdds'][trial].to_numpy().T).full()), "uw Joint velocity derivative"
            # Reserve actuator at mesh points
            if reserveActuators:
                rAct[trial] = {}
                for c_j in reserveActuatorJoints:                    
                    rAct[trial][c_j] = opti.variable(1, N[trial])
                    opti.subject_to(opti.bounded(lw['rActk'][trial][c_j], ca.vec(rAct[trial][c_j]), uw['rActk'][trial][c_j]))
                    opti.set_initial(rAct[trial][c_j], w0['rAct'][trial][c_j].to_numpy().T)
                    assert np.alltrue(lw['rActk'][trial][c_j] <= ca.vec(w0['rAct'][trial][c_j].to_numpy().T).full()), "lw reserve"
                    assert np.alltrue(uw['rActk'][trial][c_j] >= ca.vec(w0['rAct'][trial][c_j].to_numpy().T).full()), "uw reserve"
                
            # %% Plots guess vs bounds.
            if plotGuessVsBounds:            
                # States
                # Muscle activation at mesh points
                from utilsOpenSimAD import plotVSBounds
                from utilsOpenSimAD import plotVSvaryingBounds
                lwp = lw['A'][trial].to_numpy().T
                uwp = uw['A'][trial].to_numpy().T
                y = w0['A'][trial].to_numpy().T
                title='Muscle activation at mesh points'            
                plotVSBounds(y,lwp,uwp,title)  
                # Muscle activation at collocation points
                lwp = lw['A'][trial].to_numpy().T
                uwp = uw['A'][trial].to_numpy().T
                y = w0['Aj'][trial].to_numpy().T
                title='Muscle activation at collocation points' 
                plotVSBounds(y,lwp,uwp,title)  
                # Muscle force at mesh points
                lwp = lw['F'][trial].to_numpy().T
                uwp = uw['F'][trial].to_numpy().T
                y = w0['F'][trial].to_numpy().T
                title='Muscle force at mesh points' 
                plotVSBounds(y,lwp,uwp,title)  
                # Muscle force at collocation points
                lwp = lw['F'][trial].to_numpy().T
                uwp = uw['F'][trial].to_numpy().T
                y = w0['Fj'][trial].to_numpy().T
                title='Muscle force at collocation points' 
                plotVSBounds(y,lwp,uwp,title)
                # Joint position at mesh points
                lwp = np.reshape(
                    lw['Qsk'][trial], (nJoints, N[trial]+1), order='F')
                uwp = np.reshape(
                    uw['Qsk'][trial], (nJoints, N[trial]+1), order='F')
                y = guessQsEnd
                title='Joint position at mesh points' 
                plotVSvaryingBounds(y,lwp,uwp,title)             
                # Joint position at collocation points
                lwp = np.reshape(
                    lw['Qsj'][trial], (nJoints, d*N[trial]), order='F')
                uwp = np.reshape(
                    uw['Qsj'][trial], (nJoints, d*N[trial]), order='F')
                y = w0['Qsj'][trial].to_numpy().T
                title='Joint position at collocation points' 
                plotVSvaryingBounds(y,lwp,uwp,title) 
                # Joint velocity at mesh points
                lwp = lw['Qds'][trial].to_numpy().T
                uwp = uw['Qds'][trial].to_numpy().T
                y = guessQdsEnd
                title='Joint velocity at mesh points' 
                plotVSBounds(y,lwp,uwp,title) 
                # Joint velocity at collocation points
                lwp = lw['Qds'][trial].to_numpy().T
                uwp = uw['Qds'][trial].to_numpy().T
                y = w0['Qdsj'][trial].to_numpy().T
                title='Joint velocity at collocation points' 
                plotVSBounds(y,lwp,uwp,title) 
                if withArms:
                    # Arm activation at mesh points
                    lwp = lw['ArmA'][trial].to_numpy().T
                    uwp = uw['ArmA'][trial].to_numpy().T
                    y = w0['ArmA'][trial].to_numpy().T
                    title='Arm activation at mesh points' 
                    plotVSBounds(y,lwp,uwp,title) 
                    # Arm activation at collocation points
                    lwp = lw['ArmA'][trial].to_numpy().T
                    uwp = uw['ArmA'][trial].to_numpy().T
                    y = w0['ArmAj'][trial].to_numpy().T
                    title='Arm activation at collocation points' 
                    plotVSBounds(y,lwp,uwp,title) 
                if withActiveMTP:
                    # Mtp activation at mesh points
                    lwp = lw['MtpA'][trial].to_numpy().T
                    uwp = uw['MtpA'][trial].to_numpy().T
                    y = w0['MtpA'][trial].to_numpy().T
                    title='Mtp activation at mesh points' 
                    plotVSBounds(y,lwp,uwp,title) 
                    # Mtp activation at collocation points
                    lwp = lw['MtpA'][trial].to_numpy().T
                    uwp = uw['MtpA'][trial].to_numpy().T
                    y = w0['MtpAj'][trial].to_numpy().T
                    title='Mtp activation at collocation points' 
                    plotVSBounds(y,lwp,uwp,title)
                if withLumbarCoordinateActuators:
                    # Lumbar activation at mesh points
                    lwp = lw['LumbarA'][trial].to_numpy().T
                    uwp = uw['LumbarA'][trial].to_numpy().T
                    y = w0['LumbarA'][trial].to_numpy().T
                    title='Lumbar activation at mesh points' 
                    plotVSBounds(y,lwp,uwp,title) 
                    # Lumbar activation at collocation points
                    lwp = lw['LumbarA'][trial].to_numpy().T
                    uwp = uw['LumbarA'][trial].to_numpy().T
                    y = w0['LumbarAj'][trial].to_numpy().T
                    title='Lumbar activation at collocation points' 
                    plotVSBounds(y,lwp,uwp,title)
                # Controls
                # Muscle activation derivative at mesh points
                lwp = lw['ADt'][trial].to_numpy().T
                uwp = uw['ADt'][trial].to_numpy().T
                y = w0['ADt'][trial].to_numpy().T
                title='Muscle activation derivative at mesh points' 
                plotVSBounds(y,lwp,uwp,title) 
                if withArms:
                    # Arm excitation at mesh points
                    lwp = lw['ArmE'][trial].to_numpy().T
                    uwp = uw['ArmE'][trial].to_numpy().T
                    y = w0['ArmE'][trial].to_numpy().T
                    title='Arm excitation at mesh points' 
                    plotVSBounds(y,lwp,uwp,title) 
                if withActiveMTP:
                    # Mtp excitation at mesh points
                    lwp = lw['MtpE'][trial].to_numpy().T
                    uwp = uw['MtpE'][trial].to_numpy().T
                    y = w0['MtpE'][trial].to_numpy().T
                    title='Mtp excitation at mesh points' 
                    plotVSBounds(y,lwp,uwp,title)
                if withLumbarCoordinateActuators:
                    # Lumbar excitation at mesh points
                    lwp = lw['LumbarE'][trial].to_numpy().T
                    uwp = uw['LumbarE'][trial].to_numpy().T
                    y = w0['LumbarE'][trial].to_numpy().T
                    title='Lumbar excitation at mesh points' 
                    plotVSBounds(y,lwp,uwp,title)                    
                # Muscle force derivative at mesh points
                lwp = lw['FDt'][trial].to_numpy().T
                uwp = uw['FDt'][trial].to_numpy().T
                y = w0['FDt'][trial].to_numpy().T
                title='Muscle force derivative at mesh points' 
                plotVSBounds(y,lwp,uwp,title)
                # Joint velocity derivative (acceleration) at mesh points
                lwp = lw['Qdds'][trial].to_numpy().T
                uwp = uw['Qdds'][trial].to_numpy().T
                y = w0['Qdds'][trial].to_numpy().T
                title='Joint velocity derivative (acceleration) at mesh points' 
                plotVSBounds(y,lwp,uwp,title)
                
            # %%  Unscale variables.
            nF_nsc = nF[trial] * (scaling['F'][trial].to_numpy().T * np.ones((1, N[trial]+1)))
            nF_col_nsc = nF_col[trial] * (scaling['F'][trial].to_numpy().T * np.ones((1, d*N[trial])))
            Qs_nsc = Qs[trial] * (scaling['Qs'][trial].to_numpy().T * np.ones((1, N[trial]+1)))
            Qs_col_nsc = Qs_col[trial] * (scaling['Qs'][trial].to_numpy().T * np.ones((1, d*N[trial])))
            Qds_nsc = Qds[trial] * (scaling['Qds'][trial].to_numpy().T * np.ones((1, N[trial]+1)))
            Qds_col_nsc = Qds_col[trial] * (scaling['Qds'][trial].to_numpy().T * np.ones((1, d*N[trial])))
            aDt_nsc = aDt[trial] * (scaling['ADt'][trial].to_numpy().T * np.ones((1, N[trial])))
            Qdds_nsc = Qdds[trial] * (scaling['Qdds'][trial].to_numpy().T * np.ones((1, N[trial])))
            nFDt_nsc = nFDt[trial] * (scaling['FDt'][trial].to_numpy().T * np.ones((1, N[trial])))
            if reserveActuators:
                rAct_nsc = {}
                for c_j in reserveActuatorJoints:
                    rAct_nsc[c_j] = rAct[trial][c_j] * (scaling['rAct'][trial][c_j].to_numpy().T * np.ones((1, N[trial])))
            
            # if optimizeContacts:
            #     p_nsc = ((p - scaling['ContactParameters_r'][trial]) / scaling['ContactParameters_v'][trial])
    
            # %% Offset data.
            dataToTrack_nsc_offset = ca.MX(dataToTrack_nsc[trial].shape[0],
                                           dataToTrack_nsc[trial].shape[1])                    
            for j, joint in enumerate(coordinates_toTrack):                        
                if joint == "pelvis_ty":                        
                    dataToTrack_nsc_offset[j, :] = (
                        dataToTrack_nsc[trial][j, :] + offset)
                else:
                    dataToTrack_nsc_offset[j, :] = dataToTrack_nsc[trial][j, :]                
            dataToTrack_sc_offset = (
                dataToTrack_nsc_offset / 
                ((scaling['Qs'][trial].to_numpy().T)[idx_coordinates_toTrack] * 
                  np.ones((1, N[trial]))))
            
            # %%  Main block (loop over mesh points).
            for k in range(N[trial]):
                ###############################################################
                # Variables within current mesh.
                ###############################################################
                # States.
                akj = (ca.horzcat(a[trial][:, k], a_col[trial][:, k*d:(k+1)*d]))
                nFkj = (ca.horzcat(nF[trial][:, k], nF_col[trial][:, k*d:(k+1)*d]))
                nFkj_nsc = (ca.horzcat(nF_nsc[:, k], nF_col_nsc[:, k*d:(k+1)*d]))
                Qskj = (ca.horzcat(Qs[trial][:, k], Qs_col[trial][:, k*d:(k+1)*d]))
                Qskj_nsc = (ca.horzcat(Qs_nsc[:, k], Qs_col_nsc[:, k*d:(k+1)*d]))
                Qdskj = (ca.horzcat(Qds[trial][:, k], Qds_col[trial][:, k*d:(k+1)*d]))    
                Qdskj_nsc = (ca.horzcat(Qds_nsc[:, k], Qds_col_nsc[:, k*d:(k+1)*d]))
                if withArms:
                    aArmkj = (ca.horzcat(aArm[trial][:, k], aArm_col[trial][:, k*d:(k+1)*d]))
                if withLumbarCoordinateActuators:
                    aLumbarkj = (ca.horzcat(aLumbar[trial][:, k], aLumbar_col[trial][:, k*d:(k+1)*d]))
                # Controls.
                aDtk = aDt[trial][:, k]
                aDtk_nsc = aDt_nsc[:, k]
                nFDtk = nFDt[trial][:, k]
                nFDtk_nsc = nFDt_nsc[:, k]
                Qddsk = Qdds[trial][:, k]
                Qddsk_nsc = Qdds_nsc[:, k]
                if withArms:
                    eArmk = eArm[trial][:, k]
                if withLumbarCoordinateActuators:
                    eLumbark = eLumbar[trial][:, k]
                if reserveActuators:
                    rActk = {}
                    rActk_nsc = {}
                    for c_j in reserveActuatorJoints:
                        rActk[c_j] = rAct[trial][c_j][:,k]
                        rActk_nsc[c_j] = rAct_nsc[c_j][:,k]  
                # Qs and Qds are intertwined in external function.
                QsQdskj_nsc = ca.MX(nJoints*2, d+1)
                QsQdskj_nsc[::2, :] = Qskj_nsc[idxJoints4F, :]
                QsQdskj_nsc[1::2, :] = Qdskj_nsc[idxJoints4F, :]         
                
                ###############################################################
                # Polynomial approximations
                ###############################################################                
                # Left side.
                Qsink_l = Qskj_nsc[leftPolynomialJointIndices, 0]
                Qdsink_l = Qdskj_nsc[leftPolynomialJointIndices, 0]
                [lMTk_l, vMTk_l, dMk_l] = f_polynomial['l'](Qsink_l, Qdsink_l) 
                # Right side.
                Qsink_r = Qskj_nsc[rightPolynomialJointIndices, 0]
                Qdsink_r = Qdskj_nsc[rightPolynomialJointIndices, 0]
                [lMTk_r, vMTk_r, dMk_r] = f_polynomial['r'](Qsink_r, Qdsink_r)
                # Muscle-tendon lengths and velocities.      
                lMTk_lr = ca.vertcat(lMTk_l[leftPolynomialMuscleIndices], 
                                     lMTk_r[rightPolynomialMuscleIndices])
                vMTk_lr = ca.vertcat(vMTk_l[leftPolynomialMuscleIndices], 
                                     vMTk_r[rightPolynomialMuscleIndices])
                # Moment arms.
                dMk = {}
                # Left side.
                for joint in leftPolynomialJoints:
                    if ((joint != 'mtp_angle_l') and 
                        (joint != 'lumbar_extension') and
                        (joint != 'lumbar_bending') and 
                        (joint != 'lumbar_rotation')):
                            dMk[joint] = dMk_l[
                                momentArmIndices[joint], 
                                leftPolynomialJoints.index(joint)]
                # Right side.
                for joint in rightPolynomialJoints:
                    if ((joint != 'mtp_angle_r') and 
                        (joint != 'lumbar_extension') and
                        (joint != 'lumbar_bending') and 
                        (joint != 'lumbar_rotation')):
                            # We need to adjust momentArmIndices for the right 
                            # side since polynomial indices are 'one-sided'. 
                            # We subtract by the number of side muscles.
                            c_ma = [i - nSideMuscles for 
                                    i in momentArmIndices[joint]]
                            dMk[joint] = dMk_r[
                                c_ma, rightPolynomialJoints.index(joint)]
                # Trunk.
                if model_type == 'gait2392':
                    for joint in lumbarJoints:
                        dMk[joint] = dMk_l[trunkMomentArmPolynomialIndices, 
                                           leftPolynomialJoints.index(joint)]               
                
                ###############################################################
                # Hill-equilibrium 
                ###############################################################
                [hillEquilibriumk, Fk, activeFiberForcek, passiveFiberForcek,
                 normActiveFiberLengthForcek, nFiberLengthk,
                 fiberVelocityk, _, _] = (f_hillEquilibrium(
                     akj[:, 0], lMTk_lr, vMTk_lr, nFkj_nsc[:, 0],
                     nFDtk_nsc))
                     
                ###############################################################
                # Limit torques
                ###############################################################
                passiveTorque_k = {}
                if enableLimitTorques:                    
                    for joint in passiveTorqueJoints:
                        passiveTorque_k[joint] = f_passiveTorque[joint](
                            Qskj_nsc[joints.index(joint), 0], 
                            Qdskj_nsc[joints.index(joint), 0]) 
                else:
                    for joint in passiveTorqueJoints:
                        passiveTorque_k[joint] = 0
    
                ###############################################################
                # Linear torques
                ###############################################################
                if withMTP:
                    linearPassiveTorqueMtp_k = {}
                    for joint in mtpJoints:
                        linearPassiveTorqueMtp_k[joint] = (
                            f_linearPassiveMtpTorque(
                                Qskj_nsc[joints.index(joint), 0],
                                Qdskj_nsc[joints.index(joint), 0]))                    
                if withArms:
                    linearPassiveTorqueArms_k = {}
                    for joint in armJoints:
                        linearPassiveTorqueArms_k[joint] = (
                            f_linearPassiveArmTorque(
                                Qskj_nsc[joints.index(joint), 0],
                                Qdskj_nsc[joints.index(joint), 0]))
            
                ###############################################################
                # Call external function
                if optimizeContacts:
                    # Commenting for now
                    print("No longer supported")
                    # # Call first external function
                    # T1k = F1_c(ca.vertcat(QsQdskj_nsc[:, 0]))
                    
                    # RBGk, posBk, TBGk, lVelBk, aVelBk = {}, {}, {}, {}, {}
                    # for segment in foot_segments:
                    #     RBG_flat = T1k[idxF_c1[segment]['rotation']]
                    #     RBGk[segment] = ca.reshape(RBG_flat, 3, 3)
                    #     posBk[segment] = T1k[idxF_c1[segment]['position']]
                    #     TBGk[segment] = T1k[idxF_c1[segment]['translation']]
                    #     lVelBk[segment] = T1k[idxF_c1[segment]['linear_velocity']]
                    #     aVelBk[segment] = T1k[idxF_c1[segment]['angular_velocity']]
                    
                    # if nContactSpheres == 6:                   
                    #     if parameter_to_optimize == "option1":                                
                    #         locSphere_inB_p = {}
                    #         radius_p = {}
                    #         appliedFk = {}
                    #         for n_c in range(nContactSpheres):
                    #             c_n = str(n_c+1)
                    #             locSphere_inB_p[c_n] = {}
                    #             locSphere_inB_p[c_n]['r'] = (
                    #                 ca.vertcat(p_nsc[n_c*2], -0.01, 
                    #                             p_nsc[n_c*2+1]))
                    #             locSphere_inB_p[c_n]['l'] = (
                    #                 ca.vertcat(p_nsc[n_c*2], -0.01,
                    #                             -p_nsc[n_c*2+1]))
                    #             radius_p[c_n] = (
                    #                 p_nsc[2*nContactSpheres+n_c])
                    #             appliedFk[c_n] = {}
                    #             for s_c in side_contacts:
                    #                 c_s = contactSegments[n_c] + '_' + s_c                    
                    #                 appliedFk[c_n][s_c] = f_contactForce(
                    #                     dissipation, stiffness, radius_p[c_n],
                    #                     locSphere_inB_p[c_n][s_c], posBk[c_s],
                    #                     lVelBk[c_s], aVelBk[c_s], RBGk[c_s],
                    #                     TBGk[c_s])                                    
                    #         in_c2 = ca.MX(1, nContactSpheres*3*3)            
                    #         count = 0
                    #         for n_c in range(nContactSpheres):
                    #             c_n = str(n_c+1)
                    #             for s_c in side_contacts:
                    #                 in_c2[0, count*3:count*3+3] = (
                    #                     appliedFk[c_n][s_c])
                    #                 count += 1
                    #             in_c2[0, nContactSpheres*2*3+n_c*2] = (
                    #                 locSphere_inB_p[c_n]['r'][0,])
                    #             in_c2[0, nContactSpheres*2*3+n_c*2+1] = (
                    #                 locSphere_inB_p[c_n]['r'][2,])             
                    #             in_c2[0, nContactSpheres*2*4+n_c] = (
                    #                 radius_p[c_n])
                        
                    # # Call second external function (run inverse dynamics)
                    # Tk = F2_c(ca.vertcat(QsQdskj_nsc[:, 0], Qddsk_nsc,
                    #                      in_c2.T))
                    # if trackGRF:
                    #     GRF_rk = ca.MX(1, 3) 
                    #     GRF_lk = ca.MX(1, 3) 
                    #     for n_c in range(nContactSpheres):
                    #         c_n = str(n_c+1)
                    #         for s_c in side_contacts:
                    #             if s_c == 'r':
                    #                 GRF_rk += appliedFk[c_n][s_c]
                    #             elif s_c == 'l':
                    #                 GRF_lk += appliedFk[c_n][s_c]
                    #     GRFk = ca.horzcat(GRF_rk, GRF_lk).T                        
                    # if trackGRM:
                    #     GRMk = Tk[idxGRM]
                else:
                    if treadmill[trial]:
                        Tk = F[trial](ca.vertcat(
                            ca.vertcat(QsQdskj_nsc[:, 0],
                                       Qddsk_nsc[idxJoints4F]),
                            -treadmill_speed))
                    else:
                        Tk = F[trial](ca.vertcat(QsQdskj_nsc[:, 0], 
                                                 Qddsk_nsc[idxJoints4F]))
                        
                    # if trackGRF:
                    #     GRFk = Tk[idxGRF]                        
                    # if trackGRM:
                    #     GRMk = Tk[idxGRM]
                        
                for j in range(d):
                    
                    ###########################################################
                    # Expression for the state derivatives
                    ###########################################################
                    ap = ca.mtimes(akj, C[j+1])        
                    nFp_nsc = ca.mtimes(nFkj_nsc, C[j+1])
                    Qsp_nsc = ca.mtimes(Qskj_nsc, C[j+1])
                    Qdsp_nsc = ca.mtimes(Qdskj_nsc, C[j+1])
                    if withArms:
                        aArmp = ca.mtimes(aArmkj, C[j+1])
                    if withLumbarCoordinateActuators:
                        aLumbarp = ca.mtimes(aLumbarkj, C[j+1])
                    
                    ###########################################################
                    # Append collocation equations
                    ###########################################################
                    # Muscle activation dynamics (implicit formulation)
                    opti.subject_to((h*aDtk_nsc - ap) == 0)
                    # Muscle contraction dynamics (implicit formulation)  
                    opti.subject_to((h*nFDtk_nsc - nFp_nsc) / 
                                    scaling['F'][trial].to_numpy().T == 0)
                    # Skeleton dynamics (implicit formulation) 
                    # Position derivative
                    opti.subject_to((h*Qdskj_nsc[:, j+1] - Qsp_nsc) / 
                                    scaling['Qs'][trial].to_numpy().T == 0)
                    # Velocity derivative
                    opti.subject_to((h*Qddsk_nsc - Qdsp_nsc) / 
                                    scaling['Qds'][trial].to_numpy().T == 0)
                    if withArms:
                        # Arm activation dynamics (explicit formulation) 
                        aArmDtj = f_armActivationDynamics(
                            eArmk, aArmkj[:, j+1])
                        opti.subject_to(h*aArmDtj - aArmp == 0) 
                    if withLumbarCoordinateActuators:
                        # Lumbar activation dynamics (explicit formulation) 
                        aLumbarDtj = f_lumbarActivationDynamics(
                            eLumbark, aLumbarkj[:, j+1])
                        opti.subject_to(h*aLumbarDtj - aLumbarp == 0)
                    
                    ###########################################################
                    # Cost function
                    ###########################################################
                    activationTerm = f_NMusclesSumWeightedPow(
                        akj[:, j+1], s_muscleVolume * w_muscles)
                    jointAccelerationTerm = f_nJointsSum2(Qddsk)              
                    activationDtTerm = f_NMusclesSum2(aDtk)
                    forceDtTerm = f_NMusclesSum2(nFDtk)
                    positionTrackingTerm = f_NQsToTrackWSum2(
                        Qskj[idx_coordinates_toTrack, 0],
                        dataToTrack_sc_offset[:, k], w_dataToTrack)
                    velocityTrackingTerm = f_NQsToTrackWSum2(
                        Qdskj[idx_coordinates_toTrack, 0],
                        dataToTrack_dot_sc[trial][:, k], w_dataToTrack)
                    
                    J += ((
                        weights['positionTrackingTerm'] * positionTrackingTerm +
                        weights['velocityTrackingTerm'] * velocityTrackingTerm +
                        weights['activationTerm'] * activationTerm +
                        weights['jointAccelerationTerm'] * jointAccelerationTerm +                          
                        weights['activationDtTerm'] * activationDtTerm + 
                        weights['forceDtTerm'] * forceDtTerm) * h * B[j + 1])
                    
                    if withArms:     
                        armExcitationTerm = f_nArmJointsSum2(eArmk)
                        J += (weights['armExcitationTerm'] * 
                              armExcitationTerm * h * B[j + 1])
                    if withLumbarCoordinateActuators:     
                        lumbarExcitationTerm = f_nLumbarJointsSum2(eLumbark)
                        J += (weights['lumbarExcitationTerm'] * 
                              lumbarExcitationTerm * h * B[j + 1])
                    # if trackGRF:                    
                    #     grfTrackingTerm = f_NGRToTrackSum2(
                    #         (GRFk / (scaling['GRF'][trial].to_numpy().T)) - 
                    #         grfToTrack_sc[trial][:, k])
                    #     J += (weights['grfTrackingTerm'] * grfTrackingTerm * 
                    #           h * B[j + 1])                    
                    # if trackGRM:
                    #     grmTrackingTerm = f_NGRToTrackSum2(
                    #         (GRMk / (scaling['GRM'][trial].to_numpy().T)) - 
                    #         grmToTrack_sc[trial][:, k])
                    #     J += (weights['grmTrackingTerm'] * grmTrackingTerm * 
                    #           h * B[j + 1])
                    if trackQdds:
                        accelerationTrackingTerm = f_NQsToTrackWSum2(
                            Qddsk[idx_coordinates_toTrack],
                            dataToTrack_dotdot_sc[trial][:, k],
                            w_dataToTrack)
                        J += (weights['accelerationTrackingTerm'] * 
                              accelerationTrackingTerm * h * B[j + 1])                    
                    if reserveActuators:
                        reserveActuatorTerm = 0
                        for c_j in reserveActuatorJoints:                        
                            reserveActuatorTerm += ca.sumsqr(rActk[c_j])                            
                        reserveActuatorTerm /= len(reserveActuatorJoints)
                        J += (weights['reserveActuatorTerm'] * 
                              reserveActuatorTerm * h * B[j + 1])
                        
                    if isSTSs_yCalcn_vGRF[trial] and not weights['vGRFRatioTerm'] == 0:
                        vGRF_ratio_l = ca.sqrt((ca.sum1(Tk[idx_vGRF_front_l])) / 
                                               (ca.sum1(Tk[idx_vGRF_heel_l])))
                        vGRF_ratio_r = ca.sqrt((ca.sum1(Tk[idx_vGRF_front_r])) /
                                               (ca.sum1(Tk[idx_vGRF_heel_r])))
                        J += (weights['vGRFRatioTerm'] * 
                              (vGRF_ratio_l) * h * B[j + 1])
                        J += (weights['vGRFRatioTerm'] * 
                              (vGRF_ratio_r) * h * B[j + 1])
                    
                ###############################################################
                # Null pelvis residuals
                ###############################################################                
                opti.subject_to(Tk[idxGroundPelvisJointsinF, 0] == 0)
                
                ###############################################################
                # Skeleton dynamics (implicit formulation)
                ###############################################################
                # Muscle-driven joint torques
                for joint in muscleDrivenJoints:                
                    Fk_joint = Fk[momentArmIndices[joint]]
                    mTk_joint = ca.sum1(dMk[joint]*Fk_joint)
                    if reserveActuators and joint in reserveActuatorJoints:
                        mTk_joint += rActk_nsc[joint]
                    diffTk_joint = f_diffTorques(
                        Tk[F_map[trial]['residuals'][joint] ], mTk_joint,
                        passiveTorque_k[joint])
                    opti.subject_to(diffTk_joint == 0)
                    
                # Torque-driven joint torques              
                # Lumbar joints
                if withLumbarCoordinateActuators:
                    for cj, joint in enumerate(lumbarJoints):                        
                        coordAct_lumbar = (
                            scaling['LumbarE'][trial].iloc[0][joint] * 
                            aLumbarkj[cj, 0])
                        diffTk_lumbar = f_diffTorques(
                            Tk[F_map[trial]['residuals'][joint] ],
                            coordAct_lumbar, 
                            passiveTorque_k[joint])
                        opti.subject_to(diffTk_lumbar == 0)
                
                # Arm joints
                if withArms:
                    for cj, joint in enumerate(armJoints):
                        diffTk_joint = f_diffTorques(
                            Tk[F_map[trial]['residuals'][joint] ] / 
                            scaling['ArmE'][trial].iloc[0][joint],
                            aArmkj[cj, 0], linearPassiveTorqueArms_k[joint] /
                            scaling['ArmE'][trial].iloc[0][joint])
                        opti.subject_to(diffTk_joint == 0)
                
                # Mtp joints
                if withMTP:
                    for joint in mtpJoints:
                        diffTk_joint = f_diffTorques(
                            Tk[F_map[trial]['residuals'][joint] ], 
                            0, (passiveTorque_k[joint] +  
                                linearPassiveTorqueMtp_k[joint]))
                        opti.subject_to(diffTk_joint == 0)
                
                ###############################################################
                # Activation dynamics (implicit formulation)
                ###############################################################
                act1 = aDtk_nsc + akj[:, 0] / deactivationTimeConstant
                act2 = aDtk_nsc + akj[:, 0] / activationTimeConstant
                opti.subject_to(act1 >= 0)
                opti.subject_to(act2 <= 1 / activationTimeConstant)
                
                ###############################################################
                # Contraction dynamics (implicit formulation)
                ###############################################################
                opti.subject_to(hillEquilibriumk == 0)
                
                ###############################################################
                # Equality / continuity constraints
                ###############################################################
                opti.subject_to(a[trial][:, k+1] == ca.mtimes(akj, D))
                opti.subject_to(nF[trial][:, k+1] == ca.mtimes(nFkj, D))    
                opti.subject_to(Qs[trial][:, k+1] == ca.mtimes(Qskj, D))
                opti.subject_to(Qds[trial][:, k+1] == ca.mtimes(Qdskj, D))    
                if withArms:
                    opti.subject_to(aArm[trial][:, k+1] == 
                                    ca.mtimes(aArmkj, D))
                if withLumbarCoordinateActuators:
                    opti.subject_to(aLumbar[trial][:, k+1] == 
                                    ca.mtimes(aLumbarkj, D))
                    
                ###############################################################
                # Squat-specific constraints
                ###############################################################
                # We want all contact spheres to be in contact with the ground
                # at all time. This is specific to squats and corresponds to
                # instructions that would be given to subjets: "keep you feet
                # flat on the gound". Without such constraints, the model tends
                # to lean forward, likely to reduce quadriceps loads.
                if isSquats[trial]:
                    vGRFk = Tk[idx_vGRF_heel]
                    opti.subject_to(vGRFk > squatThresholds[trial])
                if isSTSs_yCalcn[trial] or isSTSs_yCalcn_vGRF[trial]:
                    yCalcnk = Tk[idx_yCalcn]
                    opti.subject_to(yCalcnk > yCalcnThresholds[trial])
                    if stsThresholds[trial] > 0:
                        vGRFk = Tk[idx_vGRF_heel_lr]
                        opti.subject_to(vGRFk > stsThresholds[trial])
                if yCalcnToes[trial]:
                    yCalcnToesk = Tk[idx_yCalcnToes]
                    opti.subject_to(yCalcnToesk > yCalcnToesThresholds[trial])
                    
                        
                # if isSTSs[trial] and k==0:
                #     vGRFk = Tk[idx_vGRF_buttocks]
                #     opti.subject_to(vGRFk > stsThresholds[trial])
                    
                    # for n_c in range(nContactSpheres):
                    #     c_n = str(n_c+1)
                    #     for s_c in side_contacts:
                    #         opti.subject_to(appliedFk[c_n][s_c][0,1] > 
                    #                         squatThresholds[trial])
                
            ###################################################################
            # Periodic constraints 
            ###################################################################
            if periodicConstraints:
                # Joint positions
                if 'Qs' in periodicConstraints:
                    opti.subject_to(Qs[trial][idxPeriodicQs, -1] - 
                                    Qs[trial][idxPeriodicQs, 0] == 0)
                # Joint velocities
                if 'Qds' in periodicConstraints:
                    opti.subject_to(Qds[trial][idxPeriodicQds, -1] - 
                                    Qds[trial][idxPeriodicQds, 0] == 0)
                # Muscle activations and forces
                if 'muscles' in periodicConstraints:
                    opti.subject_to(a[trial][idxPeriodicMuscles, -1] - 
                                    a[trial][idxPeriodicMuscles, 0] == 0)
                    opti.subject_to(nF[trial][idxPeriodicMuscles, -1] - 
                                    nF[trial][idxPeriodicMuscles, 0] == 0)
                if 'lumbar' in periodicConstraints:
                    # Lumbar activations
                    opti.subject_to(aLumbar[trial][idxPeriodicLumbar, -1] - 
                                    aLumbar[trial][idxPeriodicLumbar, 0] == 0)
                    
                # if withArms:
                #     # Arm activations
                #     opti.subject_to(aArm[trial][:, -1] - 
                #     aArm[trial][:, 0] == 0)
                    
            ###################################################################
            # Constraints on pelvis_ty if offset as design variable
            ###################################################################
            if (tracking_data == "coordinates" and offset_ty and 
                'pelvis_ty' in coordinate_constraints):
                
                pelvis_ty_sc_offset = (
                    pelvis_ty_sc[trial] + offset / 
                    scaling['Qs'][trial].iloc[0]["pelvis_ty"])            
                opti.subject_to(opti.bounded(
                    -coordinate_constraints['pelvis_ty']["env_bound"] / 
                    scaling['Qs'][trial].iloc[0]["pelvis_ty"],
                    Qs[trial][joints.index("pelvis_ty"), :-1] - 
                    pelvis_ty_sc_offset[0, :], 
                    coordinate_constraints['pelvis_ty']["env_bound"] / 
                    scaling['Qs'][trial].iloc[0]["pelvis_ty"]))
        
        # Create NLP solver        
        opti.minimize(J)
        
        # Solve problem
        from utilsOpenSimAD import solve_with_bounds
        w_opt, stats = solve_with_bounds(opti, tol)             
        np.save(os.path.join(pathResults, 'w_opt_{}.npy'.format(case)), w_opt)
        np.save(os.path.join(pathResults, 'stats_{}.npy'.format(case)), stats)
        
# %% Analyze
    if analyzeResults:
        w_opt = np.load(os.path.join(pathResults, 'w_opt_{}.npy'.format(case)))
        stats = np.load(os.path.join(pathResults, 'stats_{}.npy'.format(case)), 
                        allow_pickle=True).item()  
        if not stats['success'] == True:
            print('WARNING: PROBLEM DID NOT CONVERGE - {} - {} - {} \n\n'.format( 
                  stats['return_status'], subject, list(trials.keys())[0]))
            return
        
        # else:
        #     print('{} - {} - {}'.format( 
        #           stats['return_status'], subject, list(trials.keys())[0]))
        #     return
        
        starti = 0
        # if optimizeContacts:
        #     p_opt = w_opt[starti:starti+NContactParameters] 
        #     starti += NContactParameters
        if offset_ty:
            offset_opt = w_opt[starti:starti+1]
            starti += 1
            
        a_opt, a_col_opt = {}, {}
        nF_opt, nF_col_opt = {}, {}
        Qs_opt, Qs_col_opt = {}, {}
        Qds_opt, Qds_col_opt = {}, {}
        if withArms:
            aArm_opt, aArm_col_opt, eArm_opt = {}, {}, {}
        if withActiveMTP:    
            aMtp_opt, aMtp_col_opt, eMtp_opt = {}, {}, {}
        if withLumbarCoordinateActuators:
            aLumbar_opt, aLumbar_col_opt, eLumbar_opt = {}, {}, {}            
        aDt_opt = {}
        nFDt_col_opt = {}
        Qdds_col_opt = {}
        if reserveActuators:
            rAct_opt = {} 
        for trial in trials:
            a_opt[trial] = (
                np.reshape(w_opt[starti:starti+NMuscles*(N[trial]+1)],
                           (N[trial]+1, NMuscles))).T
            starti = starti + NMuscles*(N[trial]+1)
            a_col_opt[trial] = (
                np.reshape(w_opt[starti:starti+NMuscles*(d*N[trial])],
                           (d*N[trial], NMuscles))).T    
            starti = starti + NMuscles*(d*N[trial])
            nF_opt[trial] = (
                np.reshape(w_opt[starti:starti+NMuscles*(N[trial]+1)],
                           (N[trial]+1, NMuscles))).T  
            starti = starti + NMuscles*(N[trial]+1)
            nF_col_opt[trial] = (
                np.reshape(w_opt[starti:starti+NMuscles*(d*N[trial])],
                           (d*N[trial], NMuscles))).T
            starti = starti + NMuscles*(d*N[trial])
            Qs_opt[trial] = (
                np.reshape(w_opt[starti:starti+nJoints*(N[trial]+1)],
                           (N[trial]+1, nJoints))  ).T  
            starti = starti + nJoints*(N[trial]+1)    
            Qs_col_opt[trial] = (
                np.reshape(w_opt[starti:starti+nJoints*(d*N[trial])],
                           (d*N[trial], nJoints))).T
            starti = starti + nJoints*(d*N[trial])
            Qds_opt[trial] = (
                np.reshape(w_opt[starti:starti+nJoints*(N[trial]+1)],
                           (N[trial]+1, nJoints)) ).T   
            starti = starti + nJoints*(N[trial]+1)    
            Qds_col_opt[trial] = (
                np.reshape(w_opt[starti:starti+nJoints*(d*N[trial])],
                           (d*N[trial], nJoints))).T
            starti = starti + nJoints*(d*N[trial])    
            if withArms:
                aArm_opt[trial] = (
                    np.reshape(w_opt[starti:starti+nArmJoints*(N[trial]+1)],
                               (N[trial]+1, nArmJoints))).T
                starti = starti + nArmJoints*(N[trial]+1)    
                aArm_col_opt[trial] = (
                    np.reshape(w_opt[starti:starti+nArmJoints*(d*N[trial])],
                               (d*N[trial], nArmJoints))).T
                starti = starti + nArmJoints*(d*N[trial])       
            if withActiveMTP:
                aMtp_opt[trial] = (
                    np.reshape(w_opt[starti:starti+nMtpJoints*(N[trial]+1)],
                               (N[trial]+1, nMtpJoints))).T
                starti = starti + nMtpJoints*(N[trial]+1)    
                aMtp_col_opt[trial] = (
                    np.reshape(w_opt[starti:starti+nMtpJoints*(d*N[trial])],
                               (d*N[trial], nMtpJoints))).T
                starti = starti + nMtpJoints*(d*N[trial])
            if withLumbarCoordinateActuators:
                aLumbar_opt[trial] = (
                    np.reshape(w_opt[starti:starti+nLumbarJoints*(N[trial]+1)],
                               (N[trial]+1, nLumbarJoints))).T
                starti = starti + nLumbarJoints*(N[trial]+1)    
                aLumbar_col_opt[trial] = (
                    np.reshape(w_opt[starti:starti+nLumbarJoints*(d*N[trial])],
                               (d*N[trial], nLumbarJoints))).T
                starti = starti + nLumbarJoints*(d*N[trial])
            aDt_opt[trial] = (
                np.reshape(w_opt[starti:starti+NMuscles*N[trial]],
                           (N[trial], NMuscles))).T
            starti = starti + NMuscles*N[trial] 
            if withArms:
                eArm_opt[trial] = (
                    np.reshape(w_opt[starti:starti+nArmJoints*N[trial]],
                               (N[trial], nArmJoints))).T
                starti = starti + nArmJoints*N[trial]   
            if withActiveMTP:
                eMtp_opt[trial] = (
                    np.reshape(w_opt[starti:starti+nMtpJoints*N[trial]],
                               (N[trial], nMtpJoints))).T
                starti = starti + nMtpJoints*N[trial]
            if withLumbarCoordinateActuators:
                eLumbar_opt[trial] = (
                    np.reshape(w_opt[starti:starti+nLumbarJoints*N[trial]],
                               (N[trial], nLumbarJoints))).T
                starti = starti + nLumbarJoints*N[trial]
            nFDt_col_opt[trial] = (
                np.reshape(w_opt[starti:starti+NMuscles*(N[trial])],
                           (N[trial], NMuscles))).T
            starti = starti + NMuscles*(N[trial])
            Qdds_col_opt[trial] = (
                np.reshape(w_opt[starti:starti+nJoints*(N[trial])],
                           (N[trial], nJoints))).T
            starti = starti + nJoints*(N[trial])
            if reserveActuators:
                rAct_opt[trial] = {}
                for c_j in reserveActuatorJoints:
                    rAct_opt[trial][c_j] = (
                        np.reshape(w_opt[starti:starti+1*(N[trial])],
                               (N[trial], 1))).T
                    starti = starti + 1*(N[trial])
        assert (starti == w_opt.shape[0]), "error when extracting results"
            
        # %% Unscale results
        nF_opt_nsc, nF_col_opt_nsc = {}, {}
        Qs_opt_nsc, Qs_col_opt_nsc = {}, {}
        Qds_opt_nsc, Qds_col_opt_nsc = {}, {}
        aDt_opt_nsc = {}
        Qdds_col_opt_nsc = {}
        nFDt_col_opt_nsc = {}
        if reserveActuators:
            rAct_opt_nsc = {}
        for trial in trials:
            nF_opt_nsc[trial] = nF_opt[trial] * (scaling['F'][trial].to_numpy().T * np.ones((1, N[trial]+1)))
            nF_col_opt_nsc[trial] = nF_col_opt[trial] * (scaling['F'][trial].to_numpy().T * np.ones((1, d*N[trial])))    
            Qs_opt_nsc[trial] = Qs_opt[trial] * (scaling['Qs'][trial].to_numpy().T * np.ones((1, N[trial]+1)))
            Qs_col_opt_nsc[trial] = Qs_col_opt[trial] * (scaling['Qs'][trial].to_numpy().T * np.ones((1, d*N[trial])))
            Qds_opt_nsc[trial] = Qds_opt[trial] * (scaling['Qds'][trial].to_numpy().T * np.ones((1, N[trial]+1)))
            Qds_col_opt_nsc[trial] = Qds_col_opt[trial] * (scaling['Qds'][trial].to_numpy().T * np.ones((1, d*N[trial])))
            aDt_opt_nsc[trial] = aDt_opt[trial] * (scaling['ADt'][trial].to_numpy().T * np.ones((1, N[trial])))
            Qdds_col_opt_nsc[trial] = Qdds_col_opt[trial] * (scaling['Qdds'][trial].to_numpy().T * np.ones((1, N[trial])))
            nFDt_col_opt_nsc[trial] = nFDt_col_opt[trial] * (scaling['FDt'][trial].to_numpy().T * np.ones((1, N[trial])))
            if reserveActuators:
                rAct_opt_nsc[trial] = {}
                for c_j in reserveActuatorJoints:
                    rAct_opt_nsc[trial][c_j] = rAct_opt[trial][c_j] * (scaling['rAct'][trial][c_j].to_numpy().T * np.ones((1, N[trial])))
        if offset_ty:
            offset_opt_nsc = offset_opt * scaling['Offset'][trial]
                
        # if optimizeContacts:
        #     p_opt_nsc = ((p_opt.flatten() - scaling['ContactParameters_r'][trial]) / 
        #                   scaling['ContactParameters_v'][trial])
        
        # %% Extract passive joint torques
        if withMTP:
            linearPassiveTorqueMtp_opt = {}
            passiveTorqueMtp_opt = {}
            for trial in trials:                
                linearPassiveTorqueMtp_opt[trial] = (
                    np.zeros((nMtpJoints, N[trial]+1)))
                passiveTorqueMtp_opt[trial] = (
                    np.zeros((nMtpJoints, N[trial]+1)))
                for k in range(N[trial]+1):                    
                    for cj, joint in enumerate(mtpJoints):
                        linearPassiveTorqueMtp_opt[trial][cj, k] = (
                            f_linearPassiveMtpTorque(
                                Qs_opt_nsc[trial][joints.index(joint), k],
                                Qds_opt_nsc[trial][joints.index(joint), k]))
                        if enableLimitTorques:
                            passiveTorqueMtp_opt[trial][cj, k] = (
                                f_passiveTorque[joint](
                                    Qs_opt_nsc[trial][joints.index(joint), k], 
                                    Qds_opt_nsc[trial][joints.index(joint), k]))                        
        if withArms:
            linearPassiveTorqueArms_opt = {}
            for trial in trials:
                linearPassiveTorqueArms_opt[trial] = (
                    np.zeros((nArmJoints, N[trial]+1)))
                for k in range(N[trial]+1):  
                    for cj, joint in enumerate(armJoints):
                        linearPassiveTorqueArms_opt[trial][cj, k] = (
                            f_linearPassiveArmTorque(
                                Qs_opt_nsc[trial][joints.index(joint), k],
                                Qds_opt_nsc[trial][joints.index(joint), k]))
            
        # %% Extract joint torques and ground reaction forces.
        from utilsOpenSimAD import getCOP 
        
        QsQds_opt_nsc, Qdds_opt_nsc = {}, {}
        GRF_all_opt, GRM_all_opt, COP_all_opt, freeT_all_opt = {}, {}, {}, {}
        GRF_s_opt, COP_s_opt, torques_opt = {}, {}, {}
        GRF_all_opt['all'], GRM_all_opt['all'] = {}, {}
        for side in sides:
            GRF_all_opt[side] = {}
            GRM_all_opt[side] = {}
            COP_all_opt[side] = {}
            freeT_all_opt[side] = {}
            GRF_s_opt[side] = {}
            COP_s_opt[side] = {}
            for sphere in spheres:
                GRF_s_opt[side][sphere] = {}
                COP_s_opt[side][sphere] = {}
        
        for trial in trials:
            QsQds_opt_nsc[trial] = np.zeros((nJoints*2, N[trial]+1))
            QsQds_opt_nsc[trial][::2, :] = Qs_opt_nsc[trial][idxJoints4F, :]
            QsQds_opt_nsc[trial][1::2, :] = Qds_opt_nsc[trial][idxJoints4F, :]
            Qdds_opt_nsc[trial] = Qdds_col_opt_nsc[trial][idxJoints4F, :]
            if treadmill[trial]:
                Tj_temp = F[trial](ca.vertcat(
                    ca.vertcat(QsQds_opt_nsc[trial][:, 0], 
                               Qdds_opt_nsc[trial][:, 0]), -treadmill_speed))
            else:
                Tj_temp = F[trial](ca.vertcat(QsQds_opt_nsc[trial][:, 0], 
                                              Qdds_opt_nsc[trial][:, 0]))          
            F_out_pp = np.zeros((Tj_temp.shape[0], N[trial]))
            if withMTP:
                mtpT = np.zeros((nMtpJoints, N[trial]))
            if withArms:
                armT = np.zeros((nArmJoints, N[trial]))
            if optimizeContacts:
                GRF_optk = np.zeros((6 , N[trial]))
            for k in range(N[trial]):                
                if optimizeContacts:           
                    print("Not supported")
                    # # Call first external function
                    # T1k = F1_c(ca.vertcat(QsQds_opt_nsc[trial][:, k]))
                    # T1k_out = T1k.full()
                    
                    # RBGk_opt, posBk_opt, TBGk_opt, lVelBk_opt, aVelBk_opt = {}, {}, {}, {}, {}
                    # for segment in foot_segments:
                    #     RBG_flat = T1k_out[idxF_c1[segment]['rotation']]
                    #     RBGk_opt[segment] = ca.reshape(RBG_flat, 3, 3)
                    #     posBk_opt[segment] = T1k_out[idxF_c1[segment]['position']]
                    #     TBGk_opt[segment] = T1k_out[idxF_c1[segment]['translation']]
                    #     lVelBk_opt[segment] = T1k_out[idxF_c1[segment]['linear_velocity']]
                    #     aVelBk_opt[segment] = T1k_out[idxF_c1[segment]['angular_velocity']]
                    
                    # if nContactSpheres == 6:                        
                    #     if parameter_to_optimize == "option1":
                    #         locSphere_inB_opt = {}
                    #         radius_opt = {}
                    #         appliedFk_opt = {}
                    #         for n_c in range(nContactSpheres):
                    #             c_n = str(n_c+1)
                    #             locSphere_inB_opt[c_n] = {}
                    #             locSphere_inB_opt[c_n]['r'] = (
                    #                 ca.vertcat(p_opt_nsc[n_c*2], -0.01, p_opt_nsc[n_c*2+1]))
                    #             locSphere_inB_opt[c_n]['l'] = (
                    #                 ca.vertcat(p_opt_nsc[n_c*2], -0.01, -p_opt_nsc[n_c*2+1]))
                    #             radius_opt[c_n] = (p_opt_nsc[2*nContactSpheres+n_c])
                    #             appliedFk_opt[c_n] = {}
                    #             for s_c in side_contacts:
                    #                 c_s = contactSegments[n_c] + '_' + s_c                    
                    #                 appliedFk_opt[c_n][s_c] = f_contactForce(
                    #                     dissipation, stiffness, radius_opt[c_n],
                    #                     locSphere_inB_opt[c_n][s_c], posBk_opt[c_s],
                    #                     lVelBk_opt[c_s], aVelBk_opt[c_s], RBGk_opt[c_s],
                    #                     TBGk_opt[c_s])                                    
                    #         in_c2_opt = ca.DM(1, nContactSpheres*3*3)            
                    #         count = 0
                    #         for n_c in range(nContactSpheres):
                    #             c_n = str(n_c+1)
                    #             for s_c in side_contacts:
                    #                 in_c2_opt[0, count*3:count*3+3] = (
                    #                     appliedFk_opt[c_n][s_c])
                    #                 count += 1
                    #             in_c2_opt[0, nContactSpheres*2*3+n_c*2] = (
                    #                 locSphere_inB_opt[c_n]['r'][0,])
                    #             in_c2_opt[0, nContactSpheres*2*3+n_c*2+1] = (
                    #                 locSphere_inB_opt[c_n]['r'][2,])             
                    #             in_c2_opt[0, nContactSpheres*2*4+n_c] = (
                    #                 radius_opt[c_n]) 
                    #     Tk = F1[trial](ca.vertcat(QsQds_opt_nsc[trial][:, k],
                    #                        Qdds_opt_nsc[trial][:, k], in_c2_opt.T))
                        
                    #     GRF_rk_opt = ca.DM(1, 3) 
                    #     GRF_lk_opt = ca.DM(1, 3) 
                    #     for n_c in range(nContactSpheres):
                    #         c_n = str(n_c+1)
                    #         for s_c in side_contacts:
                    #             if s_c == 'r':
                    #                 GRF_rk_opt += appliedFk_opt[c_n][s_c]
                    #             elif s_c == 'l':
                    #                 GRF_lk_opt += appliedFk_opt[c_n][s_c]
                    #     GRF_optk[:, k] = ca.horzcat(GRF_rk_opt, GRF_lk_opt)                
                else:
                    if treadmill[trial]:
                        Tk = F[trial](ca.vertcat(
                            ca.vertcat(QsQds_opt_nsc[trial][:, k],
                                       Qdds_opt_nsc[trial][:, k]), 
                            -treadmill_speed))
                    else:
                        Tk = F[trial](ca.vertcat(QsQds_opt_nsc[trial][:, k],
                                       Qdds_opt_nsc[trial][:, k]))
                F_out_pp[:, k] = Tk.full().T                
                if withMTP:
                    for cj, joint in enumerate(mtpJoints):
                        c_aMtp_opt = 0
                        if withActiveMTP:
                            c_aMtp_opt = aMtp_opt[trial][cj, k]
                        mtpT[cj, k] = f_diffTorques(
                            F_out_pp[F_map[trial]['residuals'][joint], k], c_aMtp_opt, 
                            (linearPassiveTorqueMtp_opt[trial][cj, k] + 
                             passiveTorqueMtp_opt[trial][cj, k]))
                if withArms:
                    for cj, joint in enumerate(armJoints):
                        armT[cj, k] = f_diffTorques(
                            F_out_pp[F_map[trial]['residuals'][joint], k] / 
                            scaling['ArmE'][trial].iloc[0][joint], 
                            aArm_opt[trial][cj, k], 
                            linearPassiveTorqueArms_opt[trial][cj, k] / 
                            scaling['ArmE'][trial].iloc[0][joint])                
            # Sanity checks.
            if stats['success'] and withArms:
                assert np.alltrue(np.abs(armT) < 10**(-tol)), (
                    "Error arm torques balance")                    
            if stats['success'] and withMTP:
                assert np.alltrue(np.abs(mtpT) < 10**(-tol)), (
                    "Error mtp torques balance")
                
            for side in sides:
                GRF_all_opt[side][trial] = F_out_pp[idxGR["GRF"]["all"][side], :]
                GRM_all_opt[side][trial] = F_out_pp[idxGR["GRM"]["all"][side], :]
                COP_all_opt[side][trial], freeT_all_opt[side][trial] = getCOP(
                    GRF_all_opt[side][trial], GRM_all_opt[side][trial])
            
            GRF_all_opt['all'][trial] = np.concatenate(
                (GRF_all_opt['r'][trial], GRF_all_opt['l'][trial]), axis=0)   
            GRM_all_opt['all'][trial] = np.concatenate(
                (GRM_all_opt['r'][trial], GRM_all_opt['l'][trial]), axis=0)
            
            if optimizeContacts and stats['success']:               
               assert np.alltrue(
                       np.abs(GRF_optk - GRF_all_opt['all'][trial]) 
                       < 10**(-5)), "error GRFs"               
            for side in sides:
                for sphere in spheres:
                    GRF_s_opt[side][sphere][trial] = F_out_pp[idxGR["GRF"][sphere][side], :]
                    COP_s_opt[side][sphere][trial] = F_out_pp[idxGR["COP"][sphere][side], :]
            
            
            torques_opt[trial] = F_out_pp[[F_map[trial]['residuals'][joint] for joint in joints], :]
            
        # %% Re-organize data for plotting and GUI
        Qs_opt_nsc_deg = {}
        for trial in trials:
            Qs_opt_nsc_deg[trial] = copy.deepcopy(Qs_opt_nsc[trial])
            Qs_opt_nsc_deg[trial][idxRotationalJoints, :] = (
                Qs_opt_nsc_deg[trial][idxRotationalJoints, :] * 180 / np.pi)               
        
        # %% Write motion files for visualization in OpenSim GUI
        GR_labels = {}
        GR_labels["GRF"] = {}
        GR_labels["COP"] = {}
        GR_labels["GRM"] = {}
        for i in range(1,nContactSpheres+1):
            GR_labels["GRF"]["s" + str(i)] = {}
            GR_labels["COP"]["s" + str(i)] = {}
            GR_labels["GRM"]["s" + str(i)] = {}
            if i < 2:
                GR_labels["GRF"]["all"] = {}
                GR_labels["COP"]["all"] = {}
                GR_labels["GRM"]["all"] = {}
            for side in sides:
                GR_labels["GRF"]["s" + str(i)][side] = []
                GR_labels["COP"]["s" + str(i)][side] = []
                GR_labels["GRM"]["s" + str(i)][side] = []
                if i < 2:
                    GR_labels["GRF"]["all"][side] = []
                    GR_labels["COP"]["all"][side] = []
                    GR_labels["GRM"]["all"][side] = []
                for dimension in dimensions:
                    GR_labels["GRF"]["s" + str(i)][side] = (
                        GR_labels["GRF"]["s" + str(i)][side] + 
                        ["ground_force_s" + str(i) + "_" + side + "_v" + dimension])
                    GR_labels["COP"]["s" + str(i)][side] = (
                        GR_labels["COP"]["s" + str(i)][side] + 
                          ["ground_force_s" + str(i) + "_" + side + "_p" + dimension])
                    GR_labels["GRM"]["s" + str(i)][side] = (
                        GR_labels["GRM"]["s" + str(i)][side] + 
                        ["ground_torque_s" + str(i) + "_" + side + "_" + dimension])
                    if i < 2:
                        GR_labels["GRF"]["all"][side] = (
                        GR_labels["GRF"]["all"][side] + 
                        ["ground_force_" + side + "_v" + dimension])
                        GR_labels["COP"]["all"][side] = (
                        GR_labels["COP"]["all"][side] + 
                        ["ground_force_" + side + "_p" + dimension])
                        GR_labels["GRM"]["all"][side] = (
                        GR_labels["GRM"]["all"][side] + 
                        ["ground_torque_" + side + "_" + dimension]) 
        if writeGUI:    
            muscleLabels = ([bothSidesMuscle + '/activation' 
                              for bothSidesMuscle in bothSidesMuscles])        
            labels = ['time'] + joints   
            labels_w_muscles = labels + muscleLabels
            from utils import numpy_to_storage
            for trial in trials:
                data = np.concatenate((tgridf[trial].T, Qs_opt_nsc_deg[trial].T, a_opt[trial].T),axis=1)           
                numpy_to_storage(labels_w_muscles, data, os.path.join(
                    pathResults, 'kinematics_activations_{}_{}.mot'.format(trial, case)),
                    datatype='IK')
                
            labels = []
            for joint in joints:
                if (joint == 'pelvis_tx' or joint == 'pelvis_ty' or joint == 'pelvis_tz'):
                    temp_suffix = "_force"
                else:
                    temp_suffix = "_moment"
                labels.append(joint + temp_suffix)
            labels = ['time'] + labels
            for trial in trials:
                data = np.concatenate((tgridf[trial].T[:-1], torques_opt[trial].T), axis=1) 
                numpy_to_storage(labels, data, os.path.join(
                    pathResults, 'kinetics_{}_{}.mot'.format(trial, case)),
                    datatype='ID')
                       
            labels = ['time']
            for sphere in spheres:
                for side in sides:
                    labels += GR_labels["GRF"][sphere][side]
                    labels += GR_labels["COP"][sphere][side]
            for sphere in spheres:
                for side in sides:
                    labels += GR_labels["GRM"][sphere][side]
            for trial in trials:                
                # data_size = tgridf[trial].T[:-1].shape[0]
                data = np.zeros((tgridf[trial].T[:-1].shape[0], 
                                 1+nContactSpheres*2*9))
                data[:,0] = tgridf[trial].T[:-1].flatten()
                idx_acc = 1
                for sphere in spheres:
                    for side in sides:
                        data[:,idx_acc:idx_acc+3] = GRF_s_opt[side][sphere][trial].T
                        idx_acc += 3
                        data[:,idx_acc:idx_acc+3] = COP_s_opt[side][sphere][trial].T
                        idx_acc += 3                
                numpy_to_storage(labels, data, os.path.join(
                    pathResults, 'GRF_{}_{}.mot'.format(trial, case)),
                    datatype='GRF')                
            labels = (['time'] + 
                      GR_labels["GRF"]["all"]["r"] +  
                      GR_labels["COP"]["all"]["r"] +
                      GR_labels["GRF"]["all"]["l"] + 
                      GR_labels["COP"]["all"]["l"] +
                      GR_labels["GRM"]["all"]["r"] +
                      GR_labels["GRM"]["all"]["l"])
            for trial in trials:
                data = np.concatenate(
                    (tgridf[trial].T[:-1], 
                     GRF_all_opt['r'][trial].T, COP_all_opt['r'][trial].T,
                     GRF_all_opt['l'][trial].T, COP_all_opt['l'][trial].T, 
                     freeT_all_opt['r'][trial].T, freeT_all_opt['l'][trial].T,),
                    axis=1)                
                numpy_to_storage(labels, data, os.path.join(
                    pathResults, 'GRF_resultant_{}_{}.mot'.format(trial, case)),
                    datatype='GRF')

        # %% Visualize tracking results
        refData_offset_nsc = {}
        for trial in trials:
            refData_nsc = Qs_toTrack[trial].to_numpy()[:,1::].T
            refData_offset_nsc[trial] = copy.deepcopy(refData_nsc)
            if offset_ty and tracking_data == "coordinates":                    
                refData_offset_nsc[trial][joints.index("pelvis_ty")] = (
                    refData_nsc[joints.index("pelvis_ty")] + 
                    offset_opt_nsc)
        GR_labels_fig = (GR_labels['GRF']['all']['r'] + 
                                 GR_labels['GRF']['all']['l'])
        if visualizeTracking:
            import matplotlib.pyplot as plt
            for trial in trials:
                    
                # Visualize optimal joint coordinates against IK results.
                # Depending on the problem, the IK results might be the
                # tracking reference, but not if marker tracking problem.
                # Joint coordinate values
                ny = np.ceil(np.sqrt(nJoints))   
                fig, axs = plt.subplots(int(ny), int(ny), sharex=True)    
                fig.suptitle('Joint positions: DC vs IK - {}'.format(trial))                  
                for i, ax in enumerate(axs.flat):
                    if i < nJoints:
                        if joints[i] in rotationalJoints:
                            scale_angles = 180 / np.pi
                        else:
                            scale_angles = 1
                        # reference data
                        ax.plot(tgridf[trial][0,:-1].T, 
                                refData_offset_nsc[trial][i:i+1,:].T * scale_angles, 
                                c='black', label='experimental')
                        # simulated data
                        if tracking_data == "coordinates":
                            if (joints[i] in coordinates_toTrack_list):
                                col_sim = 'orange'
                            else:
                                col_sim = 'blue'
                        else:
                            col_sim = 'blue'                            
                        
                        ax.plot(tgridf[trial][0,:].T, 
                                Qs_opt_nsc[trial][i:i+1,:].T * scale_angles, 
                                c=col_sim, label='simulated')
                        ax.set_title(joints[i])
                plt.setp(axs[-1, :], xlabel='Time (s)')
                plt.setp(axs[:, 0], ylabel='(deg or m)')
                fig.align_ylabels()
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
                plt.draw()
                
                # Joint coordinate speeds
                ny = np.ceil(np.sqrt(nJoints))   
                fig, axs = plt.subplots(int(ny), int(ny), sharex=True)    
                fig.suptitle('Joint speeds: DC vs IK - {}'.format(trial))                  
                for i, ax in enumerate(axs.flat):
                    if i < nJoints:
                        if joints[i] in rotationalJoints:
                            scale_angles = 180 / np.pi
                        else:
                            scale_angles = 1
                        # reference data
                        ax.plot(tgridf[trial][0,:-1].T, 
                                refData_dot_nsc[trial][i:i+1,:].T * scale_angles, 
                                c='black', label='experimental')
                        # simulated data
                        if tracking_data == "coordinates":
                            if (joints[i] in coordinates_toTrack_list):
                                col_sim = 'orange'
                            else:
                                col_sim = 'blue'
                        else:
                            col_sim = 'blue'                            
                        
                        ax.plot(tgridf[trial][0,:].T, 
                                Qds_opt_nsc[trial][i:i+1,:].T * scale_angles, 
                                c=col_sim, label='simulated')
                        ax.set_title(joints[i])
                plt.setp(axs[-1, :], xlabel='Time (s)')
                plt.setp(axs[:, 0], ylabel='(deg/s or m/s)')
                fig.align_ylabels()
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
                plt.draw()
                
                # Joint coordinate accelerations
                ny = np.ceil(np.sqrt(nJoints))   
                fig, axs = plt.subplots(int(ny), int(ny), sharex=True)    
                fig.suptitle('Joint accelerations: DC vs IK - {}'.format(trial))                  
                for i, ax in enumerate(axs.flat):
                    if i < nJoints:
                        if joints[i] in rotationalJoints:
                            scale_angles = 180 / np.pi
                        else:
                            scale_angles = 1
                        # reference data
                        ax.plot(tgridf[trial][0,:-1].T, 
                                refData_dotdot_nsc[trial][i:i+1,:].T * scale_angles, 
                                c='black', label='experimental')
                        # simulated data
                        if tracking_data == "coordinates":
                            if (joints[i] in coordinates_toTrack_list):
                                col_sim = 'orange'
                            else:
                                col_sim = 'blue'
                        else:
                            col_sim = 'blue'                            
                        
                        ax.plot(tgridf[trial][0,:-1].T, 
                                Qdds_col_opt_nsc[trial][i:i+1,:].T * scale_angles, 
                                c=col_sim, label='simulated')
                        ax.set_title(joints[i])
                plt.setp(axs[-1, :], xlabel='Time (s)')
                plt.setp(axs[:, 0], ylabel='(deg/s2 or m/s2)')
                fig.align_ylabels()
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
                plt.draw()
                
                # Joint torques
                ny = np.ceil(np.sqrt(nJoints))   
                fig, axs = plt.subplots(int(ny), int(ny), sharex=True)    
                fig.suptitle('Joint torques {}'.format(trial))                  
                for i, ax in enumerate(axs.flat):
                    if i < nJoints:
                        # # reference data
                        if ID_toTrack:
                            ax.plot(tgridf[trial][0,:-1].T, 
                                    ID_toTrack[trial][i:i+1,:].T, 
                                    c='black', label='experimental')
                            
                        # ax.plot(tgridf[trial][0,:-1].T, 
                        #         dataToTrack_dotdot_nsc[trial][i:i+1,:].T * scale_angles, 
                        #         c='black', label='experimental')
                        # simulated data
                        if tracking_data == "coordinates":
                            if (joints[i] in coordinates_toTrack_list):
                                col_sim = 'orange'
                            else:
                                col_sim = 'blue'
                        else:
                            col_sim = 'blue'                            
                        
                        ax.plot(tgridf[trial][0,:-1].T, 
                                torques_opt[trial][i:i+1,:].T, 
                                c=col_sim, label='simulated')
                        ax.set_title(joints[i])
                plt.setp(axs[-1, :], xlabel='Time (s)')
                plt.setp(axs[:, 0], ylabel='(Nm)')
                fig.align_ylabels()
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
                # plt.draw()
                
                # GRFs
                ny = np.ceil(np.sqrt(nJoints))   
                fig, axs = plt.subplots(2, 3, sharex=True)    
                fig.suptitle('GRFs - {}'.format(trial))                  
                for i, ax in enumerate(axs.flat):
                    if i < nJoints:
                        # if joints[i] in rotationalJoints:
                        #     scale_angles = 180 / np.pi
                        # else:
                        #     scale_angles = 1
                        # reference data
                        if GRF_toTrack:
                            ax.plot(tgridf[trial][0,:-1].T, 
                                    grfToTrack_nsc[trial][i:i+1,:].T, 
                                    c='black', label='experimental')
                        # simulated data
                        # if tracking_data == "coordinates":
                        #     if (joints[i] in coordinates_toTrack_list):
                        #         col_sim = 'orange'
                        #     else:
                        #         col_sim = 'blue'
                        # else:
                        col_sim = 'orange'                       
                        ax.plot(tgridf[trial][0,:-1].T, 
                                GRF_all_opt['all'][trial][i:i+1,:].T, 
                                c=col_sim, label='simulated')
                        ax.set_title(GR_labels_fig[i])
                plt.setp(axs[-1, :], xlabel='Time (s)')
                plt.setp(axs[:, 0], ylabel='(N)')
                fig.align_ylabels()
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
                plt.draw()
                
                # GRMs
                ny = np.ceil(np.sqrt(nJoints))   
                fig, axs = plt.subplots(2, 3, sharex=True)    
                fig.suptitle('GRMs - {}'.format(trial))                  
                for i, ax in enumerate(axs.flat):
                    if i < nJoints:
                        # if joints[i] in rotationalJoints:
                        #     scale_angles = 180 / np.pi
                        # else:
                        #     scale_angles = 1
                        # reference data
                        if GRF_toTrack:
                            ax.plot(tgridf[trial][0,:-1].T, 
                                    grmToTrack_nsc[trial][i:i+1,:].T, 
                                    c='black', label='experimental')
                        # simulated data
                        # if tracking_data == "coordinates":
                        #     if (joints[i] in coordinates_toTrack_list):
                        #         col_sim = 'orange'
                        #     else:
                        #         col_sim = 'blue'
                        # else:
                        col_sim = 'orange'                       
                        ax.plot(tgridf[trial][0,:-1].T, 
                                GRM_all_opt['all'][trial][i:i+1,:].T, 
                                c=col_sim, label='simulated')
                        ax.set_title(GR_labels_fig[i])
                plt.setp(axs[-1, :], xlabel='Time (s)')
                plt.setp(axs[:, 0], ylabel='(Nm)')
                fig.align_ylabels()
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
                plt.draw()
                
        # %%
        dataToTrack_nsc_offset_opt = {}
        for trial in trials:
            dataToTrack_nsc_offset_opt[trial] = np.zeros((dataToTrack_nsc[trial].shape[0],
                                                   dataToTrack_nsc[trial].shape[1]))                    
            for j, joint in enumerate(coordinates_toTrack):                        
                if joint == "pelvis_ty":                        
                    dataToTrack_nsc_offset_opt[trial][j, :] = dataToTrack_nsc[trial][j, :] + offset_opt[0][0]
                else:
                    dataToTrack_nsc_offset_opt[trial][j, :] = dataToTrack_nsc[trial][j, :]                
            dataToTrack_nsc_offset_opt[trial] = (
                dataToTrack_nsc_offset_opt[trial] / 
                ((scaling['Qs'][trial].to_numpy().T)[idx_coordinates_toTrack] * 
                  np.ones((1, N[trial]))))
                          
        # %% Contribution to the cost function
        activationTerm_opt_all = 0
        if withArms:
            armExcitationTerm_opt_all = 0
        if withActiveMTP:
            mtpExcitationTerm_opt_all = 0
        if withLumbarCoordinateActuators:    
            lumbarExcitationTerm_opt_all = 0
        if trackQdds:
            accelerationTrackingTerm_opt_all = 0                
        jointAccelerationTerm_opt_all = 0
        # passiveTorqueTerm_opt_all = 0
        activationDtTerm_opt_all = 0
        forceDtTerm_opt_all = 0
        positionTrackingTerm_opt_all = 0
        velocityTrackingTerm_opt_all = 0
        if trackGRF:
            grfTrackingTerm_opt_all = 0
        if trackGRM:
            grmTrackingTerm_opt_all = 0
        if reserveActuators:    
            reserveActuatorTerm_opt_all = 0
            
        vGRFRatioTerm_opt_all = 0
            
        pMT_opt = {}
        aMT_opt = {}
        pT_opt = {}
        Ft_opt = {}
        
        for trial in trials:
            
            pMT_opt[trial] = np.zeros((len(muscleDrivenJoints), N[trial]))
            aMT_opt[trial] = np.zeros((len(muscleDrivenJoints), N[trial]))
            pT_opt[trial] = np.zeros((nPassiveTorqueJoints, N[trial]))
            Ft_opt[trial] = np.zeros((NMuscles, N[trial]))
            
            h = timeElapsed[trial] / N[trial]   
        
            for k in range(N[trial]):
                # States 
                akj_opt = (ca.horzcat(a_opt[trial][:, k], a_col_opt[trial][:, k*d:(k+1)*d]))
                nFkj_opt = (ca.horzcat(nF_opt[trial][:, k], nF_col_opt[trial][:, k*d:(k+1)*d]))
                nFkj_opt_nsc = nFkj_opt * (scaling['F'][trial].to_numpy().T * np.ones((1, d+1)))   
                Qskj_opt = (ca.horzcat(Qs_opt[trial][:, k], Qs_col_opt[trial][:, k*d:(k+1)*d]))
                Qskj_opt_nsc = Qskj_opt * (scaling['Qs'][trial].to_numpy().T * np.ones((1, d+1)))
                Qdskj_opt = (ca.horzcat(Qds_opt[trial][:, k], Qds_col_opt[trial][:, k*d:(k+1)*d]))
                Qdskj_opt_nsc = Qdskj_opt * (scaling['Qds'][trial].to_numpy().T * np.ones((1, d+1)))
                # Controls
                aDtk_opt = aDt_opt[trial][:, k]
                # aDtk_opt_nsc = aDt_opt_nsc[trial][:, k]
                if withArms:
                    eArmk_opt = eArm_opt[trial][:, k]
                if withActiveMTP:
                    eMtpk_opt = eMtp_opt[trial][:, k]
                if withLumbarCoordinateActuators:
                    eLumbark_opt = eLumbar_opt[trial][:, k]
                if reserveActuators:
                    rActk_opt = {}
                    for c_j in reserveActuatorJoints:
                        rActk_opt[c_j] = rAct_opt[trial][c_j][:, k]
                    
                # Slack controls
                Qddsk_opt = Qdds_col_opt[trial][:, k]
                # Qddsk_opt_nsc = Qdds_col_opt_nsc[trial][:, k]
                nFDtk_opt = nFDt_col_opt[trial][:, k] 
                nFDtk_opt_nsc = nFDt_col_opt_nsc[trial][:, k]
            
                QsQdskj_opt_nsc = ca.DM(nJoints*2, d+1)
                QsQdskj_opt_nsc[::2, :] = Qskj_opt_nsc
                QsQdskj_opt_nsc[1::2, :] = Qdskj_opt_nsc
                
                ###############################################################
                # Polynomial approximations
                ###############################################################                
                # Left side.
                Qsink_opt_l = Qskj_opt_nsc[leftPolynomialJointIndices, 0]
                Qdsink_opt_l = Qdskj_opt_nsc[leftPolynomialJointIndices, 0]
                [lMTk_opt_l, vMTk_opt_l, dMk_opt_l] = f_polynomial['l'](
                    Qsink_opt_l, Qdsink_opt_l) 
                # Right side.
                Qsink_opt_r = Qskj_opt_nsc[rightPolynomialJointIndices, 0]
                Qdsink_opt_r = Qdskj_opt_nsc[rightPolynomialJointIndices, 0]
                [lMTk_opt_r, vMTk_opt_r, dMk_opt_r] = f_polynomial['r'](
                    Qsink_opt_r, Qdsink_opt_r)
                # Muscle-tendon lengths and velocities.      
                lMTk_opt_lr = ca.vertcat(lMTk_opt_l[leftPolynomialMuscleIndices], 
                                     lMTk_opt_r[rightPolynomialMuscleIndices])
                vMTk_opt_lr = ca.vertcat(vMTk_opt_l[leftPolynomialMuscleIndices], 
                                     vMTk_opt_r[rightPolynomialMuscleIndices])
                
                # Moment arms.
                dMk_opt = {}
                # Left side.
                for joint in leftPolynomialJoints:
                    if ((joint != 'mtp_angle_l') and 
                        (joint != 'lumbar_extension') and
                        (joint != 'lumbar_bending') and 
                        (joint != 'lumbar_rotation')):
                            dMk_opt[joint] = dMk_opt_l[
                                momentArmIndices[joint], 
                                leftPolynomialJoints.index(joint)]
                # Right side.
                for joint in rightPolynomialJoints:
                    if ((joint != 'mtp_angle_r') and 
                        (joint != 'lumbar_extension') and
                        (joint != 'lumbar_bending') and 
                        (joint != 'lumbar_rotation')):
                            # We need to adjust momentArmIndices for the right 
                            # side since polynomial indices are 'one-sided'. 
                            # We subtract by the number of side muscles.
                            c_ma = [i - nSideMuscles for 
                                    i in momentArmIndices[joint]]
                            dMk_opt[joint] = dMk_opt_r[
                                c_ma, rightPolynomialJoints.index(joint)]
                # Trunk.
                if model_type == 'gait2392':
                    for joint in lumbarJoints:
                        dMk_opt[joint] = dMk_opt_l[trunkMomentArmPolynomialIndices, 
                                           leftPolynomialJoints.index(joint)]                        
                
                # Hill-equilibrium
                [hillEqk_opt, Fk_opt, _, _,_, _, _, aFPk_opt, pFPk_opt] = (
                    f_hillEquilibrium(akj_opt[:, 0], lMTk_opt_lr, vMTk_opt_lr,
                                      nFkj_opt_nsc[:, 0], nFDtk_opt_nsc))
                Ft_opt[trial][:,k] = Fk_opt.full().flatten()                
                
                # Passive muscle moments
                for c_j, joint in enumerate(muscleDrivenJoints):
                    pFk_opt_joint = pFPk_opt[momentArmIndices[joint]]
                    pMT_opt[trial][c_j, k] = ca.sum1(
                        dMk_opt[joint]*pFk_opt_joint)
                    
                # Active muscle moments
                for c_j, joint in enumerate(muscleDrivenJoints):
                    aFk_opt_joint = aFPk_opt[momentArmIndices[joint]]
                    aMT_opt[trial][c_j, k] = ca.sum1(
                        dMk_opt[joint]*aFk_opt_joint)                
                
                # Passive limit moments
                if enableLimitTorques:
                    for c_j, joint in enumerate(passiveTorqueJoints):
                        pT_opt[trial][c_j, k] = f_passiveTorque[joint](
                            Qskj_opt_nsc[joints.index(joint), 0], 
                            Qdskj_opt_nsc[joints.index(joint), 0])
                
                for j in range(d):
                    # Motor control terms.
                    activationTerm_opt = f_NMusclesSumWeightedPow(akj_opt[:, j+1], s_muscleVolume * w_muscles)
                    jointAccelerationTerm_opt = f_nJointsSum2(Qddsk_opt)
                    activationDtTerm_opt = f_NMusclesSum2(aDtk_opt)
                    forceDtTerm_opt = f_NMusclesSum2(nFDtk_opt)
                    # passiveTorqueTerm_opt = f_nPassiveTorqueJointsSum2(passiveTorquesj_opt)
                    positionTrackingTerm_opt = f_NQsToTrackWSum2(Qskj_opt[idx_coordinates_toTrack, 0], dataToTrack_nsc_offset_opt[trial][:, k], w_dataToTrack)                
                    velocityTrackingTerm_opt = f_NQsToTrackWSum2(Qdskj_opt[idx_coordinates_toTrack, 0], dataToTrack_dot_sc[trial][:, k], w_dataToTrack)
                    
                    positionTrackingTerm_opt_all += weights['positionTrackingTerm'] * positionTrackingTerm_opt * h * B[j + 1]
                    velocityTrackingTerm_opt_all += weights['velocityTrackingTerm'] * velocityTrackingTerm_opt * h * B[j + 1]
                    activationTerm_opt_all += weights['activationTerm'] * activationTerm_opt * h * B[j + 1]
                    jointAccelerationTerm_opt_all += weights['jointAccelerationTerm'] * jointAccelerationTerm_opt * h * B[j + 1]
                    activationDtTerm_opt_all += weights['activationDtTerm'] * activationDtTerm_opt * h * B[j + 1]
                    forceDtTerm_opt_all += weights['forceDtTerm'] * forceDtTerm_opt * h * B[j + 1] 
                    # passiveTorqueTerm_opt_all += weights['passiveTorqueTerm'] * passiveTorqueTerm_opt * h * B[j + 1]
                    if withArms:
                        armExcitationTerm_opt = f_nArmJointsSum2(eArmk_opt) 
                        armExcitationTerm_opt_all += weights['armExcitationTerm'] * armExcitationTerm_opt * h * B[j + 1]
                    if withActiveMTP:
                        mtpExcitationTerm_opt = f_nMtpJointsSum2(eMtpk_opt) 
                        mtpExcitationTerm_opt_all += weights['mtpExcitationTerm'] * mtpExcitationTerm_opt * h * B[j + 1]
                    if withLumbarCoordinateActuators:
                        lumbarExcitationTerm_opt = f_nLumbarJointsSum2(eLumbark_opt) 
                        lumbarExcitationTerm_opt_all += weights['lumbarExcitationTerm'] * lumbarExcitationTerm_opt * h * B[j + 1]
                    if trackQdds:
                        accelerationTrackingTerm_opt = f_NQsToTrackWSum2(Qddsk_opt[idx_coordinates_toTrack], dataToTrack_dotdot_sc[trial][:, k], w_dataToTrack)
                        accelerationTrackingTerm_opt_all += (weights['accelerationTrackingTerm'] * accelerationTrackingTerm_opt * h * B[j + 1])
                        
                    if trackGRF:                    
                        grfTrackingTerm_opt = f_NGRToTrackSum2(
                            (GRF_all_opt['all'][trial][:, k:k+1] / (scaling['GRF'][trial].to_numpy().T)) - 
                            grfToTrack_sc[trial][:, k:k+1])
                        grfTrackingTerm_opt_all += (weights['grfTrackingTerm'] * grfTrackingTerm_opt * h * B[j + 1])                  
                    if trackGRM:
                        grmTrackingTerm_opt = f_NGRToTrackSum2(
                            (GRM_all_opt['all'][trial][:, k:k+1] / (scaling['GRM'][trial].to_numpy().T)) - 
                            grmToTrack_sc[trial][:, k:k+1])
                        grmTrackingTerm_opt_all += (weights['grmTrackingTerm'] * grmTrackingTerm_opt * h * B[j + 1])
                    if reserveActuators:
                        reserveActuatorTerm_opt = 0
                        for c_j in reserveActuatorJoints:                        
                            reserveActuatorTerm_opt += ca.sumsqr(rActk_opt[c_j])                            
                        reserveActuatorTerm_opt /= len(reserveActuatorJoints)
                        reserveActuatorTerm_opt_all += (weights['reserveActuatorTerm'] * reserveActuatorTerm_opt * h * B[j + 1])
                    if isSTSs_yCalcn_vGRF[trial] and not weights['vGRFRatioTerm'] == 0:
                        # TODO: clean this                        
                        vGRF_heel_r_opt = GRF_s_opt['r']['s1'][trial][1,k] + GRF_s_opt['r']['s4'][trial][1,k]
                        vGRF_front_r_opt = GRF_s_opt['r']['s2'][trial][1,k] + GRF_s_opt['r']['s3'][trial][1,k] + GRF_s_opt['r']['s5'][trial][1,k] + GRF_s_opt['r']['s6'][trial][1,k] 
                        vGRF_ratio_r_opt = np.sqrt(vGRF_front_r_opt/vGRF_heel_r_opt)                        
                        vGRF_heel_l_opt = GRF_s_opt['l']['s1'][trial][1,k] + GRF_s_opt['l']['s4'][trial][1,k]
                        vGRF_front_l_opt = GRF_s_opt['l']['s2'][trial][1,k] + GRF_s_opt['l']['s3'][trial][1,k] + GRF_s_opt['l']['s5'][trial][1,k] + GRF_s_opt['l']['s6'][trial][1,k] 
                        vGRF_ratio_l_opt = np.sqrt(vGRF_front_l_opt/vGRF_heel_l_opt)
                        vGRFRatioTerm_opt_all += (weights['vGRFRatioTerm'] * vGRF_ratio_l_opt * h * B[j + 1])
                        vGRFRatioTerm_opt_all += (weights['vGRFRatioTerm'] * vGRF_ratio_r_opt * h * B[j + 1])
                
        # Motor control term
        JMotor_opt = (activationTerm_opt_all.full() +  
                      jointAccelerationTerm_opt_all.full() +
                      activationDtTerm_opt_all.full() + 
                      forceDtTerm_opt_all.full())             
        if withArms:                
            JMotor_opt += armExcitationTerm_opt_all.full()
        if withActiveMTP:
              JMotor_opt += mtpExcitationTerm_opt_all.full()
        if withLumbarCoordinateActuators:
            JMotor_opt += lumbarExcitationTerm_opt_all.full()
        if isSTSs_yCalcn_vGRF[trial] and not weights['vGRFRatioTerm'] == 0:
            JMotor_opt += vGRFRatioTerm_opt_all            
              
        JTrack_opt = (positionTrackingTerm_opt_all.full() +  
                      velocityTrackingTerm_opt_all.full())
        if trackQdds:
            JTrack_opt += accelerationTrackingTerm_opt_all.full()
        if trackGRF:        
            JTrack_opt += grfTrackingTerm_opt_all.full()
        if trackGRM:   
            JTrack_opt += grmTrackingTerm_opt_all.full()
        if reserveActuators:    
            JTrack_opt += reserveActuatorTerm_opt_all
            
        # Combined term
        JAll_opt = JTrack_opt + JMotor_opt
        if stats['success']:
            assert np.alltrue(
                np.abs(JAll_opt[0][0] - stats['iterations']['obj'][-1]) 
                <= 1e-5), "decomposition cost"
        
        JTerms = {}
        JTerms["activationTerm"] = activationTerm_opt_all.full()[0][0]
        if withArms:
            JTerms["armExcitationTerm"] = armExcitationTerm_opt_all.full()[0][0]
        if withActiveMTP:
            JTerms["mtpExcitationTerm"] = mtpExcitationTerm_opt_all.full()[0][0]
        if withLumbarCoordinateActuators:
            JTerms["lumbarExcitationTerm"] = lumbarExcitationTerm_opt_all.full()[0][0]
        JTerms["jointAccelerationTerm"] = jointAccelerationTerm_opt_all.full()[0][0]
        JTerms["activationDtTerm"] = activationDtTerm_opt_all.full()[0][0]
        JTerms["forceDtTerm"] = forceDtTerm_opt_all.full()[0][0]
        JTerms["positionTerm"] = positionTrackingTerm_opt_all.full()[0][0]
        JTerms["velocityTerm"] = velocityTrackingTerm_opt_all.full()[0][0]
        if trackQdds:
            JTerms["accelerationTerm"] = accelerationTrackingTerm_opt_all.full()[0][0]    
        if trackGRF:
            JTerms["grfTerm"] = grfTrackingTerm_opt_all.full()[0][0]                
        if trackGRM:
            JTerms["grmTerm"] = grmTrackingTerm_opt_all.full()[0][0]                 
        JTerms["activationTerm_sc"] = JTerms["activationTerm"] / JAll_opt[0][0]
        if withArms:
            JTerms["armExcitationTerm_sc"] = JTerms["armExcitationTerm"] / JAll_opt[0][0]
        if withActiveMTP:
            JTerms["mtpExcitationTerm_sc"] = JTerms["mtpExcitationTerm"] / JAll_opt[0][0]
        if withLumbarCoordinateActuators:
            JTerms["lumbarExcitationTerm_sc"] = JTerms["lumbarExcitationTerm"] / JAll_opt[0][0]
        if trackGRF:
            JTerms["grfTerm_sc"] = JTerms["grfTerm"] / JAll_opt[0][0]                
        if trackGRM:
            JTerms["grmTerm_sc"] = JTerms["grmTerm"] / JAll_opt[0][0]
        JTerms["jointAccelerationTerm_sc"] = JTerms["jointAccelerationTerm"] / JAll_opt[0][0]
        JTerms["activationDtTerm_sc"] = JTerms["activationDtTerm"] / JAll_opt[0][0]
        JTerms["forceDtTerm_sc"] = JTerms["forceDtTerm"] / JAll_opt[0][0]
        JTerms["positionTerm_sc"] = JTerms["positionTerm"] / JAll_opt[0][0]
        JTerms["velocityTerm_sc"] = JTerms["velocityTerm"] / JAll_opt[0][0]
        if trackQdds:
            JTerms["accelerationTerm_sc"] = JTerms["accelerationTerm"] / JAll_opt[0][0]                
        
        print("CONTRIBUTION TO THE COST FUNCTION")
        print("Muscle activations: " + str(np.round(JTerms["activationTerm_sc"] * 100, 2)) + "%")
        if withArms:
            print("Arm Excitations: " + str(np.round(JTerms["armExcitationTerm_sc"] * 100, 2)) + "%")
        if withLumbarCoordinateActuators:
            print("Lumbar Excitations: " + str(np.round(JTerms["lumbarExcitationTerm_sc"] * 100, 2)) + "%")
        if withActiveMTP:
            print("MTP Excitations: " + str(np.round(JTerms["mtpExcitationTerm_sc"] * 100, 2)) + "%")
        print("Joint Accelerations: " + str(np.round(JTerms["jointAccelerationTerm_sc"] * 100, 2)) + "%")
        print("Muscle activations derivatives: " + str(np.round(JTerms["activationDtTerm_sc"] * 100, 2)) + "%")
        print("Muscle-tendon forces derivatives: " + str(np.round(JTerms["forceDtTerm_sc"] * 100, 2)) + "%")
        print("Position tracking: " + str(np.round(JTerms["positionTerm_sc"] * 100, 2)) + "%")
        print("Velocity tracking: " + str(np.round(JTerms["velocityTerm_sc"] * 100, 2)) + "%")
        if trackQdds:
            print("Acceleration tracking: " + str(np.round(JTerms["accelerationTerm_sc"] * 100, 2)) + "%")
        if trackGRF:
            print("GRF tracking: " + str(np.round(JTerms["grfTerm_sc"] * 100, 2)) + "%")
        if trackGRF:
            print("GRM tracking: " + str(np.round(JTerms["grmTerm_sc"] * 100, 2)) + "%")
            
        print("# Iterations: " + str(stats["iter_count"]))
        print("\n")
            
        # # %% Print new model
        # if optimizeContacts and printModel:            
        #     import opensim             
        #     model = opensim.Model(pathModelFile[:-5] + "_contacts.osim")           
        #     contactGeometrySet = model.getContactGeometrySet()            
            
        #     if NContactParameters == 18 and parameter_to_optimize == "option1":             
        #         for c_s in range(1, contactGeometrySet.getSize()):
        #             c_sphere = contactGeometrySet.get(c_s)
        #             cObj = opensim.ContactSphere.safeDownCast(c_sphere);
        #             if c_s > 6:
        #                 c_sphere.set_location(opensim.Vec3(p_opt_nsc[(c_s-6-1)*2], -0.01, -p_opt_nsc[(c_s-6-1)*2+1]))
        #                 cObj.setRadius(p_opt_nsc[c_s-6+11])
        #             else:
        #                 c_sphere.set_location(opensim.Vec3(p_opt_nsc[(c_s-1)*2], -0.01, p_opt_nsc[(c_s-1)*2+1]))
        #                 cObj.setRadius(p_opt_nsc[c_s+11])                         
        #         # forceSet = model.getForceSet()
        #         # for c_f in range(forceSet.getSize()): 
        #         #     c_force_elt = forceSet.get(c_f)        
        #         #     if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":                    
        #         #         c_force_elt_obj =  opensim.SmoothSphereHalfSpaceForce.safeDownCast(c_force_elt)                     
        #         #         c_force_elt_obj.set_stiffness(p_opt_nsc[18])
        #         #         c_force_elt_obj.set_dissipation(p_opt_nsc[19])  
                        
        #     model.finalizeConnections
        #     model.initSystem()
        #     model.printToXML(os.path.join(pathResults, "{}_contacts_optimized.osim".format(model_full_name)))
            
            
        # %% Compute KAM
        if computeKAM:            
            from computeJointLoading import computeKAM
            from utilsOpenSimAD import interpolateNumpyArray_time  
            KAM_labels = ['KAM_r', 'KAM_l']
            KAM, KAM_ref = {}, {}
            for trial in trials:
                # Simulations
                IDPath = os.path.join(
                    pathResults, 'kinetics_{}_{}.mot'.format(trial, case))
                IKPath = os.path.join(
                    pathResults, 
                    'kinematics_act_{}_{}.mot'.format(trial, case))
                GRFPath = os.path.join(
                    pathResults, 'GRF_{}_{}.mot'.format(trial, case))
                c_KAM = computeKAM(pathResults, pathModelFile, IDPath, 
                                   IKPath, GRFPath, grfType='sphere',
                                   Qds=Qds_opt_nsc[trial].T)
                KAM[trial] = np.concatenate(
                    (np.expand_dims(c_KAM['KAM_r'], axis=1),
                     np.expand_dims(c_KAM['KAM_l'], axis=1)), axis=1).T                
                # Experimental                
                trial_exp =  trial
                if 'videoAndMocap' in trial:
                    trial_exp = trial[:-14]
                IKPathMocap = os.path.join(pathIKFolderMocap, 
                                           trial_exp + '.mot')
                IDPathMocap = os.path.join(pathIDFolder, 
                                           trial_exp + '.sto')
                GRFPathMocap = os.path.join(pathGRFFolder, 
                                            trial_exp + '_forces.mot')
                c_KAM_ref = computeKAM(pathResults, pathModelFileMocap,
                                       IDPathMocap, IKPathMocap, 
                                       GRFPathMocap, grfType='experimental')
                KAM_ref_r_interp = interpolateNumpyArray_time(
                    c_KAM_ref['KAM_r'], c_KAM_ref['time'], 
                    timeIntervals[trial][0], timeIntervals[trial][1], N[trial])
                KAM_ref_l_interp = interpolateNumpyArray_time(
                    c_KAM_ref['KAM_l'], c_KAM_ref['time'], 
                    timeIntervals[trial][0], timeIntervals[trial][1], N[trial])
                KAM_ref[trial] = np.concatenate(
                    (np.expand_dims(KAM_ref_r_interp, axis=1),
                     np.expand_dims(KAM_ref_l_interp, axis=1)), axis=1).T                
                
        # %% Compute MCF
        if computeMCF:
            # Export muscle forces and non muscle-driven torques (if existing).
            import pandas as pd
            for trial in trials:
                labels = ['time'] 
                labels += bothSidesMuscles
                data = np.concatenate((tgridf[trial].T[:-1], Ft_opt[trial].T),axis=1)
                # Add non muscle-driven torques (reserve actuators, limit torques,
                # torques for torque-driven muscles, passive torques).
                labels_torques = []
                data_torques = pd.DataFrame()
                if reserveActuators:
                    for count_j, c_j in enumerate(reserveActuatorJoints):                    
                        if c_j in data_torques:
                            data_torques[c_j] += rAct_opt_nsc[trial][c_j]
                        else:
                            labels_torques.append(c_j)
                            data_torques.insert(data_torques.shape[1], c_j, rAct_opt_nsc[trial][c_j])
                if enableLimitTorques:
                    for count_j, c_j in enumerate(passiveTorqueJoints):
                        if c_j in data_torques:
                            data_torques[c_j] += pT_opt[trial][count_j,:]
                        else:
                            labels_torques.append(c_j)
                            data_torques.insert(data_torques.shape[1], c_j, pT_opt[trial][count_j,:])
                if withLumbarCoordinateActuators:
                    for count_j, c_j in enumerate(lumbarJoints):
                        aLumbar_opt_nsc = scaling['LumbarE'][trial].iloc[0][c_j] * aLumbar_opt[trial][count_j,:-1]
                        if c_j in data_torques:
                            data_torques[c_j] += aLumbar_opt_nsc
                        else:
                            labels_torques.append(c_j)
                            data_torques.insert(data_torques.shape[1], c_j, aLumbar_opt_nsc)
                        assert np.alltrue(
                                np.abs(torques_opt[trial][joints.index(c_j),:] - data_torques[c_j]) 
                                < 10**(-4)), "error forces lumbar joint"
                if withArms:
                    for count_j, c_j in enumerate(armJoints):
                        aArm_opt_nsc = scaling['ArmE'][trial].iloc[0][c_j] * aArm_opt[trial][count_j,:-1]
                        c_torque = linearPassiveTorqueArms_opt[trial][count_j,:-1]
                        if c_j in data_torques:
                            data_torques[c_j] += (aArm_opt_nsc + c_torque)
                        else:
                            labels_torques.append(c_j)
                            data_torques.insert(data_torques.shape[1], c_j, aArm_opt_nsc + c_torque)
                        assert np.alltrue(
                                np.abs(torques_opt[trial][joints.index(c_j),:] - data_torques[c_j]) 
                                < 10**(-4)), "error forces arm joints"
                if withMTP:
                    if withActiveMTP:
                        raise ValueError("Not supported yet: TODO")                    
                    for count_j, c_j in enumerate(mtpJoints):
                        c_torque = linearPassiveTorqueMtp_opt[trial][count_j,:-1]
                        if c_j in data_torques:
                            data_torques[c_j] += c_torque
                        else:
                            labels_torques.append(c_j)
                            data_torques.insert(data_torques.shape[1], c_j, c_torque)
                        assert np.alltrue(
                                np.abs(torques_opt[trial][joints.index(c_j),:] - data_torques[c_j]) 
                                < 10**(-4)), "error forces mtp joints"
                # Sanity check for muscle-driven joints
                for count_j, c_j in enumerate(muscleDrivenJoints):
                    assert np.alltrue(
                            np.abs(torques_opt[trial][joints.index(c_j),:] - (
                                data_torques[c_j].to_numpy() + pMT_opt[trial][count_j, :] + 
                                aMT_opt[trial][count_j, :])) 
                            < 10**(-3)), "error forces muscle-driven joints"
                data_torques_np = data_torques.to_numpy()
                if len(data_torques) > 0:
                    data = np.concatenate((data, data_torques_np),axis=1)
                    labels += labels_torques
                numpy_to_storage(labels, data, os.path.join(
                    pathResults, 'forces_{}_{}.mot'.format(trial, case)),
                    datatype='muscle_forces')
                
            from computeJointLoading import computeMCF
            MCF_labels = ['MCF_r', 'MCF_l']
            MCF, MCF_ref = {}, {}
            for trial in trials:
                # Simulations
                forcePath = os.path.join(
                    pathResults, 
                    'forces_{}_{}.mot'.format(trial, case))
                IK_act_Path = os.path.join(
                    pathResults, 
                    'kinematics_act_{}_{}.mot'.format(trial, case))
                GRFPath = os.path.join(
                    pathResults, 'GRF_{}_{}.mot'.format(trial, case))                
                c_MCF = computeMCF(pathResults, pathModelFile, IK_act_Path, 
                                    IK_act_Path, GRFPath, grfType='sphere',
                                    muscleForceFilePath=forcePath,
                                    pathReserveGeneralizedForces=forcePath,
                                    Qds=Qds_opt_nsc[trial].T,
                                    replaceMuscles=True)
                MCF[trial] = np.concatenate(
                    (np.expand_dims(c_MCF['MCF_r'], axis=1),
                     np.expand_dims(c_MCF['MCF_l'], axis=1)), axis=1).T
                # Experimental from StaticOpt             
                trial_exp =  trial
                if 'videoAndMocap' in trial:
                    trial_exp = trial[:-14]
                pathJRAfromSO = os.path.join(
                    pathJRAfromSOFolderMocap,
                    trial_exp + '_JointReaction_ReactionLoads.sto')
                c_MCF_ref = computeMCF(None,None,None,None,None,None,
                                       pathJRAResults=pathJRAfromSO)                  
                MCF_ref_r_interp = interpolateNumpyArray_time(
                    c_MCF_ref['MCF_r'], c_MCF_ref['time'], 
                    timeIntervals[trial][0], timeIntervals[trial][1], N[trial])
                MCF_ref_l_interp = interpolateNumpyArray_time(
                    c_MCF_ref['MCF_l'], c_MCF_ref['time'], 
                    timeIntervals[trial][0], timeIntervals[trial][1], N[trial])
                MCF_ref[trial] = np.concatenate(
                    (np.expand_dims(MCF_ref_r_interp, axis=1),
                     np.expand_dims(MCF_ref_l_interp, axis=1)), axis=1).T
                
        # %% Filter GRFs
        # Cutoff frequency same as the one used to filter Qs.
        from utilsOpenSimAD import filterNumpyArray
        GRF_all_opt_filt = {}
        GRF_all_opt_filt['all'] = {}
        for trial in trials:            
            GRF_all_opt_filt['all'][trial] = filterNumpyArray(
                GRF_all_opt['all'][trial].T, tgridf[trial][0,:-1], 
                cutoff_frequency=cutoff_freq_coords[trial]).T
                
        # %% Express in %BW (GRF) and %BW*height (torques)
        gravity = 9.80665
        BW = subject_demo['mass_kg'] * gravity
        BW_ht = BW * subject_demo['height_m']
        
        GRF_BW_all_opt, GRF_BW_all_opt_filt, GRM_BWht_all_opt = {}, {}, {}
        GRF_BW_all_opt['all'], GRM_BWht_all_opt['all'] = {}, {}        
        torques_BWht_opt, grfToTrack_BW_nsc, GRF_peaks_BW = {}, {}, {}
        grmToTrack_BWht_nsc, ID_BWht_toTrack = {}, {}
        KAM_BWht, KAM_BWht_ref, GRF_BW_all_opt_filt['all'] = {}, {}, {}
        MCF_BW, MCF_BW_ref = {}, {}
        for trial in trials:
            GRF_BW_all_opt['all'][trial] = GRF_all_opt['all'][trial] / BW * 100
            GRF_BW_all_opt_filt['all'][trial] = GRF_all_opt_filt['all'][trial] / BW * 100
            GRM_BWht_all_opt['all'][trial] = GRM_all_opt['all'][trial] / BW_ht * 100            
            torques_BWht_opt[trial] = torques_opt[trial] / BW_ht * 100
            if GRF_toTrack:
                grfToTrack_BW_nsc[trial] = grfToTrack_nsc[trial] / BW * 100
                grmToTrack_BWht_nsc[trial] = grmToTrack_nsc[trial] / BW_ht * 100
                GRF_peaks_BW[trial] = {}
                for side in sides_all:
                    GRF_peaks_BW[trial][side] = GRF_peaks[trial][side] / BW * 100
            if ID_toTrack:
                ID_BWht_toTrack[trial] = ID_toTrack[trial] / BW_ht * 100
            if computeKAM:
                KAM_BWht[trial] = KAM[trial] / BW_ht * 100
                KAM_BWht_ref[trial] = KAM_ref[trial] / BW_ht * 100
            if computeMCF:
                MCF_BW[trial] = MCF[trial] / BW * 100
                MCF_BW_ref[trial] = MCF_ref[trial] / BW * 100
            
        # %% Save trajectories for further analysis
        # pathTrajectories = os.path.join(baseDir, "data")
        if not os.path.exists(os.path.join(pathResults,
                                            'optimaltrajectories.npy')): 
                optimaltrajectories = {}
        else:  
            optimaltrajectories = np.load(
                    os.path.join(pathResults, 'optimaltrajectories.npy'),
                    allow_pickle=True)   
            optimaltrajectories = optimaltrajectories.item()
        for trial in trials:
            optimaltrajectories[case] = {
                'coordinate_values_toTrack': {trial: refData_offset_nsc[trial]},
                'coordinate_values': {trial: Qs_opt_nsc[trial]},
                'coordinate_speeds_toTrack': {trial: refData_dot_nsc[trial]},
                'coordinate_speeds': {trial: Qds_opt_nsc[trial]}, 
                'coordinate_accelerations_toTrack': {trial: refData_dotdot_nsc[trial]},
                'coordinate_accelerations': {trial: Qdds_col_opt_nsc[trial]},
                'torques': {trial: torques_opt[trial]},
                'torques_BWht': {trial: torques_BWht_opt[trial]},
                'GRF': {trial: GRF_all_opt['all'][trial]},
                'GRF_BW': {trial: GRF_BW_all_opt['all'][trial]},
                'GRF_filt': {trial: GRF_all_opt_filt['all'][trial]},
                'GRF_filt_BW': {trial: GRF_BW_all_opt_filt['all'][trial]},
                'GRM': {trial: GRM_all_opt['all'][trial]},
                'GRM_BWht': {trial: GRM_BWht_all_opt['all'][trial]},
                'joints': joints,
                'rotationalJoints': rotationalJoints,
                'GRF_labels': GR_labels_fig,
                'time': {trial: tgridf[trial]},
                'muscle_activations': {trial: a_opt[trial]},
                'muscles': bothSidesMuscles,
                'passive_muscle_torques': {trial: pMT_opt[trial]},
                'active_muscle_torques': {trial: aMT_opt[trial]},
                'passive_limit_torques': {trial: pT_opt[trial]},
                'muscle_driven_joints': muscleDrivenJoints,
                'limit_torques_joints': passiveTorqueJoints}
            if GRF_toTrack:
                optimaltrajectories[case]['GRF_ref'] = {trial: grfToTrack_nsc[trial]}
                optimaltrajectories[case]['GRF_BW_ref'] = {trial: grfToTrack_BW_nsc[trial]}
                optimaltrajectories[case]['GRM_ref'] = {trial: grmToTrack_nsc[trial]}
                optimaltrajectories[case]['GRM_BWht_ref'] = {trial: grmToTrack_BWht_nsc[trial]}
                optimaltrajectories[case]['GRF_ref_peaks'] = {trial: GRF_peaks[trial]}
                optimaltrajectories[case]['GRF_BW_ref_peaks'] = {trial: GRF_peaks_BW[trial]}                
            if ID_toTrack:
                optimaltrajectories[case]['torques_ref'] = {trial: ID_toTrack[trial]}
                optimaltrajectories[case]['torques_BWht_ref'] = {trial: ID_BWht_toTrack[trial]}
            if Qs_mocap_ref:
                optimaltrajectories[case]['coordinate_values_ref'] = {trial: Qs_mocap_ref[trial]}
                optimaltrajectories[case]['coordinate_speeds_ref'] = {trial: Qds_mocap_ref[trial]}
                optimaltrajectories[case]['coordinate_accelerations_ref'] = {trial: Qdds_mocap_ref[trial]}                
            if computeKAM:
                optimaltrajectories[case]['KAM'] = {trial: KAM[trial]}
                optimaltrajectories[case]['KAM_BWht'] = {trial: KAM_BWht[trial]}
                optimaltrajectories[case]['KAM_ref'] = {trial: KAM_ref[trial]}
                optimaltrajectories[case]['KAM_BWht_ref'] = {trial: KAM_BWht_ref[trial]}
                optimaltrajectories[case]['KAM_labels'] = {trial: KAM_labels}
            if computeMCF:
                optimaltrajectories[case]['MCF'] = {trial: MCF[trial]}
                optimaltrajectories[case]['MCF_BW'] = {trial: MCF_BW[trial]}
                optimaltrajectories[case]['MCF_ref'] = {trial: MCF_ref[trial]}
                optimaltrajectories[case]['MCF_BW_ref'] = {trial: MCF_BW_ref[trial]}
                optimaltrajectories[case]['MCF_labels'] = {trial: MCF_labels}
            if EMG_ref:
                optimaltrajectories[case]['muscle_activations_ref'] = {trial: EMG_ref[trial]}
            if SO_toTrack:
                optimaltrajectories[case]['so_ref'] = {trial: SO_toTrack[trial]}
                
        optimaltrajectories[case]['iter'] = stats['iter_count']
                
        np.save(os.path.join(pathResults, 'optimaltrajectories.npy'),
                optimaltrajectories)
            
        # %% Visualize results against bounds
        if visualizeResultsAgainstBounds:
            from utilsOpenSimAD import plotVSBounds
            for trial in trials:
                # States
                # Muscle activation at mesh points
                lwp = lw['A'][trial].to_numpy().T
                uwp = uw['A'][trial].to_numpy().T
                y = a_opt[trial]
                title='Muscle activation at mesh points'            
                plotVSBounds(y,lwp,uwp,title)  
                # Muscle activation at collocation points
                lwp = lw['A'][trial].to_numpy().T
                uwp = uw['A'][trial].to_numpy().T
                y = a_col_opt[trial]
                title='Muscle activation at collocation points' 
                plotVSBounds(y,lwp,uwp,title)  
                # Muscle force at mesh points
                lwp = lw['F'][trial].to_numpy().T
                uwp = uw['F'][trial].to_numpy().T
                y = nF_opt[trial]
                title='Muscle force at mesh points' 
                plotVSBounds(y,lwp,uwp,title)  
                # Muscle force at collocation points
                lwp = lw['F'][trial].to_numpy().T
                uwp = uw['F'][trial].to_numpy().T
                y = nF_col_opt[trial]
                title='Muscle force at collocation points' 
                plotVSBounds(y,lwp,uwp,title)
                # Joint position at mesh points
                lwp = lw['Qs'][trial].to_numpy().T
                uwp = uw['Qs'][trial].to_numpy().T
                y = Qs_opt[trial]
                title='Joint position at mesh points' 
                plotVSBounds(y,lwp,uwp,title)             
                # Joint position at collocation points
                lwp = lw['Qs'][trial].to_numpy().T
                uwp = uw['Qs'][trial].to_numpy().T
                y = Qs_col_opt[trial]
                title='Joint position at collocation points' 
                plotVSBounds(y,lwp,uwp,title) 
                # Joint velocity at mesh points
                lwp = lw['Qds'][trial].to_numpy().T
                uwp = uw['Qds'][trial].to_numpy().T
                y = Qds_opt[trial]
                title='Joint velocity at mesh points' 
                plotVSBounds(y,lwp,uwp,title) 
                # Joint velocity at collocation points
                lwp = lw['Qds'][trial].to_numpy().T
                uwp = uw['Qds'][trial].to_numpy().T
                y = Qds_col_opt[trial]
                title='Joint velocity at collocation points' 
                plotVSBounds(y,lwp,uwp,title) 
                if withActiveMTP:
                    # Mtp activation at mesh points
                    lwp = lw['MtpA'][trial].to_numpy().T
                    uwp = uw['MtpA'][trial].to_numpy().T
                    y = aMtp_opt[trial]
                    title='Mtp activation at mesh points' 
                    plotVSBounds(y,lwp,uwp,title) 
                    # Mtp activation at collocation points
                    lwp = lw['MtpA'][trial].to_numpy().T
                    uwp = uw['MtpA'][trial].to_numpy().T
                    y = aMtp_col_opt[trial]
                    title='Mtp activation at collocation points' 
                    plotVSBounds(y,lwp,uwp,title) 
                #######################################################################
                # Controls
                # Muscle activation derivative at mesh points
                lwp = lw['ADt'][trial].to_numpy().T
                uwp = uw['ADt'][trial].to_numpy().T
                y = aDt_opt[trial]
                title='Muscle activation derivative at mesh points' 
                plotVSBounds(y,lwp,uwp,title) 
                if withActiveMTP:
                    # Mtp excitation at mesh points
                    lwp = lw['MtpE'][trial].to_numpy().T
                    uwp = uw['MtpE'][trial].to_numpy().T
                    y = eMtp_opt[trial]
                    title='Mtp excitation at mesh points' 
                    plotVSBounds(y,lwp,uwp,title)                
                #######################################################################
                # Slack controls
                # Muscle force derivative at collocation points
                lwp = lw['FDt'][trial].to_numpy().T
                uwp = uw['FDt'][trial].to_numpy().T
                y = nFDt_col_opt[trial]
                title='Muscle force derivative at collocation points' 
                plotVSBounds(y,lwp,uwp,title)
                # Joint velocity derivative (acceleration) at collocation points
                lwp = lw['Qdds'][trial].to_numpy().T
                uwp = uw['Qdds'][trial].to_numpy().T
                y = Qdds_col_opt[trial]
                title='Joint velocity derivative (acceleration) at collocation points' 
                plotVSBounds(y,lwp,uwp,title)
                
                # if optimizeContacts:
                #     from variousFunctions import scatterVSBounds
                #     uwp = ((uw['ContactParameters'][trial] - 
                #             scaling['ContactParameters_r'][trial]) / 
                #            scaling['ContactParameters_v'][trial]).flatten()
                #     lwp = ((lw['ContactParameters'][trial] - 
                #             scaling['ContactParameters_r'][trial]) / 
                #            scaling['ContactParameters_v'][trial]).flatten()
                #     y = p_opt_nsc
                #     title='Contact parameters'
                #     scatterVSBounds(y,lwp,uwp,title)
