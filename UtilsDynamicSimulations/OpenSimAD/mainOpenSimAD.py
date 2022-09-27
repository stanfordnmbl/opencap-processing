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
import platform

# %% Settings.
def run_tracking(baseDir, dataDir, subject, settings, case='0',
                 solveProblem=True, analyzeResults=True, writeGUI=True,
                 visualizeTracking=False, visualizeResultsBounds=False,
                 computeKAM=False, computeMCF=False):
    
    import copy # TODO
    
    # %% Settings.
    # Most available settings are left from trying out different formulations 
    # when solving the trajectory optimization problems. We decided to keep
    # them in this script in case users want to play with them or use them as
    # examples for creating their own formulations.    
    
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
    # Model name.
    OpenSimModel = 'LaiArnoldModified2017_poly_withArms_weldHand'
    if 'OpenSimModel' in settings:  
        OpenSimModel = settings['OpenSimModel']
    model_full_name = OpenSimModel + "_scaled_adjusted"
    
    # Set withMTP to True to include metatarshophalangeal joints.
    withMTP = True
    if 'withMTP' in settings:  
        withMTP = settings['withMTP']
    
    # Set withArms to True to include arm joints.
    withArms = True
    if "withArms" in settings:
         withArms = settings['withArms']
    
    # Set withLumbarCoordinateActuators to True to actuate the lumbar 
    # coordinates with coordinate actuators. Coordinate actuator have simple
    # dynamics to relate excitations and activations. The maximal excitations
    # and activations are given in boundsOpenSimAD. The excitations of the
    # coordinate actuators are minimized in the cost function with a weight
    # given by weights['lumbarExcitationTerm'].
    withLumbarCoordinateActuators = True
    if "withLumbarCoordinateActuators" in settings:
         withLumbarCoordinateActuators = (
             settings['withLumbarCoordinateActuators'])
    if withLumbarCoordinateActuators:   
         weights['lumbarExcitationTerm'] = 1
         if 'lumbarExcitationTerm' in settings['weights']:
            weights['lumbarExcitationTerm'] = (
                settings['weights']['lumbarExcitationTerm'])
    
    # Set scaleIsometricMuscleForce to scale the maximal isometric muscle
    # forces. By default, scaleIsometricMuscleForce is set to 1, and the forces
    # therefore correspond to those of the generic models.
    scaleIsometricMuscleForce = 1
    if 'scaleIsometricMuscleForce' in settings: 
        scaleIsometricMuscleForce = settings['scaleIsometricMuscleForce']
    
    # Set withReserveActuators to True to include reserve actuators. Reserve
    # actuators will be added for coordinates specified in
    # reserveActuatorCoordinates. Eg, 
    # 'reserveActuatorCoordinates'['hip_rotation_l'] = 30 will add a reserve
    # actuator to the hip_rotation_l with a maximal value of 30. The reserve
    # actuators are minimized in the cost function with a weight given by
    # weights['reserveActuatorTerm']
    withReserveActuators = False
    if 'withReserveActuators' in settings: 
        withReserveActuators = settings['withReserveActuators']
    if withReserveActuators:
        reserveActuatorCoordinates = settings['reserveActuatorCoordinates']
        weights['reserveActuatorTerm'] = (
            settings['weights']['reserveActuatorTerm'])
    
    # Set ignorePassiveFiberForce to True to ignore passive muscle forces.
    ignorePassiveFiberForce = False
    if 'ignorePassiveFiberForce' in settings: 
        ignorePassiveFiberForce = settings['ignorePassiveFiberForce']
     
    # lb_activation determines the lower bound on muscle activations.
    lb_activation = 0.01
    if 'lb_activation' in settings:
        lb_activation = settings['lb_activation']
        
    # Trials info.    
    trials = settings['trials']
    timeIntervals, timeElapsed, tgridf, N, treadmill = {}, {}, {}, {}, {}
    heel_vGRF_threshold, min_ratio_vGRF = {}, {}
    yCalcnToes, yCalcnToesThresholds = {}, {}
    filter_Qs_toTrack, cutoff_freq_Qs, splineQds = {}, {}, {}
    filter_Qds_toTrack, cutoff_freq_Qds = {}, {}
    filter_Qdds_toTrack, cutoff_freq_Qdds = {}, {}
    for trial in trials:        
        # Time interval.
        timeIntervals[trial] = trials[trial]['timeInterval']
        timeElapsed[trial] = timeIntervals[trial][1] - timeIntervals[trial][0]
        
        # Mesh intervals / density.
        if 'N' in trials[trial]: # number of mesh intervals.
            N[trial] = trials[trial]['N']
        else:
            meshDensity = 100 # default is N=100 for t=1s
            if 'meshDensity' in trials[trial]:
                meshDensity = trials[trial]['meshDensity']
            N[trial] = int(round(timeElapsed[trial] * meshDensity, 2))
            
        # Discretized time interval.
        tgrid = np.linspace(timeIntervals[trial][0],  
                            timeIntervals[trial][1], N[trial]+1)
        tgridf[trial] = np.zeros((1, N[trial]+1))
        tgridf[trial][:,:] = tgrid.T            
            
        # If heel_vGRF_threshold is larger than 0, a constraint will enforce
        # that the contact spheres on the model's heels generate a vertical
        # ground reaction force larger than heel_vGRF_threshold. This is used
        # to force the model to keep its heels in contact with the ground.
        heel_vGRF_threshold[trial] = 0
        if 'heel_vGRF_threshold' in trials[trial]:
            heel_vGRF_threshold[trial] = trials[trial]['heel_vGRF_threshold']
            
        # If min_ratio_vGRF is True and weights['vGRFRatioTerm'] is 
        # larger than 0, a cost term will be added to the cost function to
        # minimize the ratio between the vertical ground reaction forces of the
        # front contact spheres over the vertical ground reaction forces of the
        # rear (heel) contact spheres. This might be used to discourage the
        # model to lean forward, which would reduce muscle effort.
        min_ratio_vGRF[trial] = False
        if 'min_ratio_vGRF' in trials[trial]:
           min_ratio_vGRF[trial] = trials[trial]['min_ratio_vGRF']
        if min_ratio_vGRF[trial]: 
            weights['vGRFRatioTerm'] = 1
            if 'vGRFRatioTerm' in settings['weights']:
                weights['vGRFRatioTerm'] = settings['weights']['vGRFRatioTerm']
              
        # If yCalcnToes is set to True, a constraint will enforce the 
        # vertical position of the origin of the calcaneus and toe segments
        # to be larger than yCalcnToesThresholds. This is used to prevent the
        # model to penetrate the ground, which might otherwise occur at the
        # begnning of the trial when no periodic constraints are enforced.
        yCalcnToes[trial] = False
        if 'yCalcnToes' in trials[trial]:
            yCalcnToes[trial] = trials[trial]['yCalcnToes']            
        if yCalcnToes[trial] :
            yCalcnToesThresholds[trial] = 0.015
            if 'yCalcnToesThresholds' in trials[trial]:
                yCalcnToesThresholds[trial] = (
                    trials[trial]['yCalcnToesThresholds'])
                
        # Set filter_Qs_toTrack to True to filter the coordinate values to be
        # tracked with a cutoff frequency of cutoff_freq_Qs.
        filter_Qs_toTrack[trial] = True
        if 'filter_Qs_toTrack' in trials[trial]:
            filter_Qs_toTrack[trial] = trials[trial]['filter_Qs_toTrack']
        if filter_Qs_toTrack[trial]:
            cutoff_freq_Qs[trial] = 30 # default.
            if 'cutoff_freq_Qs' in trials[trial]:
                cutoff_freq_Qs[trial] = trials[trial]['cutoff_freq_Qs']
             
        # Set filter_Qds_toTrack to True to filter the coordinate speeds to be
        # tracked with a cutoff frequency of cutoff_freq_Qds.
        filter_Qds_toTrack[trial] = False
        if 'filter_Qds_toTrack' in trials[trial]:
            filter_Qds_toTrack[trial] =  trials[trial]['filter_Qds_toTrack']
        if filter_Qds_toTrack[trial]:
            cutoff_freq_Qds[trial] = 30 # default.
            if 'cutoff_freq_Qds' in trials[trial]:
                cutoff_freq_Qds[trial] = trials[trial]['cutoff_freq_Qds']
          
        # Set filter_Qdds_toTrack to True to filter the coordinate
        # accelerations to be tracked with a cutoff frequency of
        # cutoff_freq_Qdds.
        filter_Qdds_toTrack[trial] = False
        if 'filter_Qdds_toTrack' in trials[trial]:
            filter_Qdds_toTrack[trial] = trials[trial]['filter_Qdds_toTrack']
        if filter_Qdds_toTrack[trial]:
            cutoff_freq_Qdds[trial] = 30 # default.
            if 'cutoff_freq_Qdds' in trials[trial]:
                cutoff_freq_Qdds[trial] = trials[trial]['cutoff_freq_Qdds']
             
        # Set splineQds to True to compute the coordinate accelerations by
        # first splining the coordinate speeds and then taking the derivative.
        # The default approach is to spline the coordinate values and then 
        # take the second derivative. It might be useful to first filter the
        # coordinate speeds using filter_Qds_toTrack and then set splineQds
        # to True to obtain smoother coordinate accelerations.
        splineQds[trial] = False
        if 'splineQds' in trials[trial]:
            splineQds[trial] = trials[trial]['splineQds']
            
        # Set treadmill to True to simulate a treadmill motion. The treadmill
        # speed is given by treadmill_speed (positive is forward motion).
        treadmill[trial] = False
        if trials[trial]['treadmill_speed'] != 0:
            treadmill[trial] = True
    
    # Problem info.
    # Coordinates to track.
    coordinates_toTrack = settings['coordinates_toTrack']
    
    # Set offset_ty to True to include an optimization variable in the problem
    # that will offset the vertical pelvis position (pelvis_ty). This is used
    # when there is uncertainty in the size of the foot ground contact spheres,
    # which makes hard accurately tracking pelvis_ty without such offset.
    offset_ty = True
    if 'offset_ty' in settings:
        offset_ty = settings['offset_ty']
        
    # Set enableLimitTorques to True to include limit torques. This is used
    # to model the effect of ligaments at the extremes of the range of motion.
    # See passiveJointTorqueData_3D in muscleDataOpenSimAD for values.
    enableLimitTorques = False
    if 'enableLimitTorques' in settings:
        enableLimitTorques = settings['enableLimitTorques']    
        
    # Set periodicConstraints to True to include periodic constraints. These
    # constraints typically facilitate convergence, and we encourage users to
    # use them when appropriate. See code below for details about how to set
    # up periodic constraints for the problem states. See settingsOpenSimAD
    # for examples.
    periodicConstraints = False
    if 'periodicConstraints' in settings:
        periodicConstraints = settings['periodicConstraints']
        
    # Set trackQdds to track coordinate accelerations. We found this useful to
    # improve the convergence of the problem. The weight associated with the
    # cost term is given by weights['accelerationTrackingTerm'].
    trackQdds = True
    if 'trackQdds' in settings:
        trackQdds = settings['trackQdds']
    if trackQdds:
        weights['accelerationTrackingTerm'] = 1
        if 'accelerationTrackingTerm' in settings['weights']:
            weights['accelerationTrackingTerm'] = (
                settings['weights']['accelerationTrackingTerm'])
            
    # The variable powActivations determines the power at which muscle
    # activations are minimized. By default, we use a power of 2, but
    # researchers have used other powers in the past (eg, Ackermann and 
    # van den Bogert, 2010).
    powActivations = 2
    if 'powActivations' in settings:
        powActivations = settings['powActivations']
        
    # Set volumeScaling to True to scale individual muscle contributions to
    # the muscle effort term by their volume.
    volumeScaling = False
    if 'volumeScaling' in settings:
        volumeScaling = settings['volumeScaling']
        
    # Set coordinate_constraints to bound the joint coordinates (states) based
    # on the experimental data at each mesh point (time-based bounds). Eg,
    # coordinate_constraints['pelvis_tx'] = {"env_bound": 0.1} will bound the
    # design variable corresponding to the pelvis_tx value by 
    # "experimental Q value to be tracked +/- 0.1 m" at each mesh point;
    # coordinate_constraints['hip_flexion_l'] = {"env_bound": 0.1} will bound
    # the design variable corresponding to the hip_flexion_l value by 
    # "experimental Q value to be tracked +/- 0.1 rad" at each mesh point. This
    # is an experimental feature that might help convergence by reducing the
    # search space of the optimization problem. See settingsOpenSimAD for
    # examples.
    coordinate_constraints = {}
    if 'coordinate_constraints' in settings:
        coordinate_constraints = settings['coordinate_constraints']
        
    # Type of initial guesses. Options are dataDriven and quasiRandom (TODO 
    # not used in a long time, not sure it still working). We recommed using
    # dataDriven, which is default.
    type_guess = "dataDriven"
    if 'type_guess' in settings:
        type_guess = settings['type_guess']
        
    # Convergence tolerance of ipopt: 
    # See https://coin-or.github.io/Ipopt/OPTIONS.html for more details.
    # We recommend testing different tolerances to make sure the results are 
    # not impacted by too loose tolerances. In the examples, we did not find
    # differences between using a tolerance of 3 and 4, and therefore set it
    # to 3, since it converges faster. Nevertheless, we set it here to 4 by
    # default.
    ipopt_tolerance = 4
    if 'ipopt_tolerance' in settings:
        ipopt_tolerance = settings['ipopt_tolerance']

    # %% Paths and dirs.
    pathOSData = os.path.join(dataDir, subject, 'OpenSimData')
    pathModelFolder = os.path.join(pathOSData, 'Model')
    pathModelFile = os.path.join(pathModelFolder, model_full_name + ".osim")
    pathExternalFunctionFolder = os.path.join(pathModelFolder,
                                              'ExternalFunction')
    pathIKFolder = os.path.join(pathOSData, 'Kinematics')
    trials_list = [trial for trial in trials]
    listToStr = '_'.join([str(elem) for elem in trials_list])
    pathResults = os.path.join(pathOSData, 'Dynamics', listToStr)
    if 'repetition' in settings:
        pathResults = os.path.join(
            pathOSData, 'Dynamics', 
            listToStr + '_rep' + str(settings['repetition']))     
    os.makedirs(pathResults, exist_ok=True)
    pathSettings = os.path.join(pathResults, 'Setup_{}.yaml'.format(case))
    # Dump settings in yaml file.
    with open(pathSettings, 'w') as file:
        yaml.dump(settings, file)
    
    # %% Muscles.
    # This section specifies the muscles and some of their parameters. This is
    # specific to the Rajagopal musculoskeletal model.
    #
    # WARNING: we do not use the muscle model defined in the .osim file. We
    # use our own muscle model based on De Groote et al. 2016:
    # https://pubmed.ncbi.nlm.nih.gov/27001399/. We only extract the muscle-
    # tendon parameters from the .osim file.
    muscles = [
        'addbrev_r', 'addlong_r', 'addmagDist_r', 'addmagIsch_r', 
        'addmagMid_r', 'addmagProx_r', 'bflh_r', 'bfsh_r', 'edl_r', 'ehl_r', 
        'fdl_r', 'fhl_r', 'gaslat_r', 'gasmed_r', 'glmax1_r', 'glmax2_r',
        'glmax3_r', 'glmed1_r', 'glmed2_r', 'glmed3_r', 'glmin1_r', 'glmin2_r',
        'glmin3_r', 'grac_r', 'iliacus_r', 'perbrev_r', 'perlong_r', 'piri_r', 
        'psoas_r', 'recfem_r', 'sart_r', 'semimem_r', 'semiten_r', 'soleus_r',
        'tfl_r', 'tibant_r', 'tibpost_r', 'vasint_r', 'vaslat_r', 'vasmed_r']
    rightSideMuscles = muscles        
    leftSideMuscles = [muscle[:-1] + 'l' for muscle in rightSideMuscles]
    bothSidesMuscles = leftSideMuscles + rightSideMuscles
    nMuscles = len(bothSidesMuscles)
    nSideMuscles = len(rightSideMuscles)
    
    # Extract muscle-tendon parameters (if not done already).
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
    
    # Tendon compliance.
    from muscleDataOpenSimAD import tendonCompliance
    sideTendonCompliance = tendonCompliance(nSideMuscles)
    tendonCompliance = np.concatenate((sideTendonCompliance, 
                                       sideTendonCompliance), axis=1)
    from muscleDataOpenSimAD import tendonShift
    sideTendonShift = tendonShift(nSideMuscles)
    tendonShift = np.concatenate((sideTendonShift, sideTendonShift), axis=1)
     
    # Specific tension.
    specificTension = 0.5*np.ones((1, nMuscles))
    
    # Hill-equilibrium. We use as muscle model the DeGrooteFregly2016 model
    # introduced in: https://pubmed.ncbi.nlm.nih.gov/27001399/.
    # In particular, we use the third formulation introduced in the paper,
    # with "normalized tendon force as a state and the scaled time derivative
    # of the normalized tendon force as a new control simplifying the
    # contraction dynamic equations".
    from functionCasADiOpenSimAD import hillEquilibrium    
    f_hillEquilibrium = hillEquilibrium(
        mtParameters, tendonCompliance, tendonShift, specificTension,
        ignorePassiveFiberForce=ignorePassiveFiberForce)
        
    # Time constants for activation dynamics.
    activationTimeConstant = 0.015
    deactivationTimeConstant = 0.06
    
    # Individual muscle weights. Option to weight the contribution of the
    # muscles to the muscle effort term differently. This is an experimental
    # feature.
    w_muscles = np.ones((nMuscles,1))
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
        s_muscleVolume = np.ones((nMuscles,))
    s_muscleVolume = np.reshape(s_muscleVolume, (nMuscles, 1))
    
    # %% Coordinates.
    # This section specifies the coordinates of the model. This is
    # specific to the Rajagopal musculoskeletal model.
    from utilsOpenSimAD import getIndices
    joints = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx',
              'pelvis_ty', 'pelvis_tz', 'hip_flexion_l', 'hip_adduction_l',
              'hip_rotation_l', 'hip_flexion_r', 'hip_adduction_r',
              'hip_rotation_r', 'knee_angle_l', 'knee_angle_r',
              'ankle_angle_l', 'ankle_angle_r', 'subtalar_angle_l',
              'subtalar_angle_r', 'mtp_angle_l', 'mtp_angle_r',
              'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    # Mtp coordinates.
    mtpJoints = ['mtp_angle_l', 'mtp_angle_r']
    nMtpJoints = len(mtpJoints)
    if not withMTP:
        for joint in mtpJoints:
            joints.remove(joint)
    # Lower-limb coordinates.
    lowerLimbJoints = copy.deepcopy(joints)
    # Arm coordinates.
    armJoints = ['arm_flex_l', 'arm_add_l', 'arm_rot_l',
                 'arm_flex_r', 'arm_add_r', 'arm_rot_r',
                 'elbow_flex_l', 'elbow_flex_r', 'pro_sup_l', 'pro_sup_r']
    nArmJoints = len(armJoints)
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
    
    # Lumbar coordinates (for torque actuators).    
    lumbarJoints = ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    nLumbarJoints = len(lumbarJoints)
    
    # Coordinates with passive torques.
    passiveTorqueJoints = [
        'hip_flexion_r', 'hip_flexion_l', 'hip_adduction_r', 'hip_adduction_l',
        'hip_rotation_r', 'hip_rotation_l', 'knee_angle_r', 'knee_angle_l', 
        'ankle_angle_r', 'ankle_angle_l', 'subtalar_angle_r', 
        'subtalar_angle_l', 'mtp_angle_r', 'mtp_angle_l',
        'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    if not withMTP:
        for joint in mtpJoints:
            passiveTorqueJoints.remove(joint)
    nPassiveTorqueJoints = len(passiveTorqueJoints)
        
    # Specify which states should have periodic constraints.
    # See settingsOpenSimAD for example.
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
        'hip_adduction_r', 'hip_rotation_l', 'hip_rotation_r', 'knee_angle_l',
        'knee_angle_r', 'ankle_angle_l', 'ankle_angle_r', 'subtalar_angle_l',
        'subtalar_angle_r', 'lumbar_extension', 'lumbar_bending',
        'lumbar_rotation']
    for joint in lumbarJoints:
        muscleDrivenJoints.remove(joint)
        
    # %% Coordinate actuator activation dynamics.
    if withArms or withLumbarCoordinateActuators:
        from functionCasADiOpenSimAD import coordinateActuatorDynamics
        if withArms:
            f_armDynamics = coordinateActuatorDynamics(nArmJoints)
        if withLumbarCoordinateActuators:
            f_lumbarDynamics = coordinateActuatorDynamics(nLumbarJoints)
    
    # %% Passive/limit torques.
    from functionCasADiOpenSimAD import limitTorque, passiveTorque
    from muscleDataOpenSimAD import passiveJointTorqueData_3D    
    damping = 0.1
    f_passiveTorque = {}
    for joint in passiveTorqueJoints:
        f_passiveTorque[joint] = limitTorque(
            passiveJointTorqueData_3D(joint)[0],
            passiveJointTorqueData_3D(joint)[1], damping)    
    if withMTP:
        stiffnessMtp = 25
        dampingMtp = 2
        f_linearPassiveMtpTorque = passiveTorque(stiffnessMtp, dampingMtp)        
    if withArms:
        stiffnessArm = 0
        dampingArm = 0.1
        f_linearPassiveArmTorque = passiveTorque(stiffnessArm, dampingArm)
        
    # %% Polynomials
    from functionCasADiOpenSimAD import polynomialApproximation
    leftPolynomialJoints = [
        'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l',
        'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', 
        'lumbar_extension', 'lumbar_bending', 'lumbar_rotation'] 
    rightPolynomialJoints = [
        'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r',
        'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
        'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    if not withMTP:
        leftPolynomialJoints.remove('mtp_angle_l')
        rightPolynomialJoints.remove('mtp_angle_r')
    leftPolynomialJoints.remove('lumbar_extension')
    leftPolynomialJoints.remove('lumbar_bending')
    leftPolynomialJoints.remove('lumbar_rotation')
    rightPolynomialJoints.remove('lumbar_extension')
    rightPolynomialJoints.remove('lumbar_bending')
    rightPolynomialJoints.remove('lumbar_rotation')
        
    pathGenericTemplates = os.path.join(baseDir, "OpenSimPipeline") 
    pathDummyMotion = os.path.join(pathGenericTemplates, "MuscleAnalysis", 
                                   'DummyMotion.mot')
    loadPolynomialData = True
    if (not os.path.exists(os.path.join(
            pathModelFolder, model_full_name + '_polynomial_r_default.npy'))
            or not os.path.exists(os.path.join(
            pathModelFolder, model_full_name + '_polynomial_l_default.npy'))):
        loadPolynomialData = False
        
    from muscleDataOpenSimAD import getPolynomialData
    polynomialData = {}
    polynomialData['r'] = getPolynomialData(
        loadPolynomialData, pathModelFolder, model_full_name, pathDummyMotion, 
        rightPolynomialJoints, muscles, side='r')
    polynomialData['l'] = getPolynomialData(
        loadPolynomialData, pathModelFolder, model_full_name, pathDummyMotion, 
        leftPolynomialJoints, leftSideMuscles, side='l')
     
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
    
    leftPolynomialJointIndices = getIndices(joints, leftPolynomialJoints)
    rightPolynomialJointIndices = getIndices(joints, rightPolynomialJoints)    
    leftPolynomialMuscleIndices = (
        list(range(nSideMuscles)) + 
        list(range(nSideMuscles, nSideMuscles)))
    rightPolynomialMuscleIndices = list(range(nSideMuscles))
    from utilsOpenSimAD import getMomentArmIndices
    momentArmIndices = getMomentArmIndices(rightSideMuscles,
                                           leftPolynomialJoints,
                                           rightPolynomialJoints, 
                                           polynomialData['r'])    
    # Test polynomials
    plotPolynomials = False
    if plotPolynomials:
        from polynomialsOpenSimAD import testPolynomials
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
    
    # %% External functions.
    # The external function is written in C++ and compiled as a library, which
    # can then be called with CasADi. In the external function, we build the
    # OpenSim model and run inverse dynamics. The function takes as inputs
    # joint positions, velocities, and accelerations, which are states and
    # controls of the optimal control problem. The external function returns
    # joint torques as well as some outputs of interest, eg segment origins,
    # that you may want to use as part of the problem formulation.
    if platform.system() == 'Windows':
        ext_F = '.dll'
    elif platform.system() == 'Darwin':
        ext_F = '.dylib'
    elif platform.system() == 'Linux':
        ext_F = '.so'
    else:
        raise ValueError("Platform not supported.")
    suff_tread = ''
    if treadmill[trial]:
        suff_tread = '_treadmill'
    
    F, F_map = {}, {}
    for trial in trials:
        F[trial] = ca.external(
            'F', os.path.join(
                pathExternalFunctionFolder, 'F' + suff_tread + ext_F))
        F_map[trial] = np.load(
            os.path.join(pathExternalFunctionFolder, 
                         'F' + suff_tread + '_map.npy'), 
            allow_pickle=True).item()
    
    # Indices outputs external function.
    nContactSpheres = 6
    for trial in trials:
        if heel_vGRF_threshold[trial] > 0:
            # Indices vertical ground reaction forces heel contact spheres.
            idx_vGRF_heel = [F_map[trial]['GRFs']['Sphere_0'][1],
                             F_map[trial]['GRFs']['Sphere_6'][1]]
        if min_ratio_vGRF[trial]:
            idx_vGRF = []
            for contactSphere in range(2*nContactSpheres):
                idx_vGRF.append(F_map[trial]['GRFs'][
                    'Sphere_{}'.format(contactSphere)][1])
            # Indices vertical ground reaction forces rear contact spheres.
            idx_vGRF_rear_l = [idx_vGRF[0+nContactSpheres], 
                               idx_vGRF[3+nContactSpheres]]
            idx_vGRF_rear_r = [idx_vGRF[0], idx_vGRF[3]]
            # Indices vertical ground reaction forces front contact spheres.
            idx_vGRF_front_l = [idx_vGRF[1+nContactSpheres], 
                                idx_vGRF[2+nContactSpheres], 
                                idx_vGRF[4+nContactSpheres], 
                                idx_vGRF[5+nContactSpheres]]            
            idx_vGRF_front_r = [idx_vGRF[1], idx_vGRF[2], 
                                idx_vGRF[4], idx_vGRF[5]]            
        if yCalcnToes[trial]:
            # Indices vertical position origins calc and toes segments.
            idx_yCalcnToes = [F_map[trial]['body_origins']['calcn_l'][1],
                              F_map[trial]['body_origins']['calcn_r'][1],
                              F_map[trial]['body_origins']['toes_l'][1],
                              F_map[trial]['body_origins']['toes_r'][1]]

    # Lists to map order of coordinates defined here and in external function.
    idxGroundPelvisJointsinF = [F_map[trial]['residuals'][joint] 
                                for joint in groundPelvisJoints]    
    idxJoints4F = [joints.index(joint) 
                   for joint in list(F_map[trial]['residuals'].keys())]
        
    
            
    # %% Kinematic data
    from utilsOpenSimAD import getIK, filterDataFrame
    from utilsOpenSimAD import interpolateDataFrame, selectDataFrame
    Qs_toTrack = {}
    Qs_toTrack_sel = {}
    for trial in trials:
        pathIK = os.path.join(pathIKFolder, trial + '.mot')        
        # Extract joint positions from walking trial.
        Qs_fromIK = getIK(pathIK, joints)
        if filter_Qs_toTrack[trial]:
            Qs_fromIK_filter = filterDataFrame(
                Qs_fromIK, cutoff_frequency=cutoff_freq_Qs[trial])
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
   
    # %% Other helper CasADi functions
    from functionCasADiOpenSimAD import normSumSqr
    from functionCasADiOpenSimAD import normSumWeightedPow
    from functionCasADiOpenSimAD import diffTorques
    from functionCasADiOpenSimAD import normSumWeightedSqrDiff
    f_NMusclesSum2 = normSumSqr(nMuscles)
    f_NMusclesSumWeightedPow = normSumWeightedPow(nMuscles, powActivations)
    f_nJointsSum2 = normSumSqr(nJoints)
    f_NQsToTrackWSum2 = normSumWeightedSqrDiff(nEl_toTrack)
    if withArms:
        f_nArmJointsSum2 = normSumSqr(nArmJoints)
    if withLumbarCoordinateActuators:
        f_nLumbarJointsSum2 = normSumSqr(nLumbarJoints)  
    f_diffTorques = diffTorques()  
        
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
    if withReserveActuators:
        uw['rAct'], lw['rAct'], scaling['rAct'] = {}, {}, {}
        uw['rActk'], lw['rActk']= {}, {}
        
    d = 3 # interpolating polynomial.
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
        uw['Qs'][trial], lw['Qs'][trial], scaling['Qs'][trial] =  bounds.getBoundsPosition_fixed_update2()
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
        uw['Qdds'][trial], lw['Qdds'][trial], scaling['Qdds'][trial] = bounds.getBoundsAcceleration()
        uw['Qddsk'][trial] = ca.vec(uw['Qdds'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
        lw['Qddsk'][trial] = ca.vec(lw['Qdds'][trial].to_numpy().T * np.ones((1, N[trial]))).full()
        # Reserve actuators
        if withReserveActuators:
            uw['rAct'][trial], lw['rAct'][trial], scaling['rAct'][trial] = {}, {}, {}
            uw['rActk'][trial], lw['rActk'][trial], = {}, {}
            for c_j in reserveActuatorCoordinates:
                uw['rAct'][trial][c_j], lw['rAct'][trial][c_j], scaling['rAct'][trial][c_j] = bounds.getBoundsReserveActuators(c_j, reserveActuatorCoordinates[c_j])
                uw['rActk'][trial][c_j] = ca.vec(uw['rAct'][trial][c_j].to_numpy().T * np.ones((1, N[trial]))).full()
                lw['rActk'][trial][c_j] = ca.vec(lw['rAct'][trial][c_j].to_numpy().T * np.ones((1, N[trial]))).full()
                    
        #######################################################################
        # Static parameters
        #######################################################################
        if offset_ty:
            scaling['Offset'][trial] = 1.         
            uw['Offset'][trial], lw['Offset'][trial] = bounds.getBoundsOffset(scaling['Offset'][trial])
            uw['Offsetk'][trial] = uw['Offset'][trial].to_numpy()
            lw['Offsetk'][trial] = lw['Offset'][trial].to_numpy()
    
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
    if withLumbarCoordinateActuators:
        w0['LumbarA'], w0['LumbarAj'], w0['LumbarE'] = {}, {}, {}
    w0['ADt'] = {}
    w0['FDt'] = {}
    w0['Qdds'] = {}
    if offset_ty:
        w0['Offset'] = {}
    if withReserveActuators:
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
        w0['Qs'][trial] = guess.getGuessPosition(scaling['Qs'][trial])
        w0['Qsj'][trial] = guess.getGuessPositionCol()
        # Joint velocities
        w0['Qds'][trial] = guess.getGuessVelocity(scaling['Qds'][trial])
        w0['Qdsj'][trial] = guess.getGuessVelocityCol()    
        if withArms:
            w0['ArmA'][trial] = guess.getGuessTorqueActuatorActivation(armJoints)   
            w0['ArmAj'][trial] = guess.getGuessTorqueActuatorActivationCol(armJoints)
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
        if withLumbarCoordinateActuators:
            # Lumbar activations
            w0['LumbarE'][trial] = guess.getGuessTorqueActuatorExcitation(lumbarJoints)
        # Muscle force derivatives   
        w0['FDt'][trial] = guess.getGuessForceDerivative(scaling['FDt'][trial])
        # Joint velocity derivatives (accelerations)
        w0['Qdds'][trial] = guess.getGuessAcceleration(scaling['Qdds'][trial])
        # Reserve actuators
        if withReserveActuators:
            w0['rAct'][trial] = {}
            for c_j in reserveActuatorCoordinates:
                w0['rAct'][trial][c_j] = guess.getGuessReserveActuators(c_j)            
            
        ###########################################################################
        # Static parameters
        ###########################################################################
        if offset_ty:
            w0['Offset'][trial] = guess.getGuessOffset(scaling['Offset'][trial])
            
    # %% Tracking data
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
        if filter_Qds_toTrack[trial]:
            Qds_spline_filter = filterDataFrame(
                Qds_spline, cutoff_frequency=cutoff_freq_Qds[trial])
        else:
            Qds_spline_filter = Qds_spline
            
        if filter_Qdds_toTrack[trial]:
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
                
            if filter_Qdds_toTrack[trial]:
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
        if offset_ty:
            # Offset pelvis_ty.
            offset = opti.variable(1)
            opti.subject_to(opti.bounded(lw['Offsetk'][trial], offset, uw['Offsetk'][trial]))
            opti.set_initial(offset, w0['Offset'][trial])
        else:
            offset = 0            
        # Initialize variables.    
        a, a_col, nF, nF_col = {}, {}, {}, {}
        Qs, Qs_col, Qds, Qds_col = {}, {}, {}, {}
        if withArms:
            aArm, aArm_col, eArm = {}, {}, {}
        if withLumbarCoordinateActuators:
            aLumbar, aLumbar_col, eLumbar = {}, {}, {}            
        aDt, nFDt, Qdds = {}, {}, {}
        if withReserveActuators:
            rAct = {}        
        # Loop over trials.
        for trial in trials:        
            # Time step    
            h = timeElapsed[trial] / N[trial]
            # States
            # Muscle activation at mesh points
            a[trial] = opti.variable(nMuscles, N[trial]+1)
            opti.subject_to(opti.bounded(lw['Ak'][trial], ca.vec(a[trial]), uw['Ak'][trial]))
            opti.set_initial(a[trial], w0['A'][trial].to_numpy().T)
            assert np.alltrue(lw['Ak'][trial] <= ca.vec(w0['A'][trial].to_numpy().T).full()), "lw Muscle activation"
            assert np.alltrue(uw['Ak'][trial] >= ca.vec(w0['A'][trial].to_numpy().T).full()), "uw Muscle activation"
            # Muscle activation at collocation points
            a_col[trial] = opti.variable(nMuscles, d*N[trial])
            opti.subject_to(opti.bounded(lw['Aj'][trial], ca.vec(a_col[trial]), uw['Aj'][trial]))
            opti.set_initial(a_col[trial], w0['Aj'][trial].to_numpy().T)
            assert np.alltrue(lw['Aj'][trial] <= ca.vec(w0['Aj'][trial].to_numpy().T).full()), "lw Muscle activation col"
            assert np.alltrue(uw['Aj'][trial] >= ca.vec(w0['Aj'][trial].to_numpy().T).full()), "uw Muscle activation col"
            # Muscle force at mesh points
            nF[trial] = opti.variable(nMuscles, N[trial]+1)
            opti.subject_to(opti.bounded(lw['Fk'][trial], ca.vec(nF[trial]), uw['Fk'][trial]))
            opti.set_initial(nF[trial], w0['F'][trial].to_numpy().T)
            assert np.alltrue(lw['Fk'][trial] <= ca.vec(w0['F'][trial].to_numpy().T).full()), "lw Muscle force"
            assert np.alltrue(uw['Fk'][trial] >= ca.vec(w0['F'][trial].to_numpy().T).full()), "uw Muscle force"
            # Muscle force at collocation points
            nF_col[trial] = opti.variable(nMuscles, d*N[trial])
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
            # Small margin to account for filtering.
            assert np.alltrue(lw['Qsk'][trial] - np.pi/180 <= ca.vec(guessQsEnd).full()), "lw Joint position"
            assert np.alltrue(uw['Qsk'][trial] + np.pi/180 >= ca.vec(guessQsEnd).full()), "uw Joint position"
            # Joint position at collocation points
            Qs_col[trial] = opti.variable(nJoints, d*N[trial])
            opti.subject_to(opti.bounded(lw['Qsj'][trial], ca.vec(Qs_col[trial]), uw['Qsj'][trial]))
            opti.set_initial(Qs_col[trial], w0['Qsj'][trial].to_numpy().T)
            # Small margin to account for filtering.
            assert np.alltrue(lw['Qsj'][trial] - np.pi/180 <= ca.vec(w0['Qsj'][trial].to_numpy().T).full()), "lw Joint position col"
            assert np.alltrue(uw['Qsj'][trial] + np.pi/180 >= ca.vec(w0['Qsj'][trial].to_numpy().T).full()), "uw Joint position col"
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
            aDt[trial] = opti.variable(nMuscles, N[trial])
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
            if withLumbarCoordinateActuators:
                # Lumbar excitation at mesh points
                eLumbar[trial] = opti.variable(nLumbarJoints, N[trial])
                opti.subject_to(opti.bounded(lw['LumbarEk'][trial], ca.vec(eLumbar[trial]), uw['LumbarEk'][trial]))
                opti.set_initial(eLumbar[trial], w0['LumbarE'][trial].to_numpy().T)
                assert np.alltrue(lw['LumbarEk'][trial] <= ca.vec(w0['LumbarE'][trial].to_numpy().T).full()), "lw Lumbar excitation"
                assert np.alltrue(uw['LumbarEk'][trial] >= ca.vec(w0['LumbarE'][trial].to_numpy().T).full()), "uw Lumbar excitation"
            # Muscle force derivative at mesh points
            nFDt[trial] = opti.variable(nMuscles, N[trial])
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
            if withReserveActuators:
                rAct[trial] = {}
                for c_j in reserveActuatorCoordinates:                    
                    rAct[trial][c_j] = opti.variable(1, N[trial])
                    opti.subject_to(opti.bounded(lw['rActk'][trial][c_j], ca.vec(rAct[trial][c_j]), uw['rActk'][trial][c_j]))
                    opti.set_initial(rAct[trial][c_j], w0['rAct'][trial][c_j].to_numpy().T)
                    assert np.alltrue(lw['rActk'][trial][c_j] <= ca.vec(w0['rAct'][trial][c_j].to_numpy().T).full()), "lw reserve"
                    assert np.alltrue(uw['rActk'][trial][c_j] >= ca.vec(w0['rAct'][trial][c_j].to_numpy().T).full()), "uw reserve"
                
            # %% Plots guess vs bounds.
            plotGuessVsBounds = False
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
            if withReserveActuators:
                rAct_nsc = {}
                for c_j in reserveActuatorCoordinates:
                    rAct_nsc[c_j] = rAct[trial][c_j] * (scaling['rAct'][trial][c_j].to_numpy().T * np.ones((1, N[trial])))
                    
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
                if withReserveActuators:
                    rActk = {}
                    rActk_nsc = {}
                    for c_j in reserveActuatorCoordinates:
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
                if treadmill[trial]:
                    Tk = F[trial](ca.vertcat(
                        ca.vertcat(QsQdskj_nsc[:, 0],
                                   Qddsk_nsc[idxJoints4F]),
                        -trials[trial]['treadmill_speed']))
                else:
                    Tk = F[trial](ca.vertcat(QsQdskj_nsc[:, 0], 
                                             Qddsk_nsc[idxJoints4F]))
                        
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
                        aArmDtj = f_armDynamics(
                            eArmk, aArmkj[:, j+1])
                        opti.subject_to(h*aArmDtj - aArmp == 0) 
                    if withLumbarCoordinateActuators:
                        # Lumbar activation dynamics (explicit formulation) 
                        aLumbarDtj = f_lumbarDynamics(
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
                    if trackQdds:
                        accelerationTrackingTerm = f_NQsToTrackWSum2(
                            Qddsk[idx_coordinates_toTrack],
                            dataToTrack_dotdot_sc[trial][:, k],
                            w_dataToTrack)
                        J += (weights['accelerationTrackingTerm'] * 
                              accelerationTrackingTerm * h * B[j + 1])                    
                    if withReserveActuators:
                        reserveActuatorTerm = 0
                        for c_j in reserveActuatorCoordinates:                        
                            reserveActuatorTerm += ca.sumsqr(rActk[c_j])                            
                        reserveActuatorTerm /= len(reserveActuatorCoordinates)
                        J += (weights['reserveActuatorTerm'] * 
                              reserveActuatorTerm * h * B[j + 1])
                        
                    if min_ratio_vGRF[trial] and weights['vGRFRatioTerm'] > 0:
                        vGRF_ratio_l = ca.sqrt((ca.sum1(Tk[idx_vGRF_front_l])) / 
                                               (ca.sum1(Tk[idx_vGRF_rear_l])))
                        vGRF_ratio_r = ca.sqrt((ca.sum1(Tk[idx_vGRF_front_r])) /
                                               (ca.sum1(Tk[idx_vGRF_rear_r])))
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
                    if withReserveActuators and joint in reserveActuatorCoordinates:
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
                # Motion-specific constraints
                ###############################################################
                # TODO: deprecated doc
                # We want all contact spheres to be in contact with the ground
                # at all time. This is specific to squats and corresponds to
                # instructions that would be given to subjets: "keep you feet
                # flat on the gound". Without such constraints, the model tends
                # to lean forward, likely to reduce quadriceps loads.
                
                # During squats, we want the model's heels to remain in contact
                # with the ground. We do that here by enforcing that the
                # vertical ground reaction force of the heel contact spheres is
                # larger than heel_vGRF_threshold.
                if heel_vGRF_threshold[trial] > 0:
                    vGRFk = Tk[idx_vGRF_heel]
                    opti.subject_to(vGRFk > heel_vGRF_threshold[trial])
                        
                if yCalcnToes[trial]:
                    yCalcnToesk = Tk[idx_yCalcnToes]
                    opti.subject_to(yCalcnToesk > yCalcnToesThresholds[trial])
                
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
                # TODO arms periodic constraints                    
                # if withArms:
                #     # Arm activations
                #     opti.subject_to(aArm[trial][:, -1] - 
                #     aArm[trial][:, 0] == 0)
                    
            ###################################################################
            # Constraints on pelvis_ty if offset as design variable
            ###################################################################
            if offset_ty and 'pelvis_ty' in coordinate_constraints:
                
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
        w_opt, stats = solve_with_bounds(opti, ipopt_tolerance)             
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
        if withLumbarCoordinateActuators:
            aLumbar_opt, aLumbar_col_opt, eLumbar_opt = {}, {}, {}            
        aDt_opt = {}
        nFDt_col_opt = {}
        Qdds_col_opt = {}
        if withReserveActuators:
            rAct_opt = {} 
        for trial in trials:
            a_opt[trial] = (
                np.reshape(w_opt[starti:starti+nMuscles*(N[trial]+1)],
                           (N[trial]+1, nMuscles))).T
            starti = starti + nMuscles*(N[trial]+1)
            a_col_opt[trial] = (
                np.reshape(w_opt[starti:starti+nMuscles*(d*N[trial])],
                           (d*N[trial], nMuscles))).T    
            starti = starti + nMuscles*(d*N[trial])
            nF_opt[trial] = (
                np.reshape(w_opt[starti:starti+nMuscles*(N[trial]+1)],
                           (N[trial]+1, nMuscles))).T  
            starti = starti + nMuscles*(N[trial]+1)
            nF_col_opt[trial] = (
                np.reshape(w_opt[starti:starti+nMuscles*(d*N[trial])],
                           (d*N[trial], nMuscles))).T
            starti = starti + nMuscles*(d*N[trial])
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
                np.reshape(w_opt[starti:starti+nMuscles*N[trial]],
                           (N[trial], nMuscles))).T
            starti = starti + nMuscles*N[trial] 
            if withArms:
                eArm_opt[trial] = (
                    np.reshape(w_opt[starti:starti+nArmJoints*N[trial]],
                               (N[trial], nArmJoints))).T
                starti = starti + nArmJoints*N[trial]
            if withLumbarCoordinateActuators:
                eLumbar_opt[trial] = (
                    np.reshape(w_opt[starti:starti+nLumbarJoints*N[trial]],
                               (N[trial], nLumbarJoints))).T
                starti = starti + nLumbarJoints*N[trial]
            nFDt_col_opt[trial] = (
                np.reshape(w_opt[starti:starti+nMuscles*(N[trial])],
                           (N[trial], nMuscles))).T
            starti = starti + nMuscles*(N[trial])
            Qdds_col_opt[trial] = (
                np.reshape(w_opt[starti:starti+nJoints*(N[trial])],
                           (N[trial], nJoints))).T
            starti = starti + nJoints*(N[trial])
            if withReserveActuators:
                rAct_opt[trial] = {}
                for c_j in reserveActuatorCoordinates:
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
        if withReserveActuators:
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
            if withReserveActuators:
                rAct_opt_nsc[trial] = {}
                for c_j in reserveActuatorCoordinates:
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
                               Qdds_opt_nsc[trial][:, 0]), -trials[trial]['treadmill_speed']))
            else:
                Tj_temp = F[trial](ca.vertcat(QsQds_opt_nsc[trial][:, 0], 
                                              Qdds_opt_nsc[trial][:, 0]))          
            F_out_pp = np.zeros((Tj_temp.shape[0], N[trial]))
            if withMTP:
                mtpT = np.zeros((nMtpJoints, N[trial]))
            if withArms:
                armT = np.zeros((nArmJoints, N[trial]))
            for k in range(N[trial]):
                if treadmill[trial]:
                    Tk = F[trial](ca.vertcat(
                        ca.vertcat(QsQds_opt_nsc[trial][:, k],
                                   Qdds_opt_nsc[trial][:, k]), 
                        -trials[trial]['treadmill_speed']))
                else:
                    Tk = F[trial](ca.vertcat(QsQds_opt_nsc[trial][:, k],
                                   Qdds_opt_nsc[trial][:, k]))
                F_out_pp[:, k] = Tk.full().T
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
                assert np.alltrue(np.abs(armT) < 10**(-ipopt_tolerance)), (
                    "Error arm torques balance")                    
            if stats['success'] and withMTP:
                assert np.alltrue(np.abs(mtpT) < 10**(-ipopt_tolerance)), (
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
        dimensions = ['x', 'y', 'z']
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
            if offset_ty:                    
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
                        if (joints[i] in coordinates_toTrack_list):
                            col_sim = 'orange'
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
                        if (joints[i] in coordinates_toTrack_list):
                            col_sim = 'orange'
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
                        if (joints[i] in coordinates_toTrack_list):
                            col_sim = 'orange'
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
                        if (joints[i] in coordinates_toTrack_list):
                            col_sim = 'orange'
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
        if withReserveActuators:    
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
            Ft_opt[trial] = np.zeros((nMuscles, N[trial]))
            
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
                if withLumbarCoordinateActuators:
                    eLumbark_opt = eLumbar_opt[trial][:, k]
                if withReserveActuators:
                    rActk_opt = {}
                    for c_j in reserveActuatorCoordinates:
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
                    positionTrackingTerm_opt = f_NQsToTrackWSum2(Qskj_opt[idx_coordinates_toTrack, 0], dataToTrack_nsc_offset_opt[trial][:, k], w_dataToTrack)                
                    velocityTrackingTerm_opt = f_NQsToTrackWSum2(Qdskj_opt[idx_coordinates_toTrack, 0], dataToTrack_dot_sc[trial][:, k], w_dataToTrack)
                    
                    positionTrackingTerm_opt_all += weights['positionTrackingTerm'] * positionTrackingTerm_opt * h * B[j + 1]
                    velocityTrackingTerm_opt_all += weights['velocityTrackingTerm'] * velocityTrackingTerm_opt * h * B[j + 1]
                    activationTerm_opt_all += weights['activationTerm'] * activationTerm_opt * h * B[j + 1]
                    jointAccelerationTerm_opt_all += weights['jointAccelerationTerm'] * jointAccelerationTerm_opt * h * B[j + 1]
                    activationDtTerm_opt_all += weights['activationDtTerm'] * activationDtTerm_opt * h * B[j + 1]
                    forceDtTerm_opt_all += weights['forceDtTerm'] * forceDtTerm_opt * h * B[j + 1]
                    if withArms:
                        armExcitationTerm_opt = f_nArmJointsSum2(eArmk_opt) 
                        armExcitationTerm_opt_all += weights['armExcitationTerm'] * armExcitationTerm_opt * h * B[j + 1]
                    if withLumbarCoordinateActuators:
                        lumbarExcitationTerm_opt = f_nLumbarJointsSum2(eLumbark_opt) 
                        lumbarExcitationTerm_opt_all += weights['lumbarExcitationTerm'] * lumbarExcitationTerm_opt * h * B[j + 1]
                    if trackQdds:
                        accelerationTrackingTerm_opt = f_NQsToTrackWSum2(Qddsk_opt[idx_coordinates_toTrack], dataToTrack_dotdot_sc[trial][:, k], w_dataToTrack)
                        accelerationTrackingTerm_opt_all += (weights['accelerationTrackingTerm'] * accelerationTrackingTerm_opt * h * B[j + 1])
                    if withReserveActuators:
                        reserveActuatorTerm_opt = 0
                        for c_j in reserveActuatorCoordinates:                        
                            reserveActuatorTerm_opt += ca.sumsqr(rActk_opt[c_j])                            
                        reserveActuatorTerm_opt /= len(reserveActuatorCoordinates)
                        reserveActuatorTerm_opt_all += (weights['reserveActuatorTerm'] * reserveActuatorTerm_opt * h * B[j + 1])
                    if min_ratio_vGRF[trial] and weights['vGRFRatioTerm'] > 0:
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
        if withLumbarCoordinateActuators:
            JMotor_opt += lumbarExcitationTerm_opt_all.full()
        if min_ratio_vGRF[trial] and weights['vGRFRatioTerm'] > 0:
            JMotor_opt += vGRFRatioTerm_opt_all            
              
        JTrack_opt = (positionTrackingTerm_opt_all.full() +  
                      velocityTrackingTerm_opt_all.full())
        if trackQdds:
            JTrack_opt += accelerationTrackingTerm_opt_all.full()
        if withReserveActuators:    
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
        if withLumbarCoordinateActuators:
            JTerms["lumbarExcitationTerm"] = lumbarExcitationTerm_opt_all.full()[0][0]
        JTerms["jointAccelerationTerm"] = jointAccelerationTerm_opt_all.full()[0][0]
        JTerms["activationDtTerm"] = activationDtTerm_opt_all.full()[0][0]
        JTerms["forceDtTerm"] = forceDtTerm_opt_all.full()[0][0]
        JTerms["positionTerm"] = positionTrackingTerm_opt_all.full()[0][0]
        JTerms["velocityTerm"] = velocityTrackingTerm_opt_all.full()[0][0]
        if trackQdds:
            JTerms["accelerationTerm"] = accelerationTrackingTerm_opt_all.full()[0][0]       
        JTerms["activationTerm_sc"] = JTerms["activationTerm"] / JAll_opt[0][0]
        if withArms:
            JTerms["armExcitationTerm_sc"] = JTerms["armExcitationTerm"] / JAll_opt[0][0]
        if withLumbarCoordinateActuators:
            JTerms["lumbarExcitationTerm_sc"] = JTerms["lumbarExcitationTerm"] / JAll_opt[0][0]
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
        print("Joint Accelerations: " + str(np.round(JTerms["jointAccelerationTerm_sc"] * 100, 2)) + "%")
        print("Muscle activations derivatives: " + str(np.round(JTerms["activationDtTerm_sc"] * 100, 2)) + "%")
        print("Muscle-tendon forces derivatives: " + str(np.round(JTerms["forceDtTerm_sc"] * 100, 2)) + "%")
        print("Position tracking: " + str(np.round(JTerms["positionTerm_sc"] * 100, 2)) + "%")
        print("Velocity tracking: " + str(np.round(JTerms["velocityTerm_sc"] * 100, 2)) + "%")
        if trackQdds:
            print("Acceleration tracking: " + str(np.round(JTerms["accelerationTerm_sc"] * 100, 2)) + "%")            
        print("# Iterations: " + str(stats["iter_count"]))
        print("\n")            
            
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
                if withReserveActuators:
                    for count_j, c_j in enumerate(reserveActuatorCoordinates):                    
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
                
        # %% Filter GRFs
        # Cutoff frequency same as the one used to filter Qs.
        from utilsOpenSimAD import filterNumpyArray
        GRF_all_opt_filt = {}
        GRF_all_opt_filt['all'] = {}
        for trial in trials:            
            GRF_all_opt_filt['all'][trial] = filterNumpyArray(
                GRF_all_opt['all'][trial].T, tgridf[trial][0,:-1], 
                cutoff_frequency=cutoff_freq_Qs[trial]).T
                
        # %% Express in %BW (GRF) and %BW*height (torques)
        gravity = 9.80665
        BW = settings['mass_kg'] * gravity
        BW_ht = BW * settings['height_m']
        
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
                
        optimaltrajectories[case]['iter'] = stats['iter_count']
                
        np.save(os.path.join(pathResults, 'optimaltrajectories.npy'),
                optimaltrajectories)
            
        # %% Visualize results against bounds
        if visualizeResultsBounds:
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
                #######################################################################
                # Controls
                # Muscle activation derivative at mesh points
                lwp = lw['ADt'][trial].to_numpy().T
                uwp = uw['ADt'][trial].to_numpy().T
                y = aDt_opt[trial]
                title='Muscle activation derivative at mesh points' 
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
