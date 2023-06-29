'''
    ---------------------------------------------------------------------------
    OpenCap processing: muscleDataOpenSimAD.py
    ---------------------------------------------------------------------------
    Copyright 2022 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''

import os
import numpy as np

# %% Import muscle-tendon parameters.
# We save the muscle-tendon parameters associated with the model the first time
# we 'use' the model such that we do not need OpenSim later on. We extract 5
# parameters: maximal isometric muscle force, optimal fiber length, tendon
# slack length, pennation angle at optimal fiber length, and maximal
# contraction velocity (times optimal fiber length).
def getMTParameters(pathModel, muscles, loadMTParameters,
                    pathMTParameters=0, modelName='', side=''):
    
    if loadMTParameters:        
        mtParameters = np.load(os.path.join(
            pathMTParameters, 
            '{}_mtParameters_{}.npy'.format(modelName, side)),
            allow_pickle=True)
    else:   
        import opensim
        model = opensim.Model(pathModel)
        mtParameters = np.zeros([5,len(muscles)])
        model_muscles = model.getMuscles()
        for i in range(len(muscles)):
           muscle = model_muscles.get(muscles[i])
           mtParameters[0,i] = muscle.getMaxIsometricForce()
           mtParameters[1,i] = muscle.getOptimalFiberLength()
           mtParameters[2,i] = muscle.getTendonSlackLength()
           mtParameters[3,i] = muscle.getPennationAngleAtOptimalFiberLength()
           mtParameters[4,i] = (muscle.getMaxContractionVelocity() * 
                                muscle.getOptimalFiberLength())
        if pathMTParameters != 0:
           np.save(os.path.join(pathMTParameters, 
                                '{}_mtParameters_{}.npy'.format(
                                    modelName, side)), mtParameters)
       
    return mtParameters

# %% Extract muscle-tendon lenghts and moment arms.
# We extract data from varying limb postures, such as to later fit polynomials
# to approximate muscle tendon lenghts, velocities, and moment arms.
def get_mtu_length_and_moment_arm(pathModel, data, coordinates_table, 
                                  idxSlice):
    import opensim
    
    # Create temporary motion file.
    from utils import numpy_to_storage  
    labels = ['time'] + coordinates_table      
    time = np.linspace(0, data.shape[0]/100-0.01, data.shape[0])    
    c_data = np.concatenate((np.expand_dims(time, axis=1), data),axis=1)
    modelDir = os.path.dirname(pathModel)
    motionPath = os.path.join(modelDir, 'motion4MA_{}.mot'.format(idxSlice))  
    numpy_to_storage(labels, c_data, motionPath, datatype='IK')
    
    # Model.
    opensim.Logger.setLevelString('error')
    model = opensim.Model(pathModel)
    model.initSystem()
    
    # Create time-series table with coordinate values. 
    table = opensim.TimeSeriesTable(motionPath)
    tableProcessor = opensim.TableProcessor(table)
    tableProcessor.append(opensim.TabOpUseAbsoluteStateNames())
    time = np.asarray(table.getIndependentColumn())
    table = tableProcessor.processAndConvertToRadians(model)
    
    # Append missing states to table.
    stateVariableNames = model.getStateVariableNames()
    stateVariableNamesStr = [
        stateVariableNames.get(i) for i in range(
            stateVariableNames.getSize())]
    existingLabels = table.getColumnLabels()
    for stateVariableNameStr in stateVariableNamesStr:
        if not stateVariableNameStr in existingLabels:
            # Hack for the patella, need to provide the same value as for the
            # knee.
            if 'knee_angle_r_beta/value' in stateVariableNameStr:
                vec_0 = opensim.Vector(
                    data[:, coordinates_table.index(
                        '/jointset/walker_knee_r/knee_angle_r/value')] * 
                    np.pi/180 )         
            elif 'knee_angle_l_beta/value' in stateVariableNameStr:
                vec_0 = opensim.Vector(
                    data[:, coordinates_table.index(
                        '/jointset/walker_knee_l/knee_angle_l/value')] * 
                    np.pi/180 )
            else:
                vec_0 = opensim.Vector([0] * table.getNumRows())            
            table.appendColumn(stateVariableNameStr, vec_0)
    stateTrajectory = opensim.StatesTrajectory.createFromStatesTable(model, 
                                                                     table)
    
    # Number of muscles.
    muscles = []
    forceSet = model.getForceSet()
    for i in range(forceSet.getSize()):        
        c_force_elt = forceSet.get(i)  
        if 'Muscle' in c_force_elt.getConcreteClassName():
            muscles.append(c_force_elt.getName())
    nMuscles = len(muscles)
    
    # Coordinates.
    coordinateSet = model.getCoordinateSet()
    nCoordinates = coordinateSet.getSize()
    coordinates = [coordinateSet.get(i).getName() for i in range(nCoordinates)]
    
    # TODO: hard coded to make run faster.
    rootCoordinates = [
        'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
        'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    lumbarCoordinates = ['lumbar_extension', 'lumbar_bending', 
                         'lumbar_rotation']    
    armCoordinates = ['arm_flex_r', 'arm_add_r', 'arm_rot_r', 
                      'elbow_flex_r', 'pro_sup_r', 
                      'arm_flex_l', 'arm_add_l', 'arm_rot_l', 
                      'elbow_flex_l', 'pro_sup_l']    
    coordinates_table_short = [
        label.split('/')[-2] for label in coordinates_table] # w/o /jointset/..
    
    # Compute muscle-tendon lengths and moment arms.
    lMT = np.zeros((data.shape[0], nMuscles))
    dM =  np.zeros((data.shape[0], nMuscles, len(coordinates_table_short)))
    for i in range(data.shape[0]):
        model.realizePosition(stateTrajectory[i])
        count = 0
        for m in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(m)
            if i == 0:
                muscleNames = [] 
            if 'Muscle' in c_force_elt.getConcreteClassName():
                muscleName = c_force_elt.getName()
                cObj = opensim.Muscle.safeDownCast(c_force_elt)            
                lMT[i,count] = cObj.getLength(stateTrajectory[i])
                if i == 0:
                    muscleNames.append(muscleName)                    
                for c, coord in enumerate(coordinates_table_short):
                    # We do not want to compute moment arms that are not
                    # relevant, eg for a muscle of the left side wrt a
                    # coordinate of the right side, or for a leg muscle with
                    # respect to a lumbar coordinate.
                    if muscleName[-2:] == '_l' and coord[-2:] == '_r':
                        dM[i, count, c] = 0
                    elif muscleName[-2:] == '_r' and coord[-2:] == '_l':
                        dM[i, count, c] = 0
                    elif (coord in rootCoordinates or 
                          coord in lumbarCoordinates or 
                          coord in armCoordinates):
                        dM[i, count, c] = 0
                    else:
                        coordinate = coordinateSet.get(
                            coordinates.index(coord))
                        dM[i, count, c] = cObj.computeMomentArm(
                            stateTrajectory[i], coordinate)
                count += 1
                        
    return [lMT, dM]

# %% Fit polynomial coefficients.
# We fit the polynomial coefficients if no polynomial data exist yet, and we
# save them such that we do not need to do the fitting again.
# Note: this code leverages parallel computing. We recommend running the code
# in the terminal as parallel computing might not be leveraged in IDEs like
# Spyder.
def getPolynomialData(loadPolynomialData, pathModelFolder, modelName='', 
                      pathMotionFile4Polynomials='', joints=[],
                      muscles=[], type_bounds_polynomials='default', side='',
                      nThreads=None, overwritedata4PolynomialFitting=False):
    
    pathPolynomialData = os.path.join(
        pathModelFolder, '{}_polynomial_{}_{}.npy'.format(
            modelName, side, type_bounds_polynomials))    
    if loadPolynomialData:
        polynomialData = np.load(pathPolynomialData, allow_pickle=True) 
        
    else:
        path_data4PolynomialFitting = os.path.join(
            pathModelFolder, 'data4PolynomialFitting_{}_{}.npy'.format(modelName, type_bounds_polynomials))
        # Generate polynomial data.
        if (not os.path.exists(path_data4PolynomialFitting) or 
            overwritedata4PolynomialFitting):            
            print('Generating data to fit polynomials.')            
            import opensim
            from joblib import Parallel, delayed
            import multiprocessing
            # Get training data from motion file.
            table = opensim.TimeSeriesTable(pathMotionFile4Polynomials)
            coordinates_table = list(table.getColumnLabels()) # w/ jointset/...
            data = table.getMatrix().to_numpy() # data in degrees w/o time
            pathModel = os.path.join(pathModelFolder, modelName + '.osim')
            # Set number of threads.
            if nThreads == None:
                nThreads = multiprocessing.cpu_count()-2 # default
            if nThreads < 1:
                nThreads = 1
            elif nThreads > multiprocessing.cpu_count():
                nThreads = multiprocessing.cpu_count()                
            # Generate muscle tendon lengths and moment arms (in parallel).
            slice_size = int(np.floor(data.shape[0]/nThreads))
            rest = data.shape[0] % nThreads
            outputs = Parallel(n_jobs=nThreads)(
                delayed(get_mtu_length_and_moment_arm)(
                    pathModel, data[i*slice_size:(i+1)*slice_size,:], 
                    coordinates_table, i) for i in range(nThreads))
            if rest != 0:
                output_last = get_mtu_length_and_moment_arm(
                    pathModel, data[-rest:,:], coordinates_table, 99)  
            # Delete temporary motion files.
            for file in os.listdir(pathModelFolder):
                if 'motion4MA_' in file:
                    os.remove(os.path.join(pathModelFolder, file))                
            # Gather data.
            lMT = np.zeros((data.shape[0], outputs[0][1].shape[1]))
            dM =  np.zeros((data.shape[0], outputs[0][1].shape[1], 
                            outputs[0][1].shape[2]))
            for i in range(len(outputs)):
                lMT[i*slice_size:(i+1)*slice_size, :] = outputs[i][0]
                dM[i*slice_size:(i+1)*slice_size, :, :] = outputs[i][1]
            if rest != 0:
                lMT[-rest:, :] = output_last[0]
                dM[-rest:, :, :] = output_last[1]
            # Put data in dict.
            # Muscles as ordered in model.
            opensim.Logger.setLevelString('error')
            model = opensim.Model(pathModel)  
            allMuscles = []
            forceSet = model.getForceSet()
            for i in range(forceSet.getSize()):        
                c_force_elt = forceSet.get(i)  
                if (c_force_elt.getConcreteClassName() == 
                    "Millard2012EquilibriumMuscle"):
                    allMuscles.append(c_force_elt.getName())    
            data4PolynomialFitting = {}
            data4PolynomialFitting['mtu_lengths'] = lMT
            data4PolynomialFitting['mtu_moment_arms'] = dM
            data4PolynomialFitting['muscle_names'] = allMuscles
            data4PolynomialFitting['coordinate_names'] = [
                label.split('/')[-2] for label in coordinates_table]
            data4PolynomialFitting['coordinate_values'] = data
            # Save data.
            np.save(path_data4PolynomialFitting, data4PolynomialFitting)
        else:
            data4PolynomialFitting = np.load(path_data4PolynomialFitting, 
                                             allow_pickle=True).item()
        # Fit polynomial coefficients.
        print('Fit polynomials.')
        from polynomialsOpenSimAD import getPolynomialCoefficients
        polynomialData = getPolynomialCoefficients(
            data4PolynomialFitting, joints, muscles, side=side)
        if pathModelFolder != 0:
            np.save(pathPolynomialData, polynomialData)
        print('Done fitting polynomials.')
           
    return polynomialData

# %% Tendon stiffness.
# Default value is 35.
def tendonCompliance(NSideMuscles):
    tendonCompliance = np.full((1, NSideMuscles), 35)
    
    return tendonCompliance

# Tendon shift to ensure that the tendon force, when the normalized tendon
# lenght is 1, is the same for all tendon stiffnesses.
def tendonShift(NSideMuscles):
    tendonShift = np.full((1, NSideMuscles), 0)
    
    return tendonShift

# %% Joint limit torques.
# Data from https://www.tandfonline.com/doi/abs/10.1080/10255849908907988
def passiveJointTorqueData(joint, model_type='rajagopal2016'):    
    
    kAll = {'hip_flexion_r' : [-2.44, 5.05, 1.51, -21.88],
            'hip_flexion_l' : [-2.44, 5.05, 1.51, -21.88],
            'hip_adduction_r': [-0.03, 14.94, 0.03, -14.94],
            'hip_adduction_l': [-0.03, 14.94, 0.03, -14.94], 
            'hip_rotation_r': [-0.03, 14.94, 0.03, -14.94],
            'hip_rotation_l': [-0.03, 14.94, 0.03, -14.94],
            'ankle_angle_r': [-2.03, 38.11, 0.18, -12.12],
            'ankle_angle_l': [-2.03, 38.11, 0.18, -12.12],
            'subtalar_angle_r': [-60.21, 16.32, 60.21, -16.32],
            'subtalar_angle_l': [-60.21, 16.32, 60.21, -16.32],
            'lumbar_extension': [-0.35, 30.72, 0.25, -20.36],
            'lumbar_bending': [-0.25, 20.36, 0.25, -20.36],
            'lumbar_rotation': [-0.25, 20.36, 0.25, -20.36]}      
    
    thetaAll = {'hip_flexion_r' : [-0.6981, 1.81],
                'hip_flexion_l' : [-0.6981, 1.81],
                'hip_adduction_r': [-0.5, 0.5],
                'hip_adduction_l': [-0.5, 0.5], 
                'hip_rotation_r': [-0.92, 0.92],
                'hip_rotation_l': [-0.92, 0.92],
                'ankle_angle_r': [-0.74, 0.52],
                'ankle_angle_l': [-0.74, 0.52],
                'subtalar_angle_r': [-0.65, 0.65],
                'subtalar_angle_l': [-0.65, 0.65],
                'lumbar_extension': [-0.5235987755982988, 0.17],
                'lumbar_bending': [-0.3490658503988659, 0.3490658503988659],
                'lumbar_rotation': [-0.3490658503988659, 0.3490658503988659]}
    
    if model_type=='rajagopal2016':
        kAll['knee_angle_r'] = [-11.03, 11.33, 6.09, -33.94]
        kAll['knee_angle_l'] = [-11.03, 11.33, 6.09, -33.94]
        kAll['mtp_angle_r'] = [-0.18, 70.08, 0.9, -14.87]
        kAll['mtp_angle_l'] = [-0.18, 70.08, 0.9, -14.87]
        thetaAll['knee_angle_r'] = [-0.13, 2.4]
        thetaAll['knee_angle_l'] = [-0.13, 2.4]
        thetaAll['mtp_angle_r'] = [-1.134464013796314, 0]
        thetaAll['mtp_angle_l'] = [-1.134464013796314, 0]
    else:
        raise ValueError("Model type unkown: passive torques")    
    
    k = kAll[joint] 
    theta = thetaAll[joint]
    
    return k, theta

# %% Coordinate actuator optimal forces.
# Values inspired from Fig. S3 from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5507211/
def get_coordinate_actuator_optimal_forces():
    
    coordinate_optimal_forces = {
        'hip_flexion_l': 500,
        'hip_flexion_r': 500,
        'hip_adduction_l': 400,
        'hip_adduction_r': 400,
        'hip_rotation_l': 400,
        'hip_rotation_r': 400,
        'knee_angle_l': 400,
        'knee_angle_r': 400,
        'ankle_angle_l': 400,
        'ankle_angle_r': 400, 
        'subtalar_angle_l': 400,
        'subtalar_angle_r': 400,
        'lumbar_extension': 300,
        'lumbar_bending': 300,
        'lumbar_rotation': 300,
        'arm_flex_l': 150,
        'arm_add_l': 150,
        'arm_rot_l': 150,
        'arm_flex_r': 150,
        'arm_add_r': 150,
        'arm_rot_r': 150,
        'elbow_flex_l': 150,
        'elbow_flex_r': 150,
        'pro_sup_l': 150,
        'pro_sup_r': 150}
    
    return coordinate_optimal_forces
