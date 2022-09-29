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

def getMTParameters(pathModel, muscles, loadMTParameters,
                    pathMTParameters=0, modelName=''):
    
    if loadMTParameters:        
        mtParameters = np.load(os.path.join(
            pathMTParameters, modelName + '_mtParameters.npy'),
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
           mtParameters[4,i] = muscle.getMaxContractionVelocity()*muscle.getOptimalFiberLength()
        if pathMTParameters != 0:
           np.save(os.path.join(pathMTParameters, modelName + '_mtParameters.npy'),
                   mtParameters)
       
    return mtParameters

# %% Extract muscle-tendon lenghts and moment arms.
# We extract data from varying limb postures, such as to later fit polynomials
# to approximate muscle tendon lenghts and moment arms.
def get_mtu_length_and_moment_arm(pathModel, data, coordinates_table, idxSlice):
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
                vec_0 = opensim.Vector(data[:, coordinates_table.index('/jointset/walker_knee_r/knee_angle_r/value')] * np.pi/180 )         
            elif 'knee_angle_l_beta/value' in stateVariableNameStr:
                vec_0 = opensim.Vector(data[:, coordinates_table.index('/jointset/walker_knee_l/knee_angle_l/value')] * np.pi/180 )
            else:
                vec_0 = opensim.Vector([0] * table.getNumRows())            
            table.appendColumn(stateVariableNameStr, vec_0)
    stateTrajectory = opensim.StatesTrajectory.createFromStatesTable(model, table)
    
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
    
    # TODO: hard coded
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

# %% Import data from polynomial approximations.
# We fit the polynomial coefficients if no polynomial data exist yet, and we
# save them such that we do not need to do the fitting again.
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
            pathModelFolder, 'data4PolynomialFitting_{}.npy'.format(modelName))
        # Generate polynomial data
        if (not os.path.exists(path_data4PolynomialFitting) or 
            overwritedata4PolynomialFitting):
            import opensim
            from joblib import Parallel, delayed
            import multiprocessing
            # Get training data from motion file
            table = opensim.TimeSeriesTable(pathMotionFile4Polynomials)
            coordinates_table = list(table.getColumnLabels()) # w/ jointset/...
            data = table.getMatrix().to_numpy() # data in degrees w/o time
            pathModel = os.path.join(pathModelFolder, modelName + '.osim')
            # Set number of threads
            if nThreads == None:
                nThreads = multiprocessing.cpu_count()-2 # default
            if nThreads < 1:
                nThreads = 1
            elif nThreads > multiprocessing.cpu_count():
                nThreads = multiprocessing.cpu_count()                
            # Generate muscle tendon lengths and moment arms (in parallel)
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
            # Gather data
            lMT = np.zeros((data.shape[0], outputs[0][1].shape[1]))
            dM =  np.zeros((data.shape[0], outputs[0][1].shape[1], 
                            outputs[0][1].shape[2]))
            for i in range(len(outputs)):
                lMT[i*slice_size:(i+1)*slice_size, :] = outputs[i][0]
                dM[i*slice_size:(i+1)*slice_size, :, :] = outputs[i][1]
            if rest != 0:
                lMT[-rest:, :] = output_last[0]
                dM[-rest:, :, :] = output_last[1]
            # Put data in dict
            # muscles as ordered in model
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
            # Save data
            np.save(path_data4PolynomialFitting, data4PolynomialFitting)
        else:
            data4PolynomialFitting = np.load(path_data4PolynomialFitting, 
                                             allow_pickle=True).item()
        # Fit polynomial coefficients
        from polynomialsOpenSimAD import getPolynomialCoefficients
        polynomialData = getPolynomialCoefficients(
            data4PolynomialFitting, joints, muscles, side=side)
        if pathModelFolder != 0:
            np.save(pathPolynomialData, polynomialData)
           
    return polynomialData

def tendonCompliance(NSideMuscles):
    tendonCompliance = np.full((1, NSideMuscles), 35)
    
    return tendonCompliance

def tendonShift(NSideMuscles):
    tendonShift = np.full((1, NSideMuscles), 0)
    
    return tendonShift

def tendonCompliance_3D():
    tendonCompliance = np.full((1, 46), 35)
    
    return tendonCompliance

def tendonShift_3D():
    tendonShift = np.full((1, 46), 0)
    
    return tendonShift

def specificTension_3D(muscles):    
    
    sigma = {'glut_med1_r' : 0.74455,
             'glut_med2_r': 0.75395, 
             'glut_med3_r': 0.75057, 
             'glut_min1_r': 0.75, 
             'glut_min2_r': 0.75, 
             'glut_min3_r': 0.75116, 
             'semimem_r': 0.62524, 
             'semiten_r': 0.62121, 
             'bifemlh_r': 0.62222,
             'bifemsh_r': 1.00500, 
             'sar_r': 0.74286,
             'add_long_r': 0.74643, 
             'add_brev_r': 0.75263,
             'add_mag1_r': 0.55217,
             'add_mag2_r': 0.55323, 
             'add_mag3_r': 0.54831, 
             'tfl_r': 0.75161,
             'pect_r': 0.76000, 
             'grac_r': 0.73636, 
             'glut_max1_r': 0.75395, 
             'glut_max2_r': 0.74455, 
             'glut_max3_r': 0.74595, 
             'iliacus_r': 1.2477,
             'psoas_r': 1.5041,
             'quad_fem_r': 0.74706, 
             'gem_r': 0.74545, 
             'peri_r': 0.75254, 
             'rect_fem_r': 0.74936, 
             'vas_med_r': 0.49961, 
             'vas_int_r': 0.55263, 
             'vas_lat_r': 0.50027,
             'med_gas_r': 0.69865, 
             'lat_gas_r': 0.69694, 
             'soleus_r': 0.62703,
             'tib_post_r': 0.62520, 
             'flex_dig_r': 0.5, 
             'flex_hal_r': 0.50313,
             'tib_ant_r': 0.75417, 
             'per_brev_r': 0.62143,
             'per_long_r': 0.62450, 
             'per_tert_r': 1.0,
             'ext_dig_r': 0.75294,
             'ext_hal_r': 0.73636, 
             'ercspn_r': 0.25, 
             'intobl_r': 0.25, 
             'extobl_r': 0.25}
    
    specificTension = np.empty((1, len(muscles)))    
    for count, muscle in enumerate(muscles):
        specificTension[0, count] = sigma[muscle]
    
    return specificTension

def slowTwitchRatio_3D(muscles):    
    
    sigma = {'glut_med1_r' : 0.55,
             'glut_med2_r': 0.55, 
             'glut_med3_r': 0.55, 
             'glut_min1_r': 0.55, 
             'glut_min2_r': 0.55, 
             'glut_min3_r': 0.55, 
             'semimem_r': 0.4925, 
             'semiten_r': 0.425, 
             'bifemlh_r': 0.5425,
             'bifemsh_r': 0.529, 
             'sar_r': 0.50,
             'add_long_r': 0.50, 
             'add_brev_r': 0.50,
             'add_mag1_r': 0.552,
             'add_mag2_r': 0.552, 
             'add_mag3_r': 0.552, 
             'tfl_r': 0.50,
             'pect_r': 0.50, 
             'grac_r': 0.50, 
             'glut_max1_r': 0.55, 
             'glut_max2_r': 0.55, 
             'glut_max3_r': 0.55, 
             'iliacus_r': 0.50,
             'psoas_r': 0.50,
             'quad_fem_r': 0.50, 
             'gem_r': 0.50, 
             'peri_r': 0.50, 
             'rect_fem_r': 0.3865, 
             'vas_med_r': 0.503, 
             'vas_int_r': 0.543, 
             'vas_lat_r': 0.455,
             'med_gas_r': 0.566, 
             'lat_gas_r': 0.507, 
             'soleus_r': 0.803,
             'tib_post_r': 0.60, 
             'flex_dig_r': 0.60, 
             'flex_hal_r': 0.60,
             'tib_ant_r': 0.70, 
             'per_brev_r': 0.60,
             'per_long_r': 0.60, 
             'per_tert_r': 0.75,
             'ext_dig_r': 0.75,
             'ext_hal_r': 0.75, 
             'ercspn_r': 0.60,
             'intobl_r': 0.56, 
             'extobl_r': 0.58}
    
    slowTwitchRatio = np.empty((1, len(muscles)))    
    for count, muscle in enumerate(muscles):
        slowTwitchRatio[0, count] = sigma[muscle]
    
    return slowTwitchRatio

def passiveJointTorqueData_3D(joint, model_type='rajagopal2016'):    
    
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