'''
    ---------------------------------------------------------------------------
    OpenCap processing: utilsKinematics.py
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
import glob
import opensim
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from scipy.signal import find_peaks


from utilsProcessing import lowPassFilter
from utilsTRC import trc_2_dict

class kinematics:
    
    def __init__(self, dataDir, trialName, 
                 modelName='LaiUhlrich2022_scaled',
                 lowpass_cutoff_frequency_for_coordinate_values=-1):
        
        self.lowpass_cutoff_frequency_for_coordinate_values = (
            lowpass_cutoff_frequency_for_coordinate_values)
        
        # Model.
        opensim.Logger.setLevelString('error')
        modelPath = os.path.join(dataDir, 'OpenSimData', 'Model',
                                 '{}.osim'.format(modelName))
        self.model = opensim.Model(modelPath)
        self.model.initSystem()
        
        # Motion file with coordinate values.
        motionPath = os.path.join(dataDir, 'OpenSimData', 'Kinematics',
                                  '{}.mot'.format(trialName))
        
        # Create time-series table with coordinate values.             
        self.table = opensim.TimeSeriesTable(motionPath)        
        tableProcessor = opensim.TableProcessor(self.table)
        self.columnLabels = list(self.table.getColumnLabels())
        tableProcessor.append(opensim.TabOpUseAbsoluteStateNames())
        self.time = np.asarray(self.table.getIndependentColumn())
        
        
        # Filter coordinate values.
        if lowpass_cutoff_frequency_for_coordinate_values > 0:
            tableProcessor.append(
                opensim.TabOpLowPassFilter(
                    lowpass_cutoff_frequency_for_coordinate_values))

        # Convert in radians.
        self.table = tableProcessor.processAndConvertToRadians(self.model)
        
        # Trim if filtered.
        if lowpass_cutoff_frequency_for_coordinate_values > 0:
            time_temp = self.table.getIndependentColumn()            
            self.table.trim(
                time_temp[self.table.getNearestRowIndexForTime(self.time[0])],
                time_temp[self.table.getNearestRowIndexForTime(self.time[-1])])
                
        # Compute coordinate speeds and accelerations and add speeds to table.        
        self.Qs = self.table.getMatrix().to_numpy()
        self.Qds = np.zeros(self.Qs.shape)
        self.Qdds = np.zeros(self.Qs.shape)
        columnAbsoluteLabels = list(self.table.getColumnLabels())
        for i, columnLabel in enumerate(columnAbsoluteLabels):
            spline = interpolate.InterpolatedUnivariateSpline(
                self.time, self.Qs[:,i], k=3)
            # Coordinate speeds
            splineD1 = spline.derivative(n=1)
            self.Qds[:,i] = splineD1(self.time)
            # Coordinate accelerations.
            splineD2 = spline.derivative(n=2)
            self.Qdds[:,i] = splineD2(self.time)            
            # Add coordinate speeds to table.
            columnLabel_speed = columnLabel[:-5] + 'speed'
            self.table.appendColumn(
                columnLabel_speed, 
                opensim.Vector(self.Qds[:,i].flatten().tolist()))
            
        # Append missing muscle states to table.
        # Needed for StatesTrajectory.
        stateVariableNames = self.model.getStateVariableNames()
        stateVariableNamesStr = [
            stateVariableNames.get(i) for i in range(
                stateVariableNames.getSize())]
        existingLabels = self.table.getColumnLabels()
        for stateVariableNameStr in stateVariableNamesStr:
            if not stateVariableNameStr in existingLabels:
                vec_0 = opensim.Vector([0] * self.table.getNumRows())            
                self.table.appendColumn(stateVariableNameStr, vec_0)
                
        # Set state trajectory
        self.stateTrajectory = opensim.StatesTrajectory.createFromStatesTable(
            self.model, self.table)
        
        # Number of muscles.
        self.nMuscles = 0
        self.forceSet = self.model.getForceSet()
        for i in range(self.forceSet.getSize()):        
            c_force_elt = self.forceSet.get(i)  
            if 'Muscle' in c_force_elt.getConcreteClassName():
                self.nMuscles += 1
                
        # Coordinates.
        self.coordinateSet = self.model.getCoordinateSet()
        self.nCoordinates = self.coordinateSet.getSize()
        self.coordinates = [self.coordinateSet.get(i).getName() 
                            for i in range(self.nCoordinates)]
            
        # Find rotational and translational coords
        self.idxColumnTrLabels = [
            self.columnLabels.index(i) for i in self.coordinates if \
            self.coordinateSet.get(i).getMotionType() == 2]
        self.idxColumnRotLabels = [
            self.columnLabels.index(i) for i in self.coordinates if \
            self.coordinateSet.get(i).getMotionType() == 1]
        
        # TODO: hard coded
        self.rootCoordinates = [
            'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
            'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
        
        self.lumbarCoordinates = ['lumbar_extension', 'lumbar_bending', 
                                  'lumbar_rotation']
        
        self.armCoordinates = ['arm_flex_r', 'arm_add_r', 'arm_rot_r', 
                               'elbow_flex_r', 'pro_sup_r', 
                               'arm_flex_l', 'arm_add_l', 'arm_rot_l', 
                               'elbow_flex_l', 'pro_sup_l']
            
    def get_coordinate_values(self, in_degrees=True, 
                              lowpass_cutoff_frequency=-1):
        
        # Convert to degrees.
        if in_degrees:
            Qs = np.zeros((self.Qs.shape))
            Qs[:, self.idxColumnTrLabels] = self.Qs[:, self.idxColumnTrLabels]
            Qs[:, self.idxColumnRotLabels] = (
                self.Qs[:, self.idxColumnRotLabels] * 180 / np.pi)
        else:
            Qs = self.Qs
            
        # Filter.
        if lowpass_cutoff_frequency > 0:
            Qs = lowPassFilter(self.time, Qs, lowpass_cutoff_frequency)
            if self.lowpass_cutoff_frequency_for_coordinate_values > 0:
                print("Warning: You are filtering the coordinate values a second time; coordinate values were filtered when creating your class object.")
        
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), Qs), axis=1)
        columns = ['time'] + self.columnLabels            
        coordinate_values = pd.DataFrame(data=data, columns=columns)
        
        return coordinate_values
    
    def get_coordinate_speeds(self, in_degrees=True, 
                              lowpass_cutoff_frequency=-1):
        
        # Convert to degrees.
        if in_degrees:
            Qds = np.zeros((self.Qds.shape))
            Qds[:, self.idxColumnTrLabels] = (
                self.Qds[:, self.idxColumnTrLabels])
            Qds[:, self.idxColumnRotLabels] = (
                self.Qds[:, self.idxColumnRotLabels] * 180 / np.pi)
        else:
            Qds = self.Qds
            
        # Filter.
        if lowpass_cutoff_frequency > 0:
            Qds = lowPassFilter(self.time, Qds, lowpass_cutoff_frequency)
        
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), Qds), axis=1)
        columns = ['time'] + self.columnLabels            
        coordinate_speeds = pd.DataFrame(data=data, columns=columns)
        
        return coordinate_speeds
    
    def get_coordinate_accelerations(self, in_degrees=True, 
                                     lowpass_cutoff_frequency=-1):
        
        # Convert to degrees.
        if in_degrees:
            Qdds = np.zeros((self.Qdds.shape))
            Qdds[:, self.idxColumnTrLabels] = (
                self.Qdds[:, self.idxColumnTrLabels])
            Qdds[:, self.idxColumnRotLabels] = (
                self.Qdds[:, self.idxColumnRotLabels] * 180 / np.pi)
        else:
            Qdds = self.Qdds
            
        # Filter.
        if lowpass_cutoff_frequency > 0:
            Qdds = lowPassFilter(self.time, Qdds, lowpass_cutoff_frequency)
        
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), Qdds), axis=1)
        columns = ['time'] + self.columnLabels            
        coordinate_accelerations = pd.DataFrame(data=data, columns=columns)
        
        return coordinate_accelerations
    
    def get_muscle_tendon_lengths(self, lowpass_cutoff_frequency=-1):
        
        # Compute muscle-tendon lengths.
        lMT = np.zeros((self.table.getNumRows(), self.nMuscles))
        for i in range(self.table.getNumRows()):
            self.model.realizePosition(self.stateTrajectory[i])
            if i == 0:
                muscleNames = [] 
            for m in range(self.forceSet.getSize()):        
                c_force_elt = self.forceSet.get(m)  
                if 'Muscle' in c_force_elt.getConcreteClassName():
                    cObj = opensim.Muscle.safeDownCast(c_force_elt)            
                    lMT[i,m] = cObj.getLength(self.stateTrajectory[i])
                    if i == 0:
                        muscleNames.append(c_force_elt.getName())
                        
        # Filter.
        if lowpass_cutoff_frequency > 0:
            lMT = lowPassFilter(self.time, lMT, lowpass_cutoff_frequency)                        
              
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), lMT), axis=1)
        columns = ['time'] + muscleNames               
        muscle_tendon_lengths = pd.DataFrame(data=data, columns=columns)
        
        return muscle_tendon_lengths
    
    def get_moment_arms(self, lowpass_cutoff_frequency=-1):
        
        # Compute moment arms.
        dM =  np.zeros((self.table.getNumRows(), self.nMuscles, 
                        self.nCoordinates))
        for i in range(self.table.getNumRows()):            
            self.model.realizePosition(self.stateTrajectory[i])
            if i == 0:
                muscleNames = []
            for m in range(self.forceSet.getSize()):        
                c_force_elt = self.forceSet.get(m)  
                if 'Muscle' in c_force_elt.getConcreteClassName():
                    muscleName = c_force_elt.getName()
                    cObj = opensim.Muscle.safeDownCast(c_force_elt)
                    if i == 0:
                        muscleNames.append(c_force_elt.getName())                    
                    for c, coord in enumerate(self.coordinates):
                        # We use prior knowledge to improve computation speed;
                        # We do not want to compute moment arms that are not
                        # relevant, eg for a muscle of the left side with 
                        # respect to a coordinate of the right side.
                        if muscleName[-2:] == '_l' and coord[-2:] == '_r':
                            dM[i, m, c] = 0
                        elif muscleName[-2:] == '_r' and coord[-2:] == '_l':
                            dM[i, m, c] = 0
                        elif (coord in self.rootCoordinates or 
                              coord in self.lumbarCoordinates or 
                              coord in self.armCoordinates):
                            dM[i, m, c] = 0
                        else:
                            coordinate = self.coordinateSet.get(
                                self.coordinates.index(coord))
                            dM[i, m, c] = cObj.computeMomentArm(
                                self.stateTrajectory[i], coordinate)
                            
        # Clean numerical artefacts (ie, moment arms smaller than 1e-5 m).
        dM[np.abs(dM) < 1e-5] = 0
        
        # Filter.
        if lowpass_cutoff_frequency > 0:            
            for c, coord in enumerate(self.coordinates):
                dM[:, :, c] = lowPassFilter(self.time, dM[:, :, c], 
                                            lowpass_cutoff_frequency)
        
        # Return as DataFrame.
        moment_arms = {}
        for c, coord in enumerate(self.coordinates):
            data = np.concatenate(
                (np.expand_dims(self.time, axis=1), dM[:,:,c]), axis=1)
            columns = ['time'] + muscleNames
            moment_arms[coord] = pd.DataFrame(data=data, columns=columns)
            
        return moment_arms
    
    def compute_center_of_mass(self):        
        
        # Compute center of mass position and velocity.
        self.com_values = np.zeros((self.table.getNumRows(),3))
        self.com_speeds = np.zeros((self.table.getNumRows(),3))        
        for i in range(self.table.getNumRows()):            
            self.model.realizeVelocity(self.stateTrajectory[i])
            self.com_values[i,:] = self.model.calcMassCenterPosition(
                self.stateTrajectory[i]).to_numpy()
            self.com_speeds[i,:] = self.model.calcMassCenterVelocity(
                self.stateTrajectory[i]).to_numpy()
            
    def get_center_of_mass_values(self, lowpass_cutoff_frequency=-1):
        
        self.compute_center_of_mass()        
        com_v = self.com_values
        
        # Filter.
        if lowpass_cutoff_frequency > 0:
            com_v = lowPassFilter(self.time, com_v, lowpass_cutoff_frequency)                        
              
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), com_v), axis=1)
        columns = ['time'] + ['x','y','z']               
        com_values = pd.DataFrame(data=data, columns=columns)
        
        return com_values
    
    def get_center_of_mass_speeds(self, lowpass_cutoff_frequency=-1):
        
        self.compute_center_of_mass()        
        com_s = self.com_speeds
        
        # Filter.
        if lowpass_cutoff_frequency > 0:
            com_s = lowPassFilter(self.time, com_s, lowpass_cutoff_frequency)                        
              
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), com_s), axis=1)
        columns = ['time'] + ['x','y','z']               
        com_speeds = pd.DataFrame(data=data, columns=columns)
        
        return com_speeds
    
    def get_center_of_mass_accelerations(self, lowpass_cutoff_frequency=-1):
        
        self.compute_center_of_mass()        
        com_s = self.com_speeds
        
        # Accelerations are first time derivative of speeds.
        com_a = np.zeros((com_s.shape))
        for i in range(com_s.shape[1]):
            spline = interpolate.InterpolatedUnivariateSpline(
                self.time, com_s[:,i], k=3)
            splineD1 = spline.derivative(n=1)
            com_a[:,i] = splineD1(self.time)        
        
        # Filter.
        if lowpass_cutoff_frequency > 0:
            com_a = lowPassFilter(self.time, com_a, lowpass_cutoff_frequency)                        
              
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), com_a), axis=1)
        columns = ['time'] + ['x','y','z']               
        com_accelerations = pd.DataFrame(data=data, columns=columns)
        
        return com_accelerations        

# %%
class gait_analysis:
    
    def __init__(self, dataDir, trialName, 
                 lowpass_cutoff_frequency_for_coordinate_values=-1,
                 n_gait_cycles=1):
                
        self.lowpass_cutoff_frequency_for_coordinate_values = (
            lowpass_cutoff_frequency_for_coordinate_values)
        
        # Model.
        opensim.Logger.setLevelString('error')
        modelPath = glob.glob(os.path.join(dataDir, 'OpenSimData', 'Model',
                                 '*.osim'))[0]
        self.model = opensim.Model(modelPath)
        self.model.initSystem()
        
        # Marker data
        trcFilePath = os.path.join(dataDir,'MarkerData',
                                   '{}.trc'.format(trialName))
        self.markerDict = trc_2_dict(trcFilePath)
        
        # Motion file with coordinate values.
        motionPath = os.path.join(dataDir, 'OpenSimData', 'Kinematics',
                                  '{}.mot'.format(trialName))
        
        # Create time-series table with coordinate values.             
        self.table = opensim.TimeSeriesTable(motionPath)        
        tableProcessor = opensim.TableProcessor(self.table)
        self.columnLabels = list(self.table.getColumnLabels())
        tableProcessor.append(opensim.TabOpUseAbsoluteStateNames())
        self.time = np.asarray(self.table.getIndependentColumn())   
        
        # Filter coordinate values.
        if lowpass_cutoff_frequency_for_coordinate_values > 0:
            tableProcessor.append(
                opensim.TabOpLowPassFilter(
                    lowpass_cutoff_frequency_for_coordinate_values))

        # Process
        self.table = tableProcessor.process(self.model)
        
        # Trim if filtered.
        if lowpass_cutoff_frequency_for_coordinate_values > 0:
            time_temp = self.table.getIndependentColumn()            
            self.table.trim(
                time_temp[self.table.getNearestRowIndexForTime(self.time[0])],
                time_temp[self.table.getNearestRowIndexForTime(self.time[-1])])         
                
        self.Qs = self.table.getMatrix().to_numpy()
                
        # Coordinates.
        self.coordinateSet = self.model.getCoordinateSet()
        self.nCoordinates = self.coordinateSet.getSize()
        self.coordinates = [self.coordinateSet.get(i).getName() 
                            for i in range(self.nCoordinates)]
        
        # Find rotational and translational coords
        self.idxColumnTrLabels = [
            self.columnLabels.index(i) for i in self.coordinates if \
            self.coordinateSet.get(i).getMotionType() == 2]
        self.idxColumnRotLabels = [
            self.columnLabels.index(i) for i in self.coordinates if \
            self.coordinateSet.get(i).getMotionType() == 1]
            
        self.coordinateValues = self.get_coordinate_values()
        
        # Segment gait cycles
        self.gaitEvents = self.segment_walking(n_gait_cycles=n_gait_cycles)
        
        # determine treadmill speed (0 if overground)
        self.treadmillSpeed = self.calc_treadmill_speed()
                
    def calc_scalars(self,scalarNames):
        
        # verify that scalarNames are methods in gait_analysis        
        method_names = [func for func in dir(self) if callable(getattr(self, func))]
        nonexistant_methods = [entry for entry in scalarNames if 'calc_' + entry not in method_names]
        
        if len(nonexistant_methods) > 0:
            raise Exception(str(['calc_' + a for a in nonexistant_methods]) + ' not in gait_analysis class.')
        
        scalarDict = {}
        for scalarName in scalarNames:
            thisFunction = getattr(self, 'calc_' + scalarName)
            scalarDict[scalarName] = thisFunction()
        
        return scalarDict
    
    def calc_stride_length(self):

        # get calc positions based on self.gaitEvents['leg'] from self.markerDict
        if self.gaitEvents['ipsilateralLeg'] == 'r':
            leg = 'r'
        else:
            leg = 'L'
        calc_position = self.markerDict['markers'][leg + '_calc_study']

        # find stride length on treadmill
        # difference in ipsilateral calcaneus position at heel strike + treadmill speed * time
        strideLength = np.linalg.norm(calc_position[self.gaitEvents['ipsilateralIdx'][:,:1]] - \
                           calc_position[self.gaitEvents['ipsilateralIdx'][:,2:3]], axis=2) + \
                           self.treadmillSpeed * np.diff(self.gaitEvents['ipsilateralTime'][:,(0,2)])
        
        # average across all strides
        strideLength = np.mean(strideLength)
        
        return strideLength
    
    def calc_gait_speed(self):
        pelvis_position = np.vstack((self.coordinateValues['pelvis_tx'],
                                     self.coordinateValues['pelvis_ty'], 
                                     self.coordinateValues['pelvis_tz'])).T

        gait_speed = (np.linalg.norm(pelvis_position[self.gaitEvents['ipsilateralIdx'][:,:1]] - \
                           pelvis_position[self.gaitEvents['ipsilateralIdx'][:,2:3]], axis=2)) / \
                           np.diff(self.gaitEvents['ipsilateralTime'][:,(0,2)]) + self.treadmillSpeed 
        
        # average across all strides
        gait_speed = np.mean(gait_speed)
        
        return gait_speed
    
    def calc_treadmill_speed(self):
        if self.gaitEvents['ipsilateralLeg'] == 'r':
            leg = 'r'
        else:
            leg = 'L'
        toe_position = self.markerDict['markers'][leg + '_toe_study']
        
        stanceLength = np.round(np.diff(self.gaitEvents['ipsilateralIdx'][:,:2]))
        stanceTime = np.diff(self.gaitEvents['ipsilateralTime'][:,:2])
        startIdx = np.round(self.gaitEvents['ipsilateralIdx'][:,:1]+.3*stanceLength).astype(int)
        endIdx = np.round(self.gaitEvents['ipsilateralIdx'][:,1:2]-.3*stanceLength).astype(int)
        
        toeDistanceStance = np.linalg.norm(toe_position[startIdx] - \
                           toe_position[endIdx], axis=2)
        
        treadmillSpeed = np.mean(toeDistanceStance/stanceTime)
        # overground
        if treadmillSpeed < .2:
            treadmillSpeed = 0
                           
        return treadmillSpeed

    def segment_walking(self, n_gait_cycles=1, filterFreq=6,leg='auto',
                        visualize=False):
        # subtract sacrum from foot
        # visually, it looks like the position-based approach will be more robust
        r_calc_rel_x = (self.markerDict['markers']['r_calc_study'] - self.markerDict[
                                     'markers']['r.PSIS_study'])[:,0]
        r_calc_rel_x = lowPassFilter(self.time, r_calc_rel_x, filterFreq)

        r_toe_rel_x = (self.markerDict['markers']['r_toe_study'] - self.markerDict[
                                    'markers']['r.PSIS_study'])[:,0]
        r_toe_rel_x = lowPassFilter(self.time, r_toe_rel_x, filterFreq)  

        # repeat for left
        l_calc_rel_x = (self.markerDict['markers']['L_calc_study'] - self.markerDict[
                                     'markers']['L.PSIS_study'])[:,0]
        l_calc_rel_x = lowPassFilter(self.time, l_calc_rel_x, filterFreq)

        l_toe_rel_x = (self.markerDict['markers']['L_toe_study'] - self.markerDict[
                                    'markers']['L.PSIS_study'])[:,0]
        l_toe_rel_x = lowPassFilter(self.time, l_toe_rel_x, filterFreq)                               
        
        # Find HS
        rHS, _ = find_peaks(r_calc_rel_x)
        lHS, _ = find_peaks(l_calc_rel_x)
        
        # Find TO
        rTO, _ = find_peaks(-r_toe_rel_x)
        lTO, _ = find_peaks(-l_toe_rel_x)
        
        if visualize==True:
            import matplotlib.pyplot as plt
            plt.close('all')
            plt.figure(1)
            plt.plot(self.markerDict['time'],r_toe_rel_x,label='toe')
            plt.plot(self.markerDict['time'],r_calc_rel_x,label='calc')
            plt.scatter(self.markerDict['time'][rHS], r_calc_rel_x[rHS], color='red', label='rHS')
            plt.scatter(self.markerDict['time'][rTO], r_toe_rel_x[rTO], color='blue', label='rTO')
            plt.legend()

            plt.figure(2)
            plt.plot(self.markerDict['time'],l_toe_rel_x,label='toe')
            plt.plot(self.markerDict['time'],l_calc_rel_x,label='calc')
            plt.scatter(self.markerDict['time'][lHS], l_calc_rel_x[lHS], color='red', label='lHS')
            plt.scatter(self.markerDict['time'][lTO], l_toe_rel_x[lTO], color='blue', label='lTO')
            plt.legend()

        
        # find the number of gait cycles for the foot of interest
        if leg=='auto':
            # find the last HS of either foot
            if rHS[-1] > lHS[-1]:
                leg = 'r'
            else:
                leg = 'l'
        
        # find the number of gait cycles for the foot of interest
        if leg == 'r':
            hsIps = rHS
            toIps = rTO
            hsCont = lHS
            toCont = lTO
        elif leg == 'l':
            hsIps = lHS
            toIps = lTO
            hsCont = rHS
            toCont = rTO

        n_gait_cycles = np.min([n_gait_cycles, len(hsIps)-1])
        gaitEvents_ips = np.zeros((n_gait_cycles, 3),dtype=np.int)
        gaitEvents_cont = np.zeros((n_gait_cycles, 2),dtype=np.int)
        if n_gait_cycles <1:
            raise Exception('Not enough gait cycles found.')

        for i in range(n_gait_cycles):
            # ipsilateral HS, TO, HS
            gaitEvents_ips[i,0] = hsIps[-i-2]
            gaitEvents_ips[i,1] = toIps[-i-1]
            gaitEvents_ips[i,2] = hsIps[-i-1]

            # contralateral TO, HS
            # iterate in reverse through contralateral HS and TO, finding the one that is within the range of gaitEvents_ips
            hsContFound = False
            toContFound = False
            for j in range(len(toCont)):
                if toCont[-j-1] > gaitEvents_ips[i,0] and toCont[-j-1] < gaitEvents_ips[i,2] and not toContFound:
                    gaitEvents_cont[i,0] = toCont[-j-1]
                    toContFound = True
                    
            for j in range(len(hsCont)):
                if hsCont[-j-1] > gaitEvents_ips[i,0] and hsCont[-j-1] < gaitEvents_ips[i,2] and not hsContFound:
                    gaitEvents_cont[i,1] = hsCont[-j-1]
                    hsContFound = True
            
            # making contralateral gait events optional
            if not toContFound or not hsContFound:                   
                raise Warning('Could not find contralateral gait event within ipsilateral gait event range.')
                gaitEvents_cont[i,0] = np.nan
                gaitEvents_cont[i,1] = np.nan
            
            # convert gaitEvents to times using self.markerDict['time']
            gaitEventTimes_ips = self.markerDict['time'][gaitEvents_ips]
            gaitEventTimes_cont = self.markerDict['time'][gaitEvents_cont]
                            
        gaitEvents = {'ipsilateralIdx':gaitEvents_ips,
                      'contralateralIdx':gaitEvents_cont,
                      'ipsilateralTime':gaitEventTimes_ips,
                      'contralateralTime':gaitEventTimes_cont,
                      'eventNamesIpsilateral':['HS','TO','HS'],
                      'eventNamesContralateral':['TO','HS'],
                      'ipsilateralLeg':leg}
        
        return gaitEvents       
       
    def get_coordinate_values(self, in_degrees=True, 
                              lowpass_cutoff_frequency=-1):
        
        # Convert to degrees.
        if in_degrees:
            Qs = np.zeros((self.Qs.shape))
            Qs[:, self.idxColumnTrLabels] = self.Qs[:, self.idxColumnTrLabels]
            Qs[:, self.idxColumnRotLabels] = (
                self.Qs[:, self.idxColumnRotLabels] * 180 / np.pi)
        else:
            Qs = self.Qs
            
        # Filter.
        if lowpass_cutoff_frequency > 0:
            Qs = lowPassFilter(self.time, Qs, lowpass_cutoff_frequency)
            if self.lowpass_cutoff_frequency_for_coordinate_values > 0:
                print("Warning: You are filtering the coordinate values a second time; coordinate values were filtered when creating your class object.")
        
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), Qs), axis=1)
        columns = ['time'] + self.columnLabels            
        coordinate_values = pd.DataFrame(data=data, columns=columns)
        
        return coordinate_values