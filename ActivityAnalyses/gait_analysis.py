"""
    ---------------------------------------------------------------------------
    OpenCap processing: gaitAnalysis.py
    ---------------------------------------------------------------------------

    Copyright 2023 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import sys
sys.path.append('../')
                
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from utilsKinematics import kinematics


class gait_analysis(kinematics):
    
    def __init__(self, session_dir, trial_name, leg='auto',
                 lowpass_cutoff_frequency_for_coordinate_values=-1,
                 n_gait_cycles=-1):
        
        # Inherit init from kinematics class.
        super().__init__(
            session_dir, 
            trial_name, 
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values)
                        
        # Marker data load and filter.
        self.markerDict= self.get_marker_dict(session_dir, trial_name, 
                            lowpass_cutoff_frequency = lowpass_cutoff_frequency_for_coordinate_values)

        # Coordinate values.
        self.coordinateValues = self.get_coordinate_values()
        
        # Segment gait cycles.
        self.gaitEvents = self.segment_walking(n_gait_cycles=n_gait_cycles,leg=leg)
        self.nGaitCycles = np.shape(self.gaitEvents['ipsilateralIdx'])[0]
        
        # Determine treadmill speed (0 if overground).
        self.treadmillSpeed,_ = self.compute_treadmill_speed()
        
        # Initialize variables to be lazy loaded.
        self._comValues = None
        self._R_world_to_gait = None
    
    # Compute COM trajectory.
    def comValues(self):
        if self._comValues is None:
            self._comValues = self.get_center_of_mass_values()
        return self._comValues
    
    # Compute gait frame.
    def R_world_to_gait(self):
        if self._R_world_to_gait is None:
            self._R_world_to_gait = self.compute_gait_frame()
        return self._R_world_to_gait
    
    def get_gait_events(self):
        
        return self.gaitEvents
        
    
    def compute_scalars(self,scalarNames):
               
        # Verify that scalarNames are methods in gait_analysis.
        method_names = [func for func in dir(self) if callable(getattr(self, func))]
        possibleMethods = [entry for entry in method_names if 'compute_' in entry]
        
        if scalarNames is None:
            print('No scalars defined, these methods are available:')
            print(*possibleMethods)
            return
        
        nonexistant_methods = [entry for entry in scalarNames if 'compute_' + entry not in method_names]
        
        if len(nonexistant_methods) > 0:
            raise Exception(str(['compute_' + a for a in nonexistant_methods]) + ' does not exist in gait_analysis class.')
        
        scalarDict = {}
        for scalarName in scalarNames:
            thisFunction = getattr(self, 'compute_' + scalarName)
            scalarDict[scalarName] = {}
            (scalarDict[scalarName]['value'],
                scalarDict[scalarName]['units']) = thisFunction()
        
        return scalarDict
    
    def compute_stride_length(self):
        
        leg,_ = self.get_leg()
        
        calc_position = self.markerDict['markers'][leg + '_calc_study']

        # On treadmill, the stride length is the difference in ipsilateral
        # calcaneus position at heel strike + treadmill speed * time.
        strideLength = (
            np.linalg.norm(
                calc_position[self.gaitEvents['ipsilateralIdx'][:,:1]] - 
                calc_position[self.gaitEvents['ipsilateralIdx'][:,2:3]], axis=2) + 
                self.treadmillSpeed * np.diff(self.gaitEvents['ipsilateralTime'][:,(0,2)]))
        
        # Average across all strides.
        strideLength = np.mean(strideLength)
        
        # Define units.
        units = 'm'
        
        return strideLength, units
    
    def compute_step_length(self,report_all=False):
        leg, contLeg = self.get_leg()
        step_lengths = {}
        
        step_lengths[contLeg.lower()] = (np.linalg.norm(
            self.markerDict['markers'][leg + '_calc_study'][self.gaitEvents['ipsilateralIdx'][:,:1]] - 
            self.markerDict['markers'][contLeg + '_calc_study'][self.gaitEvents['contralateralIdx'][:,1:2]], axis=2) + 
            self.treadmillSpeed * (self.gaitEvents['contralateralTime'][:,1:2] -
                                   self.gaitEvents['ipsilateralTime'][:,:1]))
        
        step_lengths[leg.lower()]  = (np.linalg.norm(
            self.markerDict['markers'][leg + '_calc_study'][self.gaitEvents['ipsilateralIdx'][:,2:]] - 
            self.markerDict['markers'][contLeg + '_calc_study'][self.gaitEvents['contralateralIdx'][:,1:2]], axis=2) + 
            self.treadmillSpeed * (-self.gaitEvents['contralateralTime'][:,1:2] +
                                   self.gaitEvents['ipsilateralTime'][:,2:]))
               
        # Average across all strides.
        step_length = {key: np.mean(values) for key, values in step_lengths.items()}
        
        # Define units.
        units = 'm'
        
        # some functions depend on having values for each step, otherwise return average
        if report_all:
            return step_lengths, units
        else:
            return step_length, units
        
    def compute_step_length_symmetry(self):
        step_lengths,units = self.compute_step_length(report_all=True)
        
        step_length_symmetry = step_lengths['r'] / step_lengths['l'] * 100
        
        # Average across strides
        step_length_symmetry = np.mean(step_length_symmetry)
        
        # define units 
        units = '% (R/L)'
        
        return step_length_symmetry, units
    
    def compute_gait_speed(self):
                           
        comValuesArray = np.vstack((self.comValues()['x'],self.comValues()['y'],self.comValues()['z'])).T
        gait_speed = (
            np.linalg.norm(
                comValuesArray[self.gaitEvents['ipsilateralIdx'][:,:1]] -
                comValuesArray[self.gaitEvents['ipsilateralIdx'][:,2:3]], axis=2) /
                np.diff(self.gaitEvents['ipsilateralTime'][:,(0,2)]) + self.treadmillSpeed) 
        
        # Average across all strides.
        gait_speed = np.mean(gait_speed)
        
        # Define units.
        units = 'm/s'
        
        return gait_speed, units
    
    def compute_cadence(self):
        
        # In steps per minute.
        cadence = 60*2/np.diff(self.gaitEvents['ipsilateralTime'][:,(0,2)])
        
        # Average across all strides.
        cadence = np.mean(cadence)
        
        # Define units.
        units = 'steps/min'
        
        return cadence, units
        
    def compute_treadmill_speed(self, overground_speed_threshold=0.3):
        
        leg,_ = self.get_leg()
        
        foot_position = self.markerDict['markers'][leg + '_ankle_study']
        
        stanceTimeLength = np.round(np.diff(self.gaitEvents['ipsilateralIdx'][:,:2]))
        startIdx = np.round(self.gaitEvents['ipsilateralIdx'][:,:1]+.1*stanceTimeLength).astype(int)
        endIdx = np.round(self.gaitEvents['ipsilateralIdx'][:,1:2]-.3*stanceTimeLength).astype(int)
            
        # Average instantaneous velocities.
        dt = np.diff(self.markerDict['time'][:2])[0]
        for i in range(self.nGaitCycles):
            footVel = np.linalg.norm(np.mean(np.diff(
                foot_position[startIdx[i,0]:endIdx[i,0],:],axis=0),axis=0)/dt)
        
        treadmillSpeed = np.mean(footVel)
        
        # Overground.
        if treadmillSpeed < overground_speed_threshold:
            treadmillSpeed = 0
            
        # Define units.
        units = 'm/s'
                           
        return treadmillSpeed, units
    
    def compute_step_width(self):
        
        leg,contLeg = self.get_leg()
        
        # Get ankle joint center positions.
        ankle_position_ips = (
            self.markerDict['markers'][leg + '_ankle_study'] + 
            self.markerDict['markers'][leg + '_mankle_study'])/2
        ankle_position_cont = (
            self.markerDict['markers'][contLeg + '_ankle_study'] + 
            self.markerDict['markers'][contLeg + '_mankle_study'])/2        
        ankleVector = (
            ankle_position_cont[self.gaitEvents['contralateralIdx'][:,1]] - 
            ankle_position_ips[self.gaitEvents['ipsilateralIdx'][:,0]])
                      
        ankleVector_inGaitFrame = np.array(
            [np.dot(ankleVector[i,:], self.R_world_to_gait()[i,:,:]) 
             for i in range(self.nGaitCycles)])
        
        # Step width is z distance.
        stepWidth = np.abs(ankleVector_inGaitFrame[:,2])
        
        # Average across all strides.
        stepWidth = np.mean(stepWidth)
        
        # Define units.
        units = 'm'
        
        return stepWidth, units
    
    def compute_stance_time(self):
        
        stanceTime = np.diff(self.gaitEvents['ipsilateralTime'][:,:2])
        
        # Average across all strides.
        stanceTime = np.mean(stanceTime)
        
        # Define units.
        units = 's'
        
        return stanceTime, units
    
    def compute_swing_time(self):
        
        swingTime = np.diff(self.gaitEvents['ipsilateralTime'][:,1:])
        
        # Average across all strides.
        swingTime = np.mean(swingTime)
        
        # Define units.
        units = 's'
        
        return swingTime, units
    
    def compute_single_support_time(self):
        
        double_support_time,_ = self.compute_double_support_time(return_all_trials=True) 

        singleSupportTime = 100 - double_support_time    
        
        # Average across all strides.
        singleSupportTime = np.mean(singleSupportTime)
        
        # Define units.
        units = '%'
        
        return singleSupportTime, units
        
    def compute_double_support_time(self,return_all_trials=False):
        
        # Ipsilateral stance time - contralateral swing time.
        doubleSupportTimes = (
            (np.diff(self.gaitEvents['ipsilateralTime'][:,:2]) - 
            np.diff(self.gaitEvents['contralateralTime'][:,:2])) /
            np.diff(self.gaitEvents['ipsilateralTime'][:,(0,2)])) * 100
                            
        # Average across all strides.
        doubleSupportTime = np.mean(doubleSupportTimes)
        
        # Define units.
        units = '%'
        
        if return_all_trials:
            return doubleSupportTimes, units
        else:
            return doubleSupportTime, units
            
    def compute_gait_frame(self):

        # Create frame for each gait cycle with x: pelvis heading, 
        # z: average vector between ASIS during gait cycle, y: cross.
        
        # Pelvis center trajectory (for overground heading vector).
        pelvisMarkerNames = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study']
        pelvisMarkers = [self.markerDict['markers'][mkr]  for mkr in pelvisMarkerNames]
        pelvisCenter = np.mean(np.array(pelvisMarkers),axis=0)
        
        # Ankle trajectory (for treadmill heading vector).
        leg = self.gaitEvents['ipsilateralLeg']
        if leg == 'l': leg='L'
        anklePos = self.markerDict['markers'][leg + '_ankle_study']
        
        # Vector from left ASIS to right ASIS (for mediolateral direction).
        asisMarkerNames = ['L.ASIS_study','r.ASIS_study']
        asisMarkers = [self.markerDict['markers'][mkr]  for mkr in asisMarkerNames]
        asisVector = np.squeeze(np.diff(np.array(asisMarkers),axis=0))
        
        # Heading vector per gait cycle.
        # If overground, use pelvis center trajectory; treadmill: ankle trajectory.
        if self.treadmillSpeed == 0:
            x = np.diff(pelvisCenter[self.gaitEvents['ipsilateralIdx'][:,(0,2)],:],axis=1)[:,0,:]
            x = x / np.linalg.norm(x,axis=1,keepdims=True)
        else: 
            x = np.zeros((self.nGaitCycles,3))
            for i in range(self.nGaitCycles):
                x[i,:] = anklePos[self.gaitEvents['ipsilateralIdx'][i,2]] - \
                         anklePos[self.gaitEvents['ipsilateralIdx'][i,1]]
            x = x / np.linalg.norm(x,axis=1,keepdims=True)
            
        # Mean ASIS vector over gait cycle.
        z = np.zeros((self.nGaitCycles,3))
        for i in range(self.nGaitCycles):
            z[i,:] = np.mean(asisVector[self.gaitEvents['ipsilateralIdx'][i,0]: \
                             self.gaitEvents['ipsilateralIdx'][i,2]],axis=0)
        z = z / np.linalg.norm(z,axis=1,keepdims=True)
        
        # Cross to get y.
        y = np.cross(z,x)
        
        # 3x3xnSteps.
        R_lab_to_gait = np.stack((x.T,y.T,z.T),axis=1).transpose((2, 0, 1))
        
        return R_lab_to_gait
    
    def get_leg(self):

        if self.gaitEvents['ipsilateralLeg'] == 'r':
            leg = 'r'
            contLeg = 'L'
        else:
            leg = 'L'
            contLeg = 'r'
            
        return leg, contLeg
    
    def get_coordinates_normalized_time(self):
        
        colNames = self.coordinateValues.columns
        data = self.coordinateValues.to_numpy(copy=True)
        coordValuesNorm = []
        for i in range(self.nGaitCycles):
            coordValues = data[self.gaitEvents['ipsilateralIdx'][i,0]:self.gaitEvents['ipsilateralIdx'][i,2]]
            coordValuesNorm.append(np.stack([np.interp(np.linspace(0,100,101),
                                   np.linspace(0,100,len(coordValues)),coordValues[:,i]) \
                                   for i in range(coordValues.shape[1])],axis=1))
             
        coordinateValuesTimeNormalized = {}
        # Average.
        coordVals_mean = np.mean(np.array(coordValuesNorm),axis=0)
        coordinateValuesTimeNormalized['mean'] = pd.DataFrame(data=coordVals_mean, columns=colNames)
        
        # Standard deviation.
        if self.nGaitCycles >2:
            coordVals_sd = np.std(np.array(coordValuesNorm), axis=0)
            coordinateValuesTimeNormalized['sd'] = pd.DataFrame(data=coordVals_sd, columns=colNames)
        else:
            coordinateValuesTimeNormalized['sd'] = None
        
        # Return to dataframe.
        coordinateValuesTimeNormalized['indiv'] = [pd.DataFrame(data=d, columns=colNames) for d in coordValuesNorm]
        
        return coordinateValuesTimeNormalized

    def segment_walking(self, n_gait_cycles=-1, leg='auto', visualize=False):

        # n_gait_cycles = -1 finds all accessible gait cycles. Otherwise, it 
        # finds that many gait cycles, working backwards from end of trial.
        
        # Subtract sacrum from foot.
        # It looks like the position-based approach will be more robust.
        r_calc_rel_x = (
            self.markerDict['markers']['r_calc_study'] - 
            self.markerDict['markers']['r.PSIS_study'])[:,0]
        r_toe_rel_x = (
            self.markerDict['markers']['r_toe_study'] - 
            self.markerDict['markers']['r.PSIS_study'])[:,0]

        # Repeat for left.
        l_calc_rel_x = (
            self.markerDict['markers']['L_calc_study'] - 
            self.markerDict['markers']['L.PSIS_study'])[:,0]
        l_toe_rel_x = (
            self.markerDict['markers']['L_toe_study'] - 
            self.markerDict['markers']['L.PSIS_study'])[:,0]
        
        # Find HS.
        rHS, _ = find_peaks(r_calc_rel_x, prominence=0.3)
        lHS, _ = find_peaks(l_calc_rel_x, prominence=0.3)
        
        # Find TO.
        rTO, _ = find_peaks(-r_toe_rel_x, prominence=0.3)
        lTO, _ = find_peaks(-l_toe_rel_x, prominence=0.3)
        
        if visualize:
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

        # Find the number of gait cycles for the foot of interest.
        if leg=='auto':
            # Find the last HS of either foot.
            if rHS[-1] > lHS[-1]:
                leg = 'r'
            else:
                leg = 'l'
        
        # Find the number of gait cycles for the foot of interest.
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

        if len(hsIps)-1 < n_gait_cycles:
            print('You requested {} gait cycles, but only {} were found. '
                  'Proceeding with this number.'.format(n_gait_cycles,len(hsIps)-1))
            n_gait_cycles = len(hsIps)-1
        if n_gait_cycles == -1:
            n_gait_cycles = len(hsIps)-1
            print('Processing {} gait cycles, leg: '.format(n_gait_cycles) + leg + '.')
            
        # Ipsilateral gait events: heel strike, toe-off, heel strike.
        gaitEvents_ips = np.zeros((n_gait_cycles, 3),dtype=np.int)
        # Contralateral gait events: toe-off, heel strike.
        gaitEvents_cont = np.zeros((n_gait_cycles, 2),dtype=np.int)
        if n_gait_cycles <1:
            raise Exception('Not enough gait cycles found.')

        for i in range(n_gait_cycles):
            # Ipsilateral HS, TO, HS.
            gaitEvents_ips[i,0] = hsIps[-i-2]
            gaitEvents_ips[i,2] = hsIps[-i-1]
            
            # Iterate in reverse through ipsilateral TO, finding the one that
            # is within the range of gaitEvents_ips.
            toIpsFound = False
            for j in range(len(toIps)):
                if toIps[-j-1] > gaitEvents_ips[i,0] and toIps[-j-1] < gaitEvents_ips[i,2] and not toIpsFound:
                    gaitEvents_ips[i,1] = toIps[-j-1]
                    toIpsFound = True

            # Contralateral TO, HS.
            # Iterate in reverse through contralateral HS and TO, finding the
            # one that is within the range of gaitEvents_ips
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
            
            # Skip this step if no contralateral peaks fell within ipsilateral events
            # This can happen with noisy data with subject far from camera. 
            if not toContFound or not hsContFound:                   
                print('Could not find contralateral gait event within ' + 
                               'ipsilateral gait event range ' + str(i+1) + 
                               ' steps until the end. Skipping this step.')
                gaitEvents_cont[i,:] = -1
                gaitEvents_ips[i,:] = -1
        
        # Remove any nan rows
        mask_ips = (gaitEvents_ips == -1).any(axis=1)
        if all(mask_ips):
            raise Exception('No good steps for ' + leg + ' leg.')
        gaitEvents_ips = gaitEvents_ips[~mask_ips]
        gaitEvents_cont = gaitEvents_cont[~mask_ips]
            
        # Convert gaitEvents to times using self.markerDict['time'].
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
    
