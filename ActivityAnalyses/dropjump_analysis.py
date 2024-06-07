"""
    ---------------------------------------------------------------------------
    OpenCap processing: dropjump_analysis.py
    ---------------------------------------------------------------------------

    Copyright 2024 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Carmichael Ong
    
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
from scipy.signal import find_peaks, peak_widths
from matplotlib import pyplot as plt
import scipy.interpolate as interpolate

from utilsKinematics import kinematics
import opensim

class dropjump_analysis(kinematics):
    
    def __init__(self, session_dir, trial_name,
                 lowpass_cutoff_frequency_for_coordinate_values=-1,
                 trimming_start=0, trimming_end=0):
        
        # Inherit init from kinematics class.
        super().__init__(
            session_dir, 
            trial_name, 
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values)
        
        # We might want to trim the start/end of the trial to remove bad data.
        self.trimming_start = trimming_start
        self.trimming_end = trimming_end
                        
        # Marker data load and filter.
        self.markerDict = self.get_marker_dict(
            session_dir, 
            trial_name, 
            lowpass_cutoff_frequency = lowpass_cutoff_frequency_for_coordinate_values)

        # Coordinate values.
        self.coordinateValues = self.get_coordinate_values()
        
        # Body transforms
        # self.bodyTransformDict = self.get_body_transform_dict()

        # Making sure time vectors of marker and coordinate data are the same.
        if not np.allclose(self.markerDict['time'], self.coordinateValues['time'], atol=.001, rtol=0):
            raise Exception('Time vectors of marker and coordinate data are not the same.')
        
        # if not np.allclose(self.bodyTransformDict['time'], self.coordinateValues['time'], atol=.001, rtol=0):
        #     raise Exception('Time vectors of body transofrms and coordinate data are not the same.')
            
        # Trim marker, body transforms, and coordinate data.
        if self.trimming_start > 0:
            self.idx_trim_start = np.where(np.round(self.markerDict['time'] - self.trimming_start,6) <= 0)[0][-1]
            self.markerDict['time'] = self.markerDict['time'][self.idx_trim_start:]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][self.idx_trim_start:,:]
            # self.bodyTransformDict['time'] = self.bodyTransformDict['time'][self.idx_trim_start:]
            # for body in self.bodyTransformDict['body_transforms']:
            #     self.bodyTransformDict['body_transforms'][body] = \
            #         self.bodyTransformDict['body_transforms'][body][self.idx_trim_start:]
            self.coordinateValues = self.coordinateValues.iloc[self.idx_trim_start:]
        
        if self.trimming_end > 0:
            self.idx_trim_end = np.where(np.round(self.markerDict['time'],6) <= np.round(self.markerDict['time'][-1] - self.trimming_end,6))[0][-1] + 1
            self.markerDict['time'] = self.markerDict['time'][:self.idx_trim_end]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][:self.idx_trim_end,:]
            # self.bodyTransformDict['time'] = self.bodyTransformDict['time'][self.idx_trim_end:]
            # for body in self.bodyTransformDict['body_transforms']:
            #     self.bodyTransformDict['body_transforms'][body] = \
            #         self.bodyTransformDict['body_transforms'][body][self.idx_trim_end:]
            self.coordinateValues = self.coordinateValues.iloc[:self.idx_trim_end]
        
        # Segment dropjump.
        self.dropjumpEvents = self.segment_dropjump(visualizeSegmentation=True)
        # self.nRepetitions = np.shape(self.dropjumpEvents['eventIdxs'])[0]
        
        # Initialize variables to be lazy loaded.
        self._comValues = None
        
        # Time
        self.time = self.coordinateValues['time'].to_numpy()
    
    # Compute COM trajectory.
    def comValues(self):
        if self._comValues is None:
            self._comValues = self.get_center_of_mass_values()
            if self.trimming_start > 0:
                self._comValues = self._comValues.iloc[self.idx_trim_start:]            
            if self.trimming_end > 0:
                self._comValues = self._comValues.iloc[:self.idx_trim_end]
        return self._comValues
    
    def get_dropjump_events(self):
        
        return self.dropjumpEvents
    
    def compute_scalars(self, scalarNames):
               
        # Verify that scalarNames are methods in dropjump_analysis.
        method_names = [func for func in dir(self) if callable(getattr(self, func))]
        possibleMethods = [entry for entry in method_names if 'compute_' in entry]
        
        if scalarNames is None:
            print('No scalars defined, these methods are available:')
            print(*possibleMethods)
            return
        
        nonexistant_methods = [entry for entry in scalarNames if 'compute_' + entry not in method_names]
        
        if len(nonexistant_methods) > 0:
            raise Exception(str(['compute_' + a for a in nonexistant_methods]) + ' does not exist in dropjump_analysis class.')
        
        scalarDict = {}
        for scalarName in scalarNames:
            thisFunction = getattr(self, 'compute_' + scalarName)
            scalarDict[scalarName] = {}
            (scalarDict[scalarName]['value'],
                scalarDict[scalarName]['units']) = thisFunction()
        
        return scalarDict
        
    
    def segment_dropjump(self, prominence=1, distance_sec=0.4, height=1,
                      visualizeSegmentation=False):
        
        
        test_markers = ['r_toe_study', 'L_toe_study']
        time = self.markerDict['time']
        fs = np.round(1/np.mean(np.diff(time)),6)
        distance = distance_sec*fs
        
        legs = ['r', 'l']
        contactIdxs, contactNames, contactTimes = {}, {}, {}
        toeoffIdxs, toeoffNames, toeoffTimes = {}, {}, {}
        for count, test_marker in enumerate(test_markers):
            # Get reference marker position and velocity.
            m_y = self.markerDict['markers']['{}'.format(test_marker)][:,1]            
            spline = interpolate.InterpolatedUnivariateSpline(time, m_y, k=3)
            splineD1 = spline.derivative(n=1)
            m_y_vel = splineD1(time)            
            
            
            contactIdxs[legs[count]], _ = find_peaks(-m_y_vel, prominence=prominence, distance=distance, height=height)
            # TODO
            if len(contactIdxs[legs[count]]) > 2:
                raise ValueError("too many contact detected")
            contactNames[legs[count]] = ['firstContact','secondContact']
            contactTimes[legs[count]] = time[contactIdxs[legs[count]]]
            
            toeoffIdxs[legs[count]], _ = find_peaks(m_y_vel, prominence=prominence, distance=distance, height=height)
            # Exclude values not between contactIdxs[legs[count]]
            toeoffIdxs[legs[count]] = toeoffIdxs[legs[count]][(toeoffIdxs[legs[count]] > contactIdxs[legs[count]][0]) &
                                                                (toeoffIdxs[legs[count]] < contactIdxs[legs[count]][1])]
            # TODO
            if len(toeoffIdxs[legs[count]]) > 1:
                raise ValueError("too many toe off detected")
            toeoffNames[legs[count]] = ['toeoff']
            toeoffTimes[legs[count]] = time[toeoffIdxs[legs[count]]]

            if visualizeSegmentation:
                plt.figure()
                plt.plot(time, m_y, label='{} pos'.format(test_marker))
                plt.plot(time, m_y_vel, label='{} vel'.format(test_marker))
                
                plt.scatter(time[contactIdxs[legs[count]]], m_y_vel[contactIdxs[legs[count]]], color='green')
                plt.vlines(time[contactIdxs[legs[count]]], np.min(m_y_vel), np.max(m_y_vel), color='green', linewidth=1)
                
                plt.scatter(time[toeoffIdxs[legs[count]]], m_y_vel[toeoffIdxs[legs[count]]], color='red')
                plt.vlines(time[toeoffIdxs[legs[count]]], np.min(m_y_vel), np.max(m_y_vel), color='red', linewidth=1)
                
                plt.legend()
                plt.show()

        # Find if double-leg or single-leg by looking for height differences
        # during largest possible contact window
        largest_contact_window_idx = [min(contactIdxs['r'][0], contactIdxs['l'][0]),
                                       max(toeoffIdxs['r'][0], toeoffIdxs['l'][0])]
        
        m_right_y = self.markerDict['markers']['r_toe_study'][:,1]
        m_left_y = self.markerDict['markers']['L_toe_study'][:,1]
        m_diff = m_right_y - m_left_y
        
        time_window = time[largest_contact_window_idx[0]:largest_contact_window_idx[1]]
        m_diff_window = (m_right_y - m_left_y)[largest_contact_window_idx[0]:largest_contact_window_idx[1]]
        
        if visualizeSegmentation:
            plt.figure()
            plt.plot(time_window, m_diff_window)
            plt.title('smallest contact window: toe marker right - left')
        
        eventType = 'double'
        if np.percentile(abs(m_diff_window), 95) > 0.1:
            if np.mean(m_diff_window) < 0.0:
                eventType = 'right'
            else:
                eventType = 'left'
        
        print(f'eventType = {eventType}')
        

        # TODO, let's just select the widest window for now
        contactIdxsAll = [min(contactIdxs['r'][0], contactIdxs['l'][0]), max(contactIdxs['r'][1], contactIdxs['l'][1])]
        toeoffIdxsAll = [max(toeoffIdxs['r'][0], toeoffIdxs['l'][0])]
        contactNamesAll = ['firstContact','secondContact']
        toeoffNamesAll = ['toeoff']
        contactTimesAll = time[contactIdxsAll]
        toeoffTimesAll = time[toeoffIdxsAll]
        

        eventIdxs = {'contactIdxs': contactIdxsAll, 'toeoffIdxs': toeoffIdxsAll}
        eventTimes = {'contactTimes': contactTimesAll, 'toeoffTimes': toeoffTimesAll}
        eventNames = {'contactNames': contactNamesAll, 'toeoffNames': toeoffNamesAll}        
        dropjumpEvents = {
            'eventIdxs': eventIdxs,
            'eventTimes': eventTimes,
            'eventNames': eventNames,
            'eventType': eventType}
        
        return dropjumpEvents
    
    def compute_jump_height(self):
        # Get the pelvis COM trajectory.
        pelvis_ty = self.coordinateValues['pelvis_ty'].to_numpy()  

        # Select from first contact to second contact
        selIdxs = [self.dropjumpEvents['eventIdxs']['contactIdxs'][0], self.dropjumpEvents['eventIdxs']['contactIdxs'][1]]
        
        # Compute the jump height.
        # TODO: is that a proper definition of jump height
        max_pelvis_height = np.max(pelvis_ty[selIdxs[0]:selIdxs[1]+1])
        min_pelvis_height = np.min(pelvis_ty[selIdxs[0]:selIdxs[1]+1])
        jump_height = max_pelvis_height - min_pelvis_height
        
        # Define units.
        units = 'm'
        
        return jump_height, units
    
    def compute_leg_length(self):
        # Estimate leg length based on pelvis and toes COM vertical distance 
        # in default model pose.
        model = self.model
        default_state = model.initSystem()
        body_set = model.getBodySet()
        pelvis_body = body_set.get('pelvis')
        toes_r_body = body_set.get('toes_r')
        toes_l_body = body_set.get('toes_l')
        
        pelvis_y = pelvis_body.expressVectorInGround(default_state,
                                                     pelvis_body.getMassCenter()).get(1)
        toes_r_y = toes_r_body.expressVectorInGround(default_state,
                                                toes_r_body.getMassCenter()).get(1)
        toes_l_y = toes_l_body.expressVectorInGround(default_state,
                                                toes_l_body.getMassCenter()).get(1)
        
        leg_length = pelvis_y - 0.5*(toes_r_y + toes_l_y)
        
        # Define units.
        units = 'm'
        
        return leg_length, units
    
    def compute_contact_time(self):        

        # Select from first contact to toe off
        contact_time = self.dropjumpEvents['eventTimes']['contactTimes'][1] - self.dropjumpEvents['eventTimes']['toeoffTimes'][0]
        
        # Define units.
        units = 's'
        
        return contact_time, units
    
    def compute_peak_knee_flexion_angle(self):

        # Select from first contact to toe off contact
        selIdxs = [self.dropjumpEvents['eventIdxs']['contactIdxs'][0], self.dropjumpEvents['eventIdxs']['toeoffIdxs'][0]]

        # Compute peak angle
        peak_angle_r = np.max(self.coordinateValues['knee_angle_r'].to_numpy()[selIdxs[0]:selIdxs[1]+1])
        peak_angle_l = np.max(self.coordinateValues['knee_angle_l'].to_numpy()[selIdxs[0]:selIdxs[1]+1])
        
        # TODO, is this sound?
        peak_angle = np.max([peak_angle_r, peak_angle_l])
        
        # Define units.
        units = 'deg'
        
        return peak_angle, units
    
    def compute_peak_hip_flexion_angle(self):

        # Select from first contact to toe off contact
        selIdxs = [self.dropjumpEvents['eventIdxs']['contactIdxs'][0], self.dropjumpEvents['eventIdxs']['toeoffIdxs'][0]]

        # Compute peak angle
        peak_angle_r = np.max(self.coordinateValues['hip_flexion_r'].to_numpy()[selIdxs[0]:selIdxs[1]+1])
        peak_angle_l = np.max(self.coordinateValues['hip_flexion_l'].to_numpy()[selIdxs[0]:selIdxs[1]+1])
        
        # TODO, is this sound?
        peak_angle = np.max([peak_angle_r, peak_angle_l])
        
        # Define units.
        units = 'deg'
        
        return peak_angle, units
    
    def compute_peak_ankle_dorsiflexion_angle(self):

        # Select from first contact to toe off contact
        selIdxs = [self.dropjumpEvents['eventIdxs']['contactIdxs'][0], self.dropjumpEvents['eventIdxs']['toeoffIdxs'][0]]

        # Compute peak angle
        peak_angle_r = np.max(self.coordinateValues['ankle_angle_r'].to_numpy()[selIdxs[0]:selIdxs[1]+1])
        peak_angle_l = np.max(self.coordinateValues['ankle_angle_l'].to_numpy()[selIdxs[0]:selIdxs[1]+1])
        
        # TODO, is this sound?
        peak_angle = np.max([peak_angle_r, peak_angle_l])
        
        
        # Define units.
        units = 'deg'
        
        return peak_angle, units
    
    def compute_peak_trunk_lean(self,return_all=False):
        
        # Select from first contact to toe off contact
        selIdxs = [self.dropjumpEvents['eventIdxs']['contactIdxs'][0], self.dropjumpEvents['eventIdxs']['toeoffIdxs'][0]]

        # Trunk vector
        mid_shoulder =  (self.markerDict['markers']['r_shoulder_study'] + self.markerDict['markers']['L_shoulder_study']) / 2
        mid_hip = (self.markerDict['markers']['RHJC_study'] + self.markerDict['markers']['LHJC_study']) / 2
        trunk_vec = (mid_shoulder - mid_hip)[selIdxs[0]:selIdxs[1]+1, :]

        # Trunk lean
        # TODO, is this sound?
        trunk_lean = np.max(np.degrees(np.arctan2(trunk_vec[:, 0], trunk_vec[:, 1])))
        
        # Define units.
        units = 'deg'
        
        return trunk_lean, units
