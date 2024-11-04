'''
    ---------------------------------------------------------------------------
    OpenCap processing: utilsKineticsOpenSimAD.py
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
'''

import os
import numpy as np
import pandas as pd
import utils
import opensim

class kineticsOpenSimAD:
    
    def __init__(self, data_dir, session_id, trial_name, case=None, 
                 repetition=None, modelName=None):
        """
        Initializes the kineticsOpenSimAD class for extracting data from a
        dynamic simulation.

        Args:
            data_dir (str): The directory where data is stored.
            session_id (str): The identifier for the data collection session.
            trial_name (str): The name of the motion trial.
            repetition (int, optional): The index of the motion repetition, only
            applicable if motion_type is 'sit_to_stand' or 'squats' 
            case (str, optional): The case for which to retrieve results. If no
            case is specified and only one case exists, it is returned.
            modelName (str, optional): The name of the OpenSim model used in the 
            simulation (ignore if default).

        This class is designed to extract data from a dynamic simulation 
        conducted using OpenSimAD.
        """
        
        # Load optimal results.
        sessionDir = os.path.join(data_dir, session_id)
        opensimDir = os.path.join(sessionDir, 'OpenSimData')
        repetitionSuffix = ''
        if not repetition is None:
            repetitionSuffix = '_rep' + str(repetition)
        resultsDir = os.path.join(opensimDir, 'Dynamics', 
                                  trial_name + repetitionSuffix)
        # Check if results directory exists.
        if not os.path.exists(resultsDir):
            raise Exception(
                'Results directory: ' + resultsDir + ' does not exist. \
                      Have you run the simulation for ' + trial_name + ' ? \
                        Have you set the repetition index if simulating \
                            a sit to stand or a squat?')

        optimal_results = np.load(
            os.path.join(resultsDir, 'optimaltrajectories.npy'), 
            allow_pickle=True).item()

        if case is None and len(optimal_results) > 1:
            raise Exception("Multiple cases found. Please specify a case.")
        elif case is None and len(optimal_results) == 1:
            case = list(optimal_results.keys())[0]
        self.optimal_result = optimal_results[case]

        # Load OpenSim model.
        modelBasePath = os.path.join(opensimDir, 'Model')
        # Load model if specified, otherwise load the one that was on server.
        if modelName is None:
            modelName = utils.get_model_name_from_metadata(sessionDir)
            # Use adjusted model with contact spheres as used for simulation.
            modelName = modelName[:-5] + '_adjusted_contacts.osim'
            modelPath = os.path.join(modelBasePath, modelName)
        else:
            modelPath = os.path.join(
                modelBasePath, '{}_adjusted_contacts.osim'.format(modelName))            
        if not os.path.exists(modelPath):
            raise Exception('Model path: ' + modelPath + ' does not exist.')

        self.model = opensim.Model(modelPath)
        self.model.initSystem()
        
        # Coordinates.
        self.coordinateSet = self.model.getCoordinateSet()
        self.nCoordinates = self.coordinateSet.getSize()
        self.coordinates = [self.coordinateSet.get(i).getName() 
                            for i in range(self.nCoordinates)] 
        self.coordinate_names = self.optimal_result['coordinates']
        # Sanity check.
        model_coordinates = [i for i in self.coordinates]
        for coord in self.coordinate_names:
            assert coord in model_coordinates, \
                'Coordinate {} not found in model'.format(coord) 
        # Find rotational and translational coordinates.
        self.idxColumnTrLabels = [
            self.coordinate_names.index(i) for i in self.coordinates if \
            self.coordinateSet.get(i).getMotionType() == 2]
        self.idxColumnRotLabels = [
            self.coordinate_names.index(i) for i in self.coordinates if \
            self.coordinateSet.get(i).getMotionType() == 1]
        
        # Muscles.
        self.muscle_names = self.optimal_result['muscles']

        # Time.
        # We do not include the last time point control values are not specified
        # at the last time point.
        self.time = self.optimal_result['time'][0, :-1].flatten()
    
    def get_coordinate_values(self):
        """
        Retrieves the coordinate values from the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing coordinate values with 
            columns named after the coordinates.
    
        Notes:
            - Rotational degrees of freedom are reported in degrees.
            - Translational degrees of freedom are reported in meters.
        """
        coordinate_values = np.copy(self.optimal_result['coordinate_values'])
        coordinate_values[self.idxColumnRotLabels, :] *= 180/np.pi
    
        output = pd.DataFrame(coordinate_values[:,:-1].T, 
                              columns=self.coordinate_names)
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_tracked_coordinate_values(self):
        """
        Retrieves the coordinate values tracked during the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing coordinate values with 
            columns named after the coordinates.
    
        Notes:
            - Rotational degrees of freedom are reported in degrees.
            - Translational degrees of freedom are reported in meters.
        """
        coordinate_values = np.copy(
            self.optimal_result['coordinate_values_toTrack'])
        coordinate_values[self.idxColumnRotLabels, :] *= 180/np.pi
    
        output = pd.DataFrame(coordinate_values.T, 
                              columns=self.coordinate_names)
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_coordinate_speeds(self):
        """
        Retrieves the coordinate speeds from the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing coordinate speeds with 
            columns named after the coordinates.
    
        Notes:
            - Rotational degrees of freedom are reported in degrees per second.
            - Translational degrees of freedom are reported in meters per
              second.
        """
        coordinate_speeds = np.copy(self.optimal_result['coordinate_speeds'])
        coordinate_speeds[self.idxColumnRotLabels, :] *= 180/np.pi
    
        output = pd.DataFrame(coordinate_speeds[:,:-1].T, 
                              columns=self.coordinate_names)
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_tracked_coordinate_speeds(self):
        """
        Retrieves the coordinate speeds tracked during the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing coordinate speeds with 
            columns named after the coordinates.
    
        Notes:
            - Rotational degrees of freedom are reported in degrees per second.
            - Translational degrees of freedom are reported in meters per
              second.
        """
        coordinate_speeds = np.copy(
            self.optimal_result['coordinate_speeds_toTrack'])
        coordinate_speeds[self.idxColumnRotLabels, :] *= 180/np.pi
    
        output = pd.DataFrame(coordinate_speeds.T, 
                              columns=self.coordinate_names)
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_coordinate_accelerations(self):
        """
        Retrieves the coordinate accelerations from the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing coordinate accelerations 
            with columns named after the coordinates.
    
        Notes:
            - Rotational degrees of freedom are reported in degrees per second
              squared.
            - Translational degrees of freedom are reported in meters per second
              squared.
        """
        coordinate_accelerations = np.copy(
            self.optimal_result['coordinate_accelerations'])
        coordinate_accelerations[self.idxColumnRotLabels, :] *= 180/np.pi
    
        output = pd.DataFrame(coordinate_accelerations.T, 
                              columns=self.coordinate_names)
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_tracked_coordinate_accelerations(self):
        """
        Retrieves the coordinate accelerations tracked during the dynamic
        simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing coordinate accelerations
            with columns named after the coordinates.
    
        Notes:
            - Rotational degrees of freedom are reported in degrees per second
              squared.
            - Translational degrees of freedom are reported in meters per second
              squared.
        """
        coordinate_accelerations = np.copy(
            self.optimal_result['coordinate_accelerations_toTrack'])
        coordinate_accelerations[self.idxColumnRotLabels, :] *= 180/np.pi
    
        output = pd.DataFrame(coordinate_accelerations.T, 
                              columns=self.coordinate_names)
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_ground_reaction_forces(self):
        """
        Retrieves the ground reaction forces from the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing ground reaction forces
            with columns named after the side and direction.
    
        Notes:
            - Forces are reported in Newtons.
            - Side: right and left indicate right and left legs, respectively.
            - Direction: x, y, and z indicate the posterior-anterior, 
              inferior-superior, and medial-lateral directions, respectively.
        """   
        output = pd.DataFrame(self.optimal_result['GRF'].T, 
                              columns=self.optimal_result['GRF_labels'])
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_ground_reaction_moments(self):
        """
        Retrieves the ground reaction moments from the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing ground reaction moments
            with columns named after the side and direction.
    
        Notes:
            - Moments are reported in Newton-meters.
            - The moments are expressed in the ground reference frame.
            - Side: right and left indicate right and left legs, respectively.
            - Direction: x, y, and z indicate the posterior-anterior, 
              inferior-superior, and medial-lateral directions, respectively.
            - Ground reactions can be expressed with ground reaction forces and
              moments, or as ground reaction forces, center-of-pressure, and 
              free moments. These representations should not be mixed.
        """   
        output = pd.DataFrame(self.optimal_result['GRM'].T, 
                              columns=self.optimal_result['GRM_labels'])
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_ground_reaction_free_moments(self):
        """
        Retrieves the ground reaction free moments from the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing ground reaction free 
            moments with columns named after the side and direction.
    
        Notes:
            - Moments are reported in Newton-meters.
            - The moments are expressed in the ground reference frame.
            - Side: right and left indicate right and left legs, respectively.
            - Direction: x, y, and z indicate the posterior-anterior, 
              inferior-superior, and medial-lateral directions, respectively.
            - Ground reactions can be expressed with ground reaction forces and
              moments, or as ground reaction forces, center-of-pressure, and 
              free moments. These representations should not be mixed.
        """   
        output = pd.DataFrame(self.optimal_result['freeM'].T, 
                              columns=self.optimal_result['GRM_labels'])
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_centers_of_pressure(self):
        """
        Retrieves the centers of pressure from the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing center of pressures with
            columns named after the side and direction.
    
        Notes:
            - Centers of pressure are reported in meters.
            - The centers of pressure are expressed in the ground reference
              frame with respect to the ground origin.
            - Side: right and left indicate right and left legs, respectively.
            - Direction: x, y, and z indicate the posterior-anterior, 
              inferior-superior, and medial-lateral directions, respectively.
            - Ground reactions can be expressed with ground reaction forces and
              moments, or as ground reaction forces, center-of-pressure, and 
              free moments. These representations should not be mixed.
        """   
        output = pd.DataFrame(self.optimal_result['COP'].T, 
                              columns=self.optimal_result['COP_labels'])
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_joint_moments(self):
        """
        Retrieves the joint moments from the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing joint moments with columns
            after the coordinates.
    
        Notes:
            - Moments are reported in Newton-meters.
        """
        output = pd.DataFrame(self.optimal_result['torques'].T, 
                              columns=self.coordinate_names)
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_joint_powers(self):
        """
        Retrieves the joint powers from the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing joint powers with columns
            after the coordinates.
    
        Notes:
            - Powers are reported in Watts.
        """
        output = pd.DataFrame(self.optimal_result['powers'].T, 
                              columns=self.optimal_result['coordinates_power'])
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_muscle_activations(self):
        """
        Retrieves the muscle activations from the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing muscle activations with
            columns after the muscles.
    
        Notes:
            - Activations are unitless.
        """
        output = pd.DataFrame(self.optimal_result['muscle_activations'][:,:-1].T, 
                              columns=self.muscle_names)
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_muscle_forces(self):
        """
        Retrieves the muscle forces from the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing muscle forces with
            columns after the muscles.
    
        Notes:
            - Froces are reported in Newtons.
        """
        output = pd.DataFrame(self.optimal_result['muscle_forces'].T, 
                              columns=self.muscle_names)
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_knee_adduction_moments(self):
        """
        Retrieves the knee adduction moments from the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing knee adduction moments.
    
        Notes:
            - Moments are reported in Newton-meters.
        """
        output = pd.DataFrame(self.optimal_result['KAM'].T, 
                              columns=self.optimal_result['KAM_labels'])
        output.insert(0, 'time', self.time)
    
        return output
    
    def get_medial_knee_contact_forces(self):
        """
        Retrieves the medial knee contact forces from the dynamic simulation.
    
        Returns:
            pandas.DataFrame: A DataFrame containing medial knee contact forces.
    
        Notes:
            - Forces are reported in Newtons.
        """
        output = pd.DataFrame(self.optimal_result['MCF'].T, 
                              columns=self.optimal_result['MCF_labels'])
        output.insert(0, 'time', self.time)
    
        return output
        