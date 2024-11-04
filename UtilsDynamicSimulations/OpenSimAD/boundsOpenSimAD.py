'''
    ---------------------------------------------------------------------------
    OpenCap processing: boundsOpenSimAD.py
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

import scipy.interpolate as interpolate
import pandas as pd
import numpy as np

class bounds_tracking:
    
    def __init__(self, Qs, joints, muscles):
        
        self.Qs = Qs
        self.joints = joints
        self.muscles = muscles
        
    def splineQs(self):
        
        self.Qs_spline = self.Qs.copy()
        self.Qdots_spline = self.Qs.copy()
        self.Qdotdots_spline = self.Qs.copy()
        for joint in self.joints:
            spline = interpolate.InterpolatedUnivariateSpline(
                self.Qs['time'], self.Qs[joint], k=3)
            self.Qs_spline[joint] = spline(self.Qs['time'])
            splineD1 = spline.derivative(n=1)
            self.Qdots_spline[joint] = splineD1(self.Qs['time'])
            splineD2 = spline.derivative(n=2)
            self.Qdotdots_spline[joint] = splineD2(self.Qs['time'])
    
    def getBoundsPosition(self, polynomial_bounds):
        
        self.splineQs()        
        upperBoundsPosition_all = {}
        lowerBoundsPosition_all = {}
        for coord in polynomial_bounds:
            upperBoundsPosition_all[coord] = (
                [polynomial_bounds[coord]['max'] * np.pi / 180])
            lowerBoundsPosition_all[coord] = (
                [polynomial_bounds[coord]['min'] * np.pi / 180])
        upperBoundsPosition = pd.DataFrame()   
        lowerBoundsPosition = pd.DataFrame() 
        scalingPosition = pd.DataFrame() 
        for count, joint in enumerate(self.joints):             
            if (self.joints.count(joint[:-1] + 'l')) == 1:        
                ub = max(max(self.Qs_spline[joint[:-1] + 'l']), 
                         max(self.Qs_spline[joint[:-1] + 'r']))
                lb = min(min(self.Qs_spline[joint[:-1] + 'l']), 
                         min(self.Qs_spline[joint[:-1] + 'r']))                              
            else:
                ub = max(self.Qs_spline[joint])
                lb = min(self.Qs_spline[joint])                
            r = abs(ub - lb)
            ub = ub + r
            lb = lb - r            
            if joint in upperBoundsPosition_all: 
                ub = min(ub, upperBoundsPosition_all[joint][0])
                lb = max(lb, lowerBoundsPosition_all[joint][0])            
            upperBoundsPosition.insert(count, joint, [ub])
            lowerBoundsPosition.insert(count, joint, [lb]) 
            # Special cases.
            if joint == 'mtp_angle_l' or joint == 'mtp_angle_r':
                # The kinematic data for the mtp angles is not meaningful, not
                # estimated from the videos. We therefore keep the original
                # values from polynomial bounds.
                if joint in upperBoundsPosition_all:
                    upperBoundsPosition[joint] = [
                        upperBoundsPosition_all[joint][0]]
                    lowerBoundsPosition[joint] = [
                        lowerBoundsPosition_all[joint][0]]
                else:
                    message = 'No bounds set for mtp angles in ' \
                        'polynomial_bounds. Setting to default [-45, 5] deg.'
                    print(message)
                    upperBoundsPosition[joint] = [5 * np.pi / 180]
                    lowerBoundsPosition[joint] = [-45 * np.pi / 180]                    
            # Scaling.               
            s = np.max(np.array([abs(upperBoundsPosition[joint])[0],
                                 abs(lowerBoundsPosition[joint])[0]]))
            scalingPosition.insert(count, joint, [s])
            lowerBoundsPosition[joint] /= scalingPosition[joint]
            upperBoundsPosition[joint] /= scalingPosition[joint]
                
        return upperBoundsPosition, lowerBoundsPosition, scalingPosition
    
    def getBoundsVelocity(self):
        
        self.splineQs()        
        upperBoundsVelocity = pd.DataFrame()   
        lowerBoundsVelocity = pd.DataFrame() 
        scalingVelocity = pd.DataFrame() 
        for count, joint in enumerate(self.joints):     
            if (self.joints.count(joint[:-1] + 'l')) == 1:        
                ub = max(max(self.Qdots_spline[joint[:-1] + 'l']), 
                          max(self.Qdots_spline[joint[:-1] + 'r']))
                lb = min(min(self.Qdots_spline[joint[:-1] + 'l']), 
                          min(self.Qdots_spline[joint[:-1] + 'r']))                              
            else:
                ub = max(self.Qdots_spline[joint])
                lb = min(self.Qdots_spline[joint])
            r = abs(ub - lb)
            ub = ub + r
            lb = lb - r                        
            upperBoundsVelocity.insert(count, joint, [ub])
            lowerBoundsVelocity.insert(count, joint, [lb])
            # Special cases.
            if joint == 'mtp_angle_l' or joint == 'mtp_angle_r':
                upperBoundsVelocity[joint] = [50]
                lowerBoundsVelocity[joint] = [-50]
            # Scaling.             
            s = np.max(np.array([abs(upperBoundsVelocity[joint])[0],
                                 abs(lowerBoundsVelocity[joint])[0]]))
            scalingVelocity.insert(count, joint, [s])
            upperBoundsVelocity[joint] /= scalingVelocity[joint]
            lowerBoundsVelocity[joint] /= scalingVelocity[joint]

        return upperBoundsVelocity, lowerBoundsVelocity, scalingVelocity
    
    def getBoundsAcceleration(self):
        
        self.splineQs()        
        upperBoundsAcceleration = pd.DataFrame()   
        lowerBoundsAcceleration = pd.DataFrame() 
        scalingAcceleration = pd.DataFrame() 
        for count, joint in enumerate(self.joints):
            if (self.joints.count(joint[:-1] + 'l')) == 1:        
                ub = max(max(self.Qdotdots_spline[joint[:-1] + 'l']), 
                          max(self.Qdotdots_spline[joint[:-1] + 'r']))
                lb = min(min(self.Qdotdots_spline[joint[:-1] + 'l']), 
                          min(self.Qdotdots_spline[joint[:-1] + 'r']))                              
            else:
                ub = max(self.Qdotdots_spline[joint])
                lb = min(self.Qdotdots_spline[joint])
            r = abs(ub - lb)
            ub = ub + r
            lb = lb - r                        
            upperBoundsAcceleration.insert(count, joint, [ub])
            lowerBoundsAcceleration.insert(count, joint, [lb])   
            # Special cases.
            if joint == 'mtp_angle_l' or joint == 'mtp_angle_r':
                upperBoundsAcceleration[joint] = [1000]
                lowerBoundsAcceleration[joint] = [-1000]        
            # Scaling.   
            s = np.max(np.array([abs(upperBoundsAcceleration[joint])[0],
                                 abs(lowerBoundsAcceleration[joint])[0]]))
            scalingAcceleration.insert(count, joint, [s])
            upperBoundsAcceleration[joint] /= scalingAcceleration[joint]
            lowerBoundsAcceleration[joint] /= scalingAcceleration[joint]

        return (upperBoundsAcceleration, lowerBoundsAcceleration, 
                scalingAcceleration)
    
    def getBoundsActivation(self, lb_activation=0.01):
        
        lb = [lb_activation] 
        lb_vec = lb * len(self.muscles)
        ub = [1]
        ub_vec = ub * len(self.muscles)
        s = [1]
        s_vec = s * len(self.muscles)
        upperBoundsActivation = pd.DataFrame([ub_vec], columns=self.muscles)   
        lowerBoundsActivation = pd.DataFrame([lb_vec], columns=self.muscles) 
        scalingActivation = pd.DataFrame([s_vec], columns=self.muscles)
        upperBoundsActivation = upperBoundsActivation.div(scalingActivation)
        lowerBoundsActivation = lowerBoundsActivation.div(scalingActivation)
        for count, muscle in enumerate(self.muscles):
            upperBoundsActivation.insert(count + len(self.muscles), 
                                          muscle[:-1] + 'l', ub)
            lowerBoundsActivation.insert(count + len(self.muscles), 
                                          muscle[:-1] + 'l', lb)  
            # Scaling.
            scalingActivation.insert(count + len(self.muscles), 
                                      muscle[:-1] + 'l', s)  
            upperBoundsActivation[
                    muscle[:-1] + 'l'] /= scalingActivation[muscle[:-1] + 'l']
            lowerBoundsActivation[
                    muscle[:-1] + 'l'] /= scalingActivation[muscle[:-1] + 'l']
        
        return upperBoundsActivation, lowerBoundsActivation, scalingActivation
    
    def getBoundsForce(self):
        
        lb = [0] 
        lb_vec = lb * len(self.muscles)
        ub = [5]
        ub_vec = ub * len(self.muscles)
        s = max([abs(lbi) for lbi in lb], [abs(ubi) for ubi in ub])
        s_vec = s * len(self.muscles)
        upperBoundsForce = pd.DataFrame([ub_vec], columns=self.muscles)   
        lowerBoundsForce = pd.DataFrame([lb_vec], columns=self.muscles) 
        scalingForce = pd.DataFrame([s_vec], columns=self.muscles)
        upperBoundsForce = upperBoundsForce.div(scalingForce)
        lowerBoundsForce = lowerBoundsForce.div(scalingForce)
        for count, muscle in enumerate(self.muscles):
            upperBoundsForce.insert(count + len(self.muscles), 
                                    muscle[:-1] + 'l', ub)
            lowerBoundsForce.insert(count + len(self.muscles), 
                                    muscle[:-1] + 'l', lb)
            # Scaling.                   
            scalingForce.insert(count + len(self.muscles), 
                                          muscle[:-1] + 'l', s)   
            upperBoundsForce[
                    muscle[:-1] + 'l'] /= scalingForce[muscle[:-1] + 'l']
            lowerBoundsForce[
                    muscle[:-1] + 'l'] /= scalingForce[muscle[:-1] + 'l']
        
        return upperBoundsForce, lowerBoundsForce, scalingForce
    
    def getBoundsActivationDerivative(self, activationTimeConstant=0.015,
                                      deactivationTimeConstant=0.06):
        
        lb = [-1 / deactivationTimeConstant] 
        lb_vec = lb * len(self.muscles)
        ub = [1 / activationTimeConstant]
        ub_vec = ub * len(self.muscles)
        s = [100]
        s_vec = s * len(self.muscles)
        upperBoundsActivationDerivative = pd.DataFrame([ub_vec], 
                                                        columns=self.muscles)   
        lowerBoundsActivationDerivative = pd.DataFrame([lb_vec], 
                                                        columns=self.muscles) 
        scalingActivationDerivative = pd.DataFrame([s_vec], 
                                                    columns=self.muscles)
        upperBoundsActivationDerivative = upperBoundsActivationDerivative.div(
                scalingActivationDerivative)
        lowerBoundsActivationDerivative = lowerBoundsActivationDerivative.div(
                scalingActivationDerivative)
        for count, muscle in enumerate(self.muscles):
            upperBoundsActivationDerivative.insert(count + len(self.muscles), 
                                                    muscle[:-1] + 'l', ub)
            lowerBoundsActivationDerivative.insert(count + len(self.muscles), 
                                                    muscle[:-1] + 'l', lb)
            # Scaling.
            scalingActivationDerivative.insert(count + len(self.muscles), 
                                                muscle[:-1] + 'l', s)  
            upperBoundsActivationDerivative[muscle[:-1] + 'l'] /= (
                    scalingActivationDerivative[muscle[:-1] + 'l'])
            lowerBoundsActivationDerivative[muscle[:-1] + 'l'] /= (
                    scalingActivationDerivative[muscle[:-1] + 'l'])             
        
        return (upperBoundsActivationDerivative, 
                lowerBoundsActivationDerivative, scalingActivationDerivative)
    
    def getBoundsForceDerivative(self):
        
        lb = [-100] 
        lb_vec = lb * len(self.muscles)
        ub = [100]
        ub_vec = ub * len(self.muscles)
        s = [100]
        s_vec = s * len(self.muscles)
        upperBoundsForceDerivative = pd.DataFrame([ub_vec], 
                                                  columns=self.muscles)   
        lowerBoundsForceDerivative = pd.DataFrame([lb_vec], 
                                                  columns=self.muscles) 
        scalingForceDerivative = pd.DataFrame([s_vec], 
                                              columns=self.muscles)
        upperBoundsForceDerivative = upperBoundsForceDerivative.div(
                scalingForceDerivative)
        lowerBoundsForceDerivative = lowerBoundsForceDerivative.div(
                scalingForceDerivative)
        for count, muscle in enumerate(self.muscles):
            upperBoundsForceDerivative.insert(count + len(self.muscles), 
                                              muscle[:-1] + 'l', ub)
            lowerBoundsForceDerivative.insert(count + len(self.muscles), 
                                              muscle[:-1] + 'l', lb)            
            # Scaling.
            scalingForceDerivative.insert(count + len(self.muscles), 
                                                muscle[:-1] + 'l', s)  
            upperBoundsForceDerivative[muscle[:-1] + 'l'] /= (
                    scalingForceDerivative[muscle[:-1] + 'l'])
            lowerBoundsForceDerivative[muscle[:-1] + 'l'] /= (
                    scalingForceDerivative[muscle[:-1] + 'l']) 
        
        return (upperBoundsForceDerivative, lowerBoundsForceDerivative, 
                scalingForceDerivative)
    
    def getBoundsCoordinateDynamics(self, coordinates, optimal_forces = {}, 
                                    default_optimal_force=500):
        
        lb = [-1] 
        lb_vec = lb * len(coordinates)
        ub = [1]
        ub_vec = ub * len(coordinates)
        s_vec = []
        for coord in coordinates:
            if coord in optimal_forces:
                s_vec.append(optimal_forces[coord])
            else:
                s_vec.append(default_optimal_force)
        upperBoundsCoordinateExcitation = pd.DataFrame([ub_vec], 
                                                columns=coordinates)   
        lowerBoundsCoordinateExcitation = pd.DataFrame([lb_vec], 
                                                columns=coordinates)            
        scalingCoordinateExcitation = pd.DataFrame([s_vec], columns=coordinates)
        
        return (upperBoundsCoordinateExcitation, lowerBoundsCoordinateExcitation,
                scalingCoordinateExcitation)
    
    def getBoundsReserveActuators(self, joint, value):
        
        lb = [-1] 
        lb_vec = lb
        ub = [1]
        ub_vec = ub
        s = [value]
        s_vec = s
        upperBoundsReserveActuator = pd.DataFrame([ub_vec], 
                                                  columns=[joint])   
        lowerBoundsReserveActuator = pd.DataFrame([lb_vec], 
                                                  columns=[joint])            
        scalingReserveActuator = pd.DataFrame([s_vec], columns=[joint])
        
        return (upperBoundsReserveActuator, lowerBoundsReserveActuator,
                scalingReserveActuator)
    
    def getBoundsOffset(self, scaling):
        
        upperBoundsOffset = pd.DataFrame([0.5 / scaling], 
                                         columns=['offset_y']) 
        lowerBoundsOffset = pd.DataFrame([-0.5 / scaling], 
                                         columns=['offset_y'])
        
        return upperBoundsOffset, lowerBoundsOffset