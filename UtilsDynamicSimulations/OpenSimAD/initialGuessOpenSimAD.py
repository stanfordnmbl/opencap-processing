'''
    ---------------------------------------------------------------------------
    OpenCap processing: initialGuessOpenSimAD.py
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

import pandas as pd
import scipy.interpolate as interpolate

# %% Data-driven initial guess.
class dataDrivenGuess_tracking:    
    def __init__(self, Qs, N, d, joints, muscles):        
        
        self.Qs = Qs
        self.N = N
        self.d = d
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
    
    # Mesh points
    def getGuessPosition(self, scaling):
        self.splineQs()
        self.guessPosition = pd.DataFrame()  
        g = [0] * (self.N)
        for count, joint in enumerate(self.joints):  
            if joint == 'mtp_angle_l' or joint == 'mtp_angle_r':
                self.guessPosition.insert(count, joint, g) 
            
            else:
                self.guessPosition.insert(count, joint, 
                                          self.Qs_spline[joint] / 
                                          scaling.iloc[0][joint]) 
        
        return self.guessPosition
    
    def getGuessVelocity(self, scaling):
        self.splineQs()
        self.guessVelocity = pd.DataFrame()  
        g = [0] * (self.N)
        for count, joint in enumerate(self.joints): 
            if joint == 'mtp_angle_l' or joint == 'mtp_angle_r':
                self.guessVelocity.insert(count, joint, g)             
            else:
                self.guessVelocity.insert(count, joint, 
                                          self.Qdots_spline[joint] / 
                                          scaling.iloc[0][joint])       
        return self.guessVelocity
    
    def getGuessAcceleration(self, scaling, zeroAcceleration=False):
        self.splineQs()
        self.guessAcceleration = pd.DataFrame()  
        g = [0] * self.N
        g1 = [0] * (self.N)
        for count, joint in enumerate(self.joints):   
            if zeroAcceleration:
                self.guessAcceleration.insert(
                    count, joint, g / scaling.iloc[0][joint]) 
            else:
                if joint == 'mtp_angle_l' or joint == 'mtp_angle_r':
                    self.guessAcceleration.insert(count, joint, g1) 
                else:                
                    self.guessAcceleration.insert(
                        count, joint, self.Qdotdots_spline[joint] /
                        scaling.iloc[0][joint])                               
                    
        return self.guessAcceleration
    
    def getGuessActivation(self, scaling):
        g = [0.1] * (self.N + 1)
        self.guessActivation = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessActivation.insert(count, muscle, 
                                        g / scaling.iloc[0][muscle])
            
        return self.guessActivation
    
    def getGuessActivationDerivative(self, scaling):
        g = [0.01] * self.N
        guessActivationDerivative = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            guessActivationDerivative.insert(count, muscle, 
                                             g / scaling.iloc[0][muscle])
            
        return guessActivationDerivative
    
    def getGuessForce(self, scaling):
        g = [0.1] * (self.N + 1)
        self.guessForce = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessForce.insert(count, muscle, g / scaling.iloc[0][muscle])
            
        return self.guessForce
    
    def getGuessForceDerivative(self, scaling):
        g = [0.01] * self.N
        self.guessForceDerivative = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessForceDerivative.insert(count, muscle, 
                                        g / scaling.iloc[0][muscle])
            
        return self.guessForceDerivative
    
    def getGuessTorqueActuatorActivation(self, torqueActuatorJoints):
        g = [0.1] * (self.N + 1)
        self.guessTorqueActuatorActivation = pd.DataFrame()  
        for count, torqueActuatorJoint in enumerate(torqueActuatorJoints):
            self.guessTorqueActuatorActivation.insert(
                    count, torqueActuatorJoint, g)
            
        return self.guessTorqueActuatorActivation
    
    def getGuessTorqueActuatorExcitation(self, torqueActuatorJoints):
        g = [0.1] * self.N
        guessTorqueActuatorExcitation = pd.DataFrame()  
        for count, torqueActuatorJoint in enumerate(torqueActuatorJoints):
            guessTorqueActuatorExcitation.insert(count, torqueActuatorJoint, g)
            
        return guessTorqueActuatorExcitation 
    
    def getGuessReserveActuators(self, joint):
        g = [0] * self.N
        guessReserveActuators = pd.DataFrame(g, columns=[joint])  
            
        return guessReserveActuators 
    
    # Collocation points   
    def getGuessActivationCol(self):            
        temp = []
        for k in range(self.N):
            for c in range(self.d):
                temp.append(self.guessActivation.iloc[k])
        guessActivationCol = pd.DataFrame.from_records(temp)
            
        return guessActivationCol
    
    def getGuessForceCol(self):
        temp = []
        for k in range(self.N):
            for c in range(self.d):          
                temp.append(self.guessForce.iloc[k])
        guessForceCol = pd.DataFrame.from_records(temp)
            
        return guessForceCol
    
    def getGuessForceDerivativeCol(self):
        temp = []         
        for k in range(self.N):
            for c in range(self.d):
                temp.append(self.guessForceDerivative.iloc[k])
        guessForceDerivativeCol = pd.DataFrame.from_records(temp)
            
        return guessForceDerivativeCol
    
    def getGuessTorqueActuatorActivationCol(self, torqueActuatorJoints):
        temp = []
        for k in range(self.N):
            for c in range(self.d):
                temp.append(self.guessTorqueActuatorActivation.iloc[k])
        guessTorqueActuatorActivationCol = pd.DataFrame.from_records(temp)
            
        return guessTorqueActuatorActivationCol        
    
    def getGuessPositionCol(self):
        temp = []
        for k in range(self.N):
            for c in range(self.d):
                temp.append(self.guessPosition.iloc[k])
        guessPositionCol = pd.DataFrame.from_records(temp)
        
        return guessPositionCol
    
    def getGuessVelocityCol(self):
        temp = []      
        for k in range(self.N):
            for c in range(self.d):
                temp.append(self.guessVelocity.iloc[k])
        guessVelocityCol = pd.DataFrame.from_records(temp)
        
        return guessVelocityCol
    
    def getGuessAccelerationCol(self):
        temp = []
        guessAccelerationCol = pd.DataFrame(columns=self.joints)  
        for k in range(self.N):
            for c in range(self.d):
                temp.append(self.guessAcceleration.iloc[k])
        guessAccelerationCol = pd.DataFrame.from_records(temp)
                
        return guessAccelerationCol
    
    def getGuessOffset(self, scaling):
        
        guessOffset = 0.2 / scaling
        
        return guessOffset