'''
    ---------------------------------------------------------------------------
    OpenCap processing: muscleModelOpenSimAD.py
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

import numpy as np

# %% DeGrooteFregly2016MuscleModel
# This class implements the muscle model described in De Groote et al. (2016).
# https://link.springer.com/article/10.1007%2Fs10439-016-1591-9
class DeGrooteFregly2016MuscleModel:
    
    def __init__(self, mtParameters, activation, mtLength, mtVelocity,
                 normTendonForce, normTendonForceDT, tendonCompliance,
                 tendonShift, specificTension, ignorePassiveFiberForce=False):
        self.mtParameters = mtParameters
        
        self.maximalIsometricForce = mtParameters[0]
        self.optimalFiberLength = mtParameters[1]          
        self.tendonSlackLength = mtParameters[2]
        self.optimalPennationAngle = mtParameters[3]        
        self.maximalFiberVelocity = mtParameters[4]
        
        self.activation = activation
        self.mtLength = mtLength
        self.mtVelocity = mtVelocity
        self.normTendonForce = normTendonForce
        self.normTendonForceDT = normTendonForceDT
        self.tendonCompliance = tendonCompliance
        self.tendonShift = tendonShift
        self.specificTension = specificTension
        self.paramFLa = np.array([0.814483478343008, 1.05503342897057,
                                  0.162384573599574, 0.0633034484654646,
                                  0.433004984392647, 0.716775413397760, 
                                  -0.0299471169706956, 0.200356847296188])
        self.paramFLp = np.array([-0.995172050006169, 53.5981500331442])
        self.paramFV = np.array([-0.318323436899127, -8.14915604347525,
                                 -0.374121508647863, 0.885644059915004])
        
        self.ignorePassiveFiberForce = ignorePassiveFiberForce
    
    def getMuscleVolume(self):
        self.muscleVolume = np.multiply(self.maximalIsometricForce, 
                                        self.optimalFiberLength)
        return self.muscleVolume
        
        
    def getMuscleMass(self):                
        muscleMass = np.divide(np.multiply(self.muscleVolume, 1059.7), 
                               np.multiply(self.specificTension, 1e6))
        
        return muscleMass        
        
    def getTendonForce(self):          
        tendonForce = np.multiply(self.normTendonForce, 
                                  self.maximalIsometricForce)  
        
        return tendonForce
            
    def getTendonLength(self):          
        # Tendon force-length relationship.
        self.normTendonLength = np.divide(
                np.log(5*(self.normTendonForce + 0.25 - self.tendonShift)), 
                self.tendonCompliance) + 0.995                                     
        self.tendonLength = np.multiply(self.tendonSlackLength, 
                                        self.normTendonLength)
        
        return self.tendonLength, self.normTendonLength
                
    def getFiberLength(self):
        # Hill-type muscle model: geometric relationships.
        self.getTendonLength()
        w = np.multiply(self.optimalFiberLength, 
                        np.sin(self.optimalPennationAngle))        
        self.fiberLength = np.sqrt(
                (self.mtLength - self.tendonLength)**2 + w**2)
        self.normFiberLength = np.divide(self.fiberLength, 
                                         self.optimalFiberLength)   

        return self.fiberLength, self.normFiberLength         
    
    def getFiberVelocity(self):            
        # Hill-type muscle model: geometric relationships.
        self.getFiberLength()
        tendonVelocity = np.divide(np.multiply(self.tendonSlackLength, 
                                               self.normTendonForceDT), 
            0.2 * self.tendonCompliance * np.exp(self.tendonCompliance * 
                                                 (self.normTendonLength - 
                                                  0.995)))        
        self.cosPennationAngle = np.divide((self.mtLength - self.tendonLength), 
                                           self.fiberLength)        
        self.fiberVelocity = np.multiply((self.mtVelocity - tendonVelocity), 
                                         self.cosPennationAngle)        
        self.normFiberVelocity = np.divide(self.fiberVelocity, 
                                           self.maximalFiberVelocity)  
        
        return self.fiberVelocity, self.normFiberVelocity 
    
    def getActiveFiberLengthForce(self):  
        self.getFiberLength()        
        # Active muscle force-length relationship.
        b11 = self.paramFLa[0]
        b21 = self.paramFLa[1]
        b31 = self.paramFLa[2]
        b41 = self.paramFLa[3]
        b12 = self.paramFLa[4]
        b22 = self.paramFLa[5]
        b32 = self.paramFLa[6]
        b42 = self.paramFLa[7]
        b13 = 0.1
        b23 = 1
        b33 = 0.5 * np.sqrt(0.5)
        b43 = 0
        num3 = self.normFiberLength - b23
        den3 = b33 + b43 * self.normFiberLength
        FMtilde3 = b13 * np.exp(-0.5 * (np.divide(num3**2, den3**2)))
        num1 = self.normFiberLength - b21
        den1 = b31 + b41 * self.normFiberLength        
        FMtilde1 = b11 * np.exp(-0.5 * (np.divide(num1**2, den1**2)))
        num2 = self.normFiberLength - b22
        den2 = b32 + b42 * self.normFiberLength
        FMtilde2 = b12 * np.exp(-0.5 * (np.divide(num2**2, den2**2)))
        self.normActiveFiberLengthForce = FMtilde1 + FMtilde2 + FMtilde3
        
        return self.normActiveFiberLengthForce
        
    def getActiveFiberVelocityForce(self):   
        self.getFiberVelocity()        
        # Active muscle force-velocity relationship.
        e1 = self.paramFV[0]
        e2 = self.paramFV[1]
        e3 = self.paramFV[2]
        e4 = self.paramFV[3]
        
        self.normActiveFiberVelocityForce = e1 * np.log(
                (e2 * self.normFiberVelocity + e3) 
                + np.sqrt((e2 * self.normFiberVelocity + e3)**2 + 1)) + e4
        
    def getActiveFiberForce(self):
        d = 0.01
        self.getActiveFiberLengthForce()
        self.getActiveFiberVelocityForce()
        
        self.normActiveFiberForce = ((self.activation * 
                                      self.normActiveFiberLengthForce * 
                                      self.normActiveFiberVelocityForce) + 
            d * self.normFiberVelocity)
            
        activeFiberForce = (self.normActiveFiberForce * 
                            self.maximalIsometricForce)
        
        activeFiberForcePen = np.multiply(activeFiberForce,
                                          self.cosPennationAngle)
            
        return (activeFiberForce, self.normActiveFiberForce,
                activeFiberForcePen)
        
    def getPassiveFiberForce(self):
        
        if self.ignorePassiveFiberForce:            
            passiveFiberForce = 0
            self.normPassiveFiberForce = 0
            passiveFiberForcePen = 0
            
        else:        
            paramFLp = self.paramFLp
            self.getFiberLength()
            
            # Passive muscle force-length relationship.
            e0 = 0.6
            kpe = 4        
            t5 = np.exp(kpe * (self.normFiberLength - 1) / e0)
            self.normPassiveFiberForce = np.divide(((t5 - 1) - paramFLp[0]), 
                                                   paramFLp[1])
            
            passiveFiberForce = (self.normPassiveFiberForce * 
                                 self.maximalIsometricForce)
            
            passiveFiberForcePen = np.multiply(passiveFiberForce,
                                               self.cosPennationAngle)
            
        return (passiveFiberForce, self.normPassiveFiberForce, 
                passiveFiberForcePen)
        
    def deriveHillEquilibrium(self):        
        self.getActiveFiberForce()
        self.getPassiveFiberForce()
        
        hillEquilibrium = ((np.multiply(self.normActiveFiberForce + 
                                        self.normPassiveFiberForce, 
                                        self.cosPennationAngle)) - 
                                        self.normTendonForce)
        
        return hillEquilibrium