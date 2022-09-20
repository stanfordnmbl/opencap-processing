import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
from scipy import signal
   
# %% Quasi-random initial guess    
class quasiRandomGuess:    
    def __init__(self, N, d, joints, muscles, time, Qs):      
        
        self.N = N
        self.d = d
        self.joints = joints
        self.guessFinalTime = time
        self.muscles = muscles
        self.targetSpeed = 1.2
        self.Qs = Qs
    
    # Mesh points
    def getGuessPosition(self, scaling):
        g = [0] * (self.N + 1)
        
        pelvis_tx_first = self.Qs["pelvis_tx"][0]
        g_pelvis_tx = np.linspace(
            pelvis_tx_first, 
            pelvis_tx_first + self.guessFinalTime * self.targetSpeed, 
            self.N)
        g_pelvis_tx = np.append(g_pelvis_tx, g_pelvis_tx[-1] + 
                                (g_pelvis_tx[-1] - g_pelvis_tx[-2]))
        g_pelvis_ty =  [0.9385] * (self.N + 1)
        self.guessPosition = pd.DataFrame()  
        for count, joint in enumerate(self.joints): 
            if joint == 'pelvis_tx':
                self.guessPosition.insert(count, joint, 
                                          g_pelvis_tx / scaling.iloc[0][joint])
            elif joint == 'pelvis_ty':
                self.guessPosition.insert(count, joint, 
                                          g_pelvis_ty / scaling.iloc[0][joint])                    
            else:
                self.guessPosition.insert(count, joint, 
                                          g / scaling.iloc[0][joint])
        
        return self.guessPosition
    
    def getGuessVelocity(self, scaling):
        g = [0] * (self.N + 1)
        g_pelvis_tx =  [self.targetSpeed] * (self.N + 1)
        self.guessVelocity = pd.DataFrame()  
        for count, joint in enumerate(self.joints): 
            if joint == 'pelvis_tx':
                self.guessVelocity.insert(count, joint,
                                          g_pelvis_tx / scaling.iloc[0][joint])             
            else:
                self.guessVelocity.insert(count, joint, 
                                          g / scaling.iloc[0][joint])
        
        return self.guessVelocity
    
    # TODO: zeroAcceleration to match data-driven - not great
    def getGuessAcceleration(self, scaling, zeroAcceleration=True):
        if zeroAcceleration:
            g = [0] * self.N
        else:
            raise ValueError('Guess acceleration - zero')
        self.guessAcceleration = pd.DataFrame()  
        for count, joint in enumerate(self.joints):
            self.guessAcceleration.insert(count, joint, 
                                          g / scaling.iloc[0][joint])
            
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
    
    def getGuessMarker(self, markers, marker_data, scaling, 
                       dimensions = ["x", "y", "z"]):
        guessMarker = pd.DataFrame() 
        count = 0
        for marker in markers:  
            for dimension in dimensions:
                guessMarker.insert(count, marker + "_" + dimension, 
                                   marker_data[marker + "_" + dimension] / 
                                   scaling.iloc[0][marker + "_" + dimension]) 
                count += 1
        
        return guessMarker
    
    def getGuessOffset(self, scaling):
        
        guessOffset = 1.4 / scaling
        
        return guessOffset
    
class contactParameterGuess:
    
    def getGuessContactParameters(self, NContactSpheres, 
                                  parameter_to_optimize, 
                                  scaling_v, scaling_r):
        
        if NContactSpheres == 6:
        
            location_s1 = np.array([0.00215773306688716053,   -0.00434269152195360195])
            location_s2 = np.array([0.16841223157345971972,   -0.03258850869005603529])
            location_s3 = np.array([0.15095065283989317351,    0.05860493716970469752])
            location_s4 = np.array([0.07517351958454182581,    0.02992219727974926649])
            location_s5 = np.array([0.06809743951165971032,   -0.02129214951175864221])
            location_s6 = np.array([0.05107307963374478621,    0.07020500618327656095])
            
            radius_s1 = 0.032
            radius_s2 = 0.032
            radius_s3 = 0.032
            radius_s4 = 0.032
            radius_s5 = 0.032
            radius_s6 = 0.032
            
            if parameter_to_optimize == 'option1' :           
                contactParameters_unsc = np.concatenate((location_s1, location_s2, location_s3, location_s4, location_s5, location_s6, [radius_s1], [radius_s2], [radius_s3], [radius_s4], [radius_s5], [radius_s6]))
        
        guessContactParameters = contactParameters_unsc * scaling_v + scaling_r
                
        return guessContactParameters    
    
    # def getGuessContactParameters_6s_option2(self, scaling_v, scaling_r):
        
    #     location_s1 = np.array([-0.000452297523548588,   -0.0053620734121204309])
    #     location_s2 = np.array([0.064380934268635601,    0.021461384438449679])
    #     location_s3 = np.array([0.17704756923874793,     0.0227296888435418])
    #     location_s4 = np.array([0.17704756923874793,     -0.010730155711439269]) 
    #     location_s5 = np.array([0.057035069668584286,    -0.0036668161112701409])   
    #     location_s6 = np.array([1.8650083642052589e-06,  0.023921809143082704])
        
    #     radius_s1 = 0.032320
    #     radius_s2 = 0.032320
    #     radius_s3 = 0.023374
    #     radius_s4 = 0.020508
    #     radius_s5 = 0.016244
    #     radius_s6 = 0.018414
        
    #     stiffness = 1e6        
    #     dissipation = 2
        
    #     staticFriction =  0.8
    #     dynamicFriction = 0.8       
    #     viscousFriction = 0.5       
    #     transitionVelocity = 0.2
        
    #     contactParameters_unsc = np.concatenate(
    #         (location_s1, location_s2, location_s3, location_s4, location_s5, location_s6,
    #          [radius_s1], [radius_s2], [radius_s3], [radius_s4], [radius_s5], [radius_s6],
    #          [stiffness], [dissipation], [staticFriction], [dynamicFriction], [viscousFriction], [transitionVelocity]))
    #     guessContactParameters = contactParameters_unsc * scaling_v + scaling_r
                
    #     return guessContactParameters
    
# %% Data-driven initial guess    
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
            spline = interpolate.InterpolatedUnivariateSpline(self.Qs['time'], 
                                                              self.Qs[joint],
                                                              k=3)
            self.Qs_spline[joint] = spline(self.Qs['time'])
            splineD1 = spline.derivative(n=1)
            self.Qdots_spline[joint] = splineD1(self.Qs['time'])
            splineD2 = spline.derivative(n=2)
            self.Qdotdots_spline[joint] = splineD2(self.Qs['time'])
            
        fs=1/np.mean(np.diff(self.Qdotdots_spline['time']))    
        fc = 10  # Cut-off frequency of the filter
        order = 4
        w = fc / (fs / 2) # Normalize the frequency
        b, a = signal.butter(order/2, w, 'low')  
        output = signal.filtfilt(b, a, self.Qdotdots_spline.loc[:, self.Qdotdots_spline.columns != 'time'], axis=0, 
                                 padtype='odd', padlen=3*(max(len(b),len(a))-1))    
        output = pd.DataFrame(data=output, columns=self.joints)
        self.Qdotdots_spline_filter = pd.concat([pd.DataFrame(data=self.Qdotdots_spline['time'], columns=['time']), 
                            output], axis=1) 
            
    # def interpQs(self):
    #     self.splineQs()            
    #     tOut = np.linspace(self.Qs['time'].iloc[0], 
    #                        self.Qs['time'].iloc[-1], 
    #                        self.N + 1)    
        
    #     self.Qs_spline_interp = pd.DataFrame()  
    #     self.Qdots_spline_interp = pd.DataFrame()  
    #     self.Qdotdots_spline_interp = pd.DataFrame()  
    #     for count, joint in enumerate(self.joints):  
    #         set_interp = interp1d(self.Qs['time'], self.Qs_spline[joint])
    #         self.Qs_spline_interp.insert(count, joint, set_interp(tOut))
            
    #         set_interp = interp1d(self.Qs['time'], self.Qdots_spline[joint])
    #         self.Qdots_spline_interp.insert(count, joint, set_interp(tOut))
            
    #         set_interp = interp1d(self.Qs['time'], self.Qdotdots_spline[joint])
    #         self.Qdotdots_spline_interp.insert(count, joint, set_interp(tOut))
        
    
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
    
    def getGuessAccelerationFiltered(self, scaling, zeroAcceleration=False):
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
                        count, joint, self.Qdotdots_spline_filter[joint] /
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
    
    def getGuessMarker(self, markers, marker_data, scaling, 
                       dimensions = ["x", "y", "z"]):
        guessMarker = pd.DataFrame() 
        count = 0
        for marker in markers:  
            for dimension in dimensions:
                guessMarker.insert(count, marker + "_" + dimension, 
                                   marker_data[marker + "_" + dimension] / 
                                   scaling.iloc[0][marker + "_" + dimension]) 
                count += 1
        
        return guessMarker
    
    def getGuessOffset(self, scaling):
        
        guessOffset = 0.2 / scaling
        
        return guessOffset
    
    def getGuessGRFCol(self, GRFs):
        g = [0] * (self.N * self.d)
        guessGRFCol = pd.DataFrame()  
        for count, GRF in enumerate(GRFs):
            guessGRFCol.insert(count, GRF, g)
            
        return guessGRFCol