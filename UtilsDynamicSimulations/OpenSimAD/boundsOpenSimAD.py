import scipy.interpolate as interpolate
from scipy import signal
import pandas as pd
import numpy as np

class bounds_tracking:
    
    def __init__(self, Qs, joints, muscles):
        
        self.Qs = Qs
        self.joints = joints
        # self.targetSpeed = targetSpeed
        self.muscles = muscles
        # self.armJoints = armJoints
        # self.mtpJoints = mtpJoints
        
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
    
    def getBoundsPosition(self):
        self.splineQs()
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
            upperBoundsPosition.insert(count, joint, [ub])
            lowerBoundsPosition.insert(count, joint, [lb]) 
            # Special cases
            if joint == 'mtp_angle_l' or joint == 'mtp_angle_r':
                upperBoundsPosition[joint] = [0.52359878000000004]
                lowerBoundsPosition[joint] = [-0.7853981633974483]
                
            # Scaling                       
            s = pd.concat([abs(upperBoundsPosition[joint]), 
                           abs(lowerBoundsPosition[joint])]).max(level=0)
            scalingPosition.insert(count, joint, s)
            lowerBoundsPosition[joint] /= scalingPosition[joint]
            upperBoundsPosition[joint] /= scalingPosition[joint]
                
        return upperBoundsPosition, lowerBoundsPosition, scalingPosition
    
    
    # For the mocap data, the bounds should be extended. Ideally, we should do
    # that too for the video data but simulations were already generated data
    # to track were within the video-bounds.
    def getBoundsPosition_fixed(self, data_type='Video'):
        self.splineQs()
        
        upperBoundsPosition_all = {}
        lowerBoundsPosition_all = {}
       
        if data_type == 'Mocap':            
            upperBoundsPosition_all['hip_flexion_l'] = [120 * np.pi / 180]
            upperBoundsPosition_all['hip_flexion_r'] = [120 * np.pi / 180]
            upperBoundsPosition_all['hip_adduction_l'] = [20 * np.pi / 180]
            upperBoundsPosition_all['hip_adduction_r'] = [20 * np.pi / 180]
            upperBoundsPosition_all['hip_rotation_l'] = [30 * np.pi / 180]
            upperBoundsPosition_all['hip_rotation_r'] = [30 * np.pi / 180]  
            upperBoundsPosition_all['knee_angle_l'] = [125 * np.pi / 180]
            upperBoundsPosition_all['knee_angle_r'] = [125 * np.pi / 180]
            upperBoundsPosition_all['knee_adduction_l'] = [20 * np.pi / 180]
            upperBoundsPosition_all['knee_adduction_r'] = [20 * np.pi / 180]
            upperBoundsPosition_all['ankle_angle_l'] = [50 * np.pi / 180]
            upperBoundsPosition_all['ankle_angle_r'] = [50 * np.pi / 180]
            upperBoundsPosition_all['subtalar_angle_l'] = [35 * np.pi / 180] 
            upperBoundsPosition_all['subtalar_angle_r'] = [35 * np.pi / 180] 
            upperBoundsPosition_all['mtp_angle_l'] = [5 * np.pi / 180]
            upperBoundsPosition_all['mtp_angle_r'] = [5 * np.pi / 180]        
          
            lowerBoundsPosition_all['hip_flexion_l'] = [-30 * np.pi / 180]
            lowerBoundsPosition_all['hip_flexion_r'] = [-30 * np.pi / 180]
            lowerBoundsPosition_all['hip_adduction_l'] = [-40 * np.pi / 180]
            lowerBoundsPosition_all['hip_adduction_r'] = [-40 * np.pi / 180]
            lowerBoundsPosition_all['hip_rotation_l'] = [-40 * np.pi / 180]
            lowerBoundsPosition_all['hip_rotation_r'] = [-40 * np.pi / 180]       
            lowerBoundsPosition_all['knee_angle_l'] = [0 * np.pi / 180]
            lowerBoundsPosition_all['knee_angle_r'] = [0 * np.pi / 180]        
            lowerBoundsPosition_all['knee_adduction_l'] = [-30 * np.pi / 180]
            lowerBoundsPosition_all['knee_adduction_r'] = [-30 * np.pi / 180]        
            lowerBoundsPosition_all['ankle_angle_l'] = [-50 * np.pi / 180]
            lowerBoundsPosition_all['ankle_angle_r'] = [-50 * np.pi / 180]        
            lowerBoundsPosition_all['subtalar_angle_l'] = [-35 * np.pi / 180] 
            lowerBoundsPosition_all['subtalar_angle_r'] = [-35 * np.pi / 180]        
            lowerBoundsPosition_all['mtp_angle_l'] = [-45 * np.pi / 180]
            lowerBoundsPosition_all['mtp_angle_r'] = [-45 * np.pi / 180]   
            
        elif data_type == 'Video':       
            upperBoundsPosition_all['hip_flexion_l'] = [120 * np.pi / 180]
            upperBoundsPosition_all['hip_flexion_r'] = [120 * np.pi / 180]
            upperBoundsPosition_all['hip_adduction_l'] = [20 * np.pi / 180]
            upperBoundsPosition_all['hip_adduction_r'] = [20 * np.pi / 180]
            upperBoundsPosition_all['hip_rotation_l'] = [25 * np.pi / 180]
            upperBoundsPosition_all['hip_rotation_r'] = [25 * np.pi / 180]  
            upperBoundsPosition_all['knee_angle_l'] = [120 * np.pi / 180]
            upperBoundsPosition_all['knee_angle_r'] = [120 * np.pi / 180]
            upperBoundsPosition_all['knee_adduction_l'] = [20 * np.pi / 180]
            upperBoundsPosition_all['knee_adduction_r'] = [20 * np.pi / 180]
            upperBoundsPosition_all['ankle_angle_l'] = [50 * np.pi / 180]
            upperBoundsPosition_all['ankle_angle_r'] = [50 * np.pi / 180]
            upperBoundsPosition_all['subtalar_angle_l'] = [20 * np.pi / 180] 
            upperBoundsPosition_all['subtalar_angle_r'] = [20 * np.pi / 180] 
            upperBoundsPosition_all['mtp_angle_l'] = [5 * np.pi / 180]
            upperBoundsPosition_all['mtp_angle_r'] = [5 * np.pi / 180]        
          
            lowerBoundsPosition_all['hip_flexion_l'] = [-30 * np.pi / 180]
            lowerBoundsPosition_all['hip_flexion_r'] = [-30 * np.pi / 180]
            lowerBoundsPosition_all['hip_adduction_l'] = [-35 * np.pi / 180]
            lowerBoundsPosition_all['hip_adduction_r'] = [-35 * np.pi / 180]
            lowerBoundsPosition_all['hip_rotation_l'] = [-30 * np.pi / 180]
            lowerBoundsPosition_all['hip_rotation_r'] = [-30 * np.pi / 180]       
            lowerBoundsPosition_all['knee_angle_l'] = [0 * np.pi / 180]
            lowerBoundsPosition_all['knee_angle_r'] = [0 * np.pi / 180]        
            lowerBoundsPosition_all['knee_adduction_l'] = [-30 * np.pi / 180]
            lowerBoundsPosition_all['knee_adduction_r'] = [-30 * np.pi / 180]        
            lowerBoundsPosition_all['ankle_angle_l'] = [-40 * np.pi / 180]
            lowerBoundsPosition_all['ankle_angle_r'] = [-40 * np.pi / 180]        
            lowerBoundsPosition_all['subtalar_angle_l'] = [-35 * np.pi / 180] 
            lowerBoundsPosition_all['subtalar_angle_r'] = [-35 * np.pi / 180]        
            lowerBoundsPosition_all['mtp_angle_l'] = [-45 * np.pi / 180]
            lowerBoundsPosition_all['mtp_angle_r'] = [-45 * np.pi / 180]        
        
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
            # Special cases
            if joint == 'mtp_angle_l' or joint == 'mtp_angle_r':
                upperBoundsPosition[joint] = [5 * np.pi / 180]
                lowerBoundsPosition[joint] = [-45 * np.pi / 180]
                
            # Scaling                       
            s = pd.concat([abs(upperBoundsPosition[joint]), 
                           abs(lowerBoundsPosition[joint])]).max(level=0)
            scalingPosition.insert(count, joint, s)
            lowerBoundsPosition[joint] /= scalingPosition[joint]
            upperBoundsPosition[joint] /= scalingPosition[joint]
                
        return upperBoundsPosition, lowerBoundsPosition, scalingPosition
    
    # For the mocap data, the bounds should be extended. Ideally, we should do
    # that too for the video data but simulations were already generated data
    # to track were within the video-bounds.
    def getBoundsPosition_fixed_update1(self, data_type='Video'):
        self.splineQs()
        
        upperBoundsPosition_all = {}
        lowerBoundsPosition_all = {}
       
        if data_type == 'Mocap': 
            raise ValueError("Not supported")
            # upperBoundsPosition_all['hip_flexion_l'] = [120 * np.pi / 180]
            # upperBoundsPosition_all['hip_flexion_r'] = [120 * np.pi / 180]
            # upperBoundsPosition_all['hip_adduction_l'] = [20 * np.pi / 180]
            # upperBoundsPosition_all['hip_adduction_r'] = [20 * np.pi / 180]
            # upperBoundsPosition_all['hip_rotation_l'] = [30 * np.pi / 180]
            # upperBoundsPosition_all['hip_rotation_r'] = [30 * np.pi / 180]  
            # upperBoundsPosition_all['knee_angle_l'] = [125 * np.pi / 180]
            # upperBoundsPosition_all['knee_angle_r'] = [125 * np.pi / 180]
            # upperBoundsPosition_all['knee_adduction_l'] = [20 * np.pi / 180]
            # upperBoundsPosition_all['knee_adduction_r'] = [20 * np.pi / 180]
            # upperBoundsPosition_all['ankle_angle_l'] = [50 * np.pi / 180]
            # upperBoundsPosition_all['ankle_angle_r'] = [50 * np.pi / 180]
            # upperBoundsPosition_all['subtalar_angle_l'] = [35 * np.pi / 180] 
            # upperBoundsPosition_all['subtalar_angle_r'] = [35 * np.pi / 180] 
            # upperBoundsPosition_all['mtp_angle_l'] = [5 * np.pi / 180]
            # upperBoundsPosition_all['mtp_angle_r'] = [5 * np.pi / 180]        
          
            # lowerBoundsPosition_all['hip_flexion_l'] = [-30 * np.pi / 180]
            # lowerBoundsPosition_all['hip_flexion_r'] = [-30 * np.pi / 180]
            # lowerBoundsPosition_all['hip_adduction_l'] = [-40 * np.pi / 180]
            # lowerBoundsPosition_all['hip_adduction_r'] = [-40 * np.pi / 180]
            # lowerBoundsPosition_all['hip_rotation_l'] = [-40 * np.pi / 180]
            # lowerBoundsPosition_all['hip_rotation_r'] = [-40 * np.pi / 180]       
            # lowerBoundsPosition_all['knee_angle_l'] = [0 * np.pi / 180]
            # lowerBoundsPosition_all['knee_angle_r'] = [0 * np.pi / 180]        
            # lowerBoundsPosition_all['knee_adduction_l'] = [-30 * np.pi / 180]
            # lowerBoundsPosition_all['knee_adduction_r'] = [-30 * np.pi / 180]        
            # lowerBoundsPosition_all['ankle_angle_l'] = [-50 * np.pi / 180]
            # lowerBoundsPosition_all['ankle_angle_r'] = [-50 * np.pi / 180]        
            # lowerBoundsPosition_all['subtalar_angle_l'] = [-35 * np.pi / 180] 
            # lowerBoundsPosition_all['subtalar_angle_r'] = [-35 * np.pi / 180]        
            # lowerBoundsPosition_all['mtp_angle_l'] = [-45 * np.pi / 180]
            # lowerBoundsPosition_all['mtp_angle_r'] = [-45 * np.pi / 180]   
            
        elif data_type == 'Video':       
            upperBoundsPosition_all['hip_flexion_l'] = [120 * np.pi / 180]
            upperBoundsPosition_all['hip_flexion_r'] = [120 * np.pi / 180]
            upperBoundsPosition_all['hip_adduction_l'] = [20 * np.pi / 180]
            upperBoundsPosition_all['hip_adduction_r'] = [20 * np.pi / 180]
            upperBoundsPosition_all['hip_rotation_l'] = [30 * np.pi / 180]
            upperBoundsPosition_all['hip_rotation_r'] = [30 * np.pi / 180]  
            upperBoundsPosition_all['knee_angle_l'] = [125 * np.pi / 180]
            upperBoundsPosition_all['knee_angle_r'] = [125 * np.pi / 180]
            upperBoundsPosition_all['knee_adduction_l'] = [20 * np.pi / 180]
            upperBoundsPosition_all['knee_adduction_r'] = [20 * np.pi / 180]
            upperBoundsPosition_all['ankle_angle_l'] = [50 * np.pi / 180]
            upperBoundsPosition_all['ankle_angle_r'] = [50 * np.pi / 180]
            upperBoundsPosition_all['subtalar_angle_l'] = [35 * np.pi / 180] 
            upperBoundsPosition_all['subtalar_angle_r'] = [35 * np.pi / 180] 
            upperBoundsPosition_all['mtp_angle_l'] = [5 * np.pi / 180]
            upperBoundsPosition_all['mtp_angle_r'] = [5 * np.pi / 180]        
          
            lowerBoundsPosition_all['hip_flexion_l'] = [-30 * np.pi / 180]
            lowerBoundsPosition_all['hip_flexion_r'] = [-30 * np.pi / 180]
            lowerBoundsPosition_all['hip_adduction_l'] = [-40 * np.pi / 180]
            lowerBoundsPosition_all['hip_adduction_r'] = [-40 * np.pi / 180]
            lowerBoundsPosition_all['hip_rotation_l'] = [-40 * np.pi / 180]
            lowerBoundsPosition_all['hip_rotation_r'] = [-40 * np.pi / 180]       
            lowerBoundsPosition_all['knee_angle_l'] = [0 * np.pi / 180]
            lowerBoundsPosition_all['knee_angle_r'] = [0 * np.pi / 180]        
            lowerBoundsPosition_all['knee_adduction_l'] = [-30 * np.pi / 180]
            lowerBoundsPosition_all['knee_adduction_r'] = [-30 * np.pi / 180]        
            lowerBoundsPosition_all['ankle_angle_l'] = [-50 * np.pi / 180]
            lowerBoundsPosition_all['ankle_angle_r'] = [-50 * np.pi / 180]        
            lowerBoundsPosition_all['subtalar_angle_l'] = [-35 * np.pi / 180] 
            lowerBoundsPosition_all['subtalar_angle_r'] = [-35 * np.pi / 180]        
            lowerBoundsPosition_all['mtp_angle_l'] = [-45 * np.pi / 180]
            lowerBoundsPosition_all['mtp_angle_r'] = [-45 * np.pi / 180]        
        
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
            # Special cases
            if joint == 'mtp_angle_l' or joint == 'mtp_angle_r':
                upperBoundsPosition[joint] = [5 * np.pi / 180]
                lowerBoundsPosition[joint] = [-45 * np.pi / 180]
                
            # Scaling                       
            s = pd.concat([abs(upperBoundsPosition[joint]), 
                           abs(lowerBoundsPosition[joint])]).max(level=0)
            scalingPosition.insert(count, joint, s)
            lowerBoundsPosition[joint] /= scalingPosition[joint]
            upperBoundsPosition[joint] /= scalingPosition[joint]
                
        return upperBoundsPosition, lowerBoundsPosition, scalingPosition
    
    def getBoundsPosition_fixed_update2(self, data_type='Video'):
        self.splineQs()
        
        upperBoundsPosition_all = {}
        lowerBoundsPosition_all = {}
       
        if data_type == 'Mocap':    
            raise ValueError("Not supported")
            # upperBoundsPosition_all['hip_flexion_l'] = [120 * np.pi / 180]
            # upperBoundsPosition_all['hip_flexion_r'] = [120 * np.pi / 180]
            # upperBoundsPosition_all['hip_adduction_l'] = [20 * np.pi / 180]
            # upperBoundsPosition_all['hip_adduction_r'] = [20 * np.pi / 180]
            # upperBoundsPosition_all['hip_rotation_l'] = [35 * np.pi / 180]
            # upperBoundsPosition_all['hip_rotation_r'] = [35 * np.pi / 180]  
            # upperBoundsPosition_all['knee_angle_l'] = [138 * np.pi / 180]
            # upperBoundsPosition_all['knee_angle_r'] = [138 * np.pi / 180]
            # upperBoundsPosition_all['knee_adduction_l'] = [20 * np.pi / 180]
            # upperBoundsPosition_all['knee_adduction_r'] = [20 * np.pi / 180]
            # upperBoundsPosition_all['ankle_angle_l'] = [50 * np.pi / 180]
            # upperBoundsPosition_all['ankle_angle_r'] = [50 * np.pi / 180]
            # upperBoundsPosition_all['subtalar_angle_l'] = [35 * np.pi / 180] 
            # upperBoundsPosition_all['subtalar_angle_r'] = [35 * np.pi / 180] 
            # upperBoundsPosition_all['mtp_angle_l'] = [5 * np.pi / 180]
            # upperBoundsPosition_all['mtp_angle_r'] = [5 * np.pi / 180]        
          
            # lowerBoundsPosition_all['hip_flexion_l'] = [-30 * np.pi / 180]
            # lowerBoundsPosition_all['hip_flexion_r'] = [-30 * np.pi / 180]
            # lowerBoundsPosition_all['hip_adduction_l'] = [-50 * np.pi / 180]
            # lowerBoundsPosition_all['hip_adduction_r'] = [-50 * np.pi / 180]
            # lowerBoundsPosition_all['hip_rotation_l'] = [-40 * np.pi / 180]
            # lowerBoundsPosition_all['hip_rotation_r'] = [-40 * np.pi / 180]       
            # lowerBoundsPosition_all['knee_angle_l'] = [0 * np.pi / 180]
            # lowerBoundsPosition_all['knee_angle_r'] = [0 * np.pi / 180]        
            # lowerBoundsPosition_all['knee_adduction_l'] = [-30 * np.pi / 180]
            # lowerBoundsPosition_all['knee_adduction_r'] = [-30 * np.pi / 180]        
            # lowerBoundsPosition_all['ankle_angle_l'] = [-50 * np.pi / 180]
            # lowerBoundsPosition_all['ankle_angle_r'] = [-50 * np.pi / 180]        
            # lowerBoundsPosition_all['subtalar_angle_l'] = [-35 * np.pi / 180] 
            # lowerBoundsPosition_all['subtalar_angle_r'] = [-35 * np.pi / 180]        
            # lowerBoundsPosition_all['mtp_angle_l'] = [-45 * np.pi / 180]
            # lowerBoundsPosition_all['mtp_angle_r'] = [-45 * np.pi / 180]   
            
        elif data_type == 'Video':       
            upperBoundsPosition_all['hip_flexion_l'] = [120 * np.pi / 180]
            upperBoundsPosition_all['hip_flexion_r'] = [120 * np.pi / 180]
            upperBoundsPosition_all['hip_adduction_l'] = [20 * np.pi / 180]
            upperBoundsPosition_all['hip_adduction_r'] = [20 * np.pi / 180]
            upperBoundsPosition_all['hip_rotation_l'] = [35 * np.pi / 180]
            upperBoundsPosition_all['hip_rotation_r'] = [35 * np.pi / 180]  
            upperBoundsPosition_all['knee_angle_l'] = [138 * np.pi / 180]
            upperBoundsPosition_all['knee_angle_r'] = [138 * np.pi / 180]
            upperBoundsPosition_all['knee_adduction_l'] = [20 * np.pi / 180]
            upperBoundsPosition_all['knee_adduction_r'] = [20 * np.pi / 180]
            upperBoundsPosition_all['ankle_angle_l'] = [50 * np.pi / 180]
            upperBoundsPosition_all['ankle_angle_r'] = [50 * np.pi / 180]
            upperBoundsPosition_all['subtalar_angle_l'] = [35 * np.pi / 180] 
            upperBoundsPosition_all['subtalar_angle_r'] = [35 * np.pi / 180] 
            upperBoundsPosition_all['mtp_angle_l'] = [5 * np.pi / 180]
            upperBoundsPosition_all['mtp_angle_r'] = [5 * np.pi / 180]        
          
            lowerBoundsPosition_all['hip_flexion_l'] = [-30 * np.pi / 180]
            lowerBoundsPosition_all['hip_flexion_r'] = [-30 * np.pi / 180]
            lowerBoundsPosition_all['hip_adduction_l'] = [-50 * np.pi / 180]
            lowerBoundsPosition_all['hip_adduction_r'] = [-50 * np.pi / 180]
            lowerBoundsPosition_all['hip_rotation_l'] = [-40 * np.pi / 180]
            lowerBoundsPosition_all['hip_rotation_r'] = [-40 * np.pi / 180]       
            lowerBoundsPosition_all['knee_angle_l'] = [0 * np.pi / 180]
            lowerBoundsPosition_all['knee_angle_r'] = [0 * np.pi / 180]        
            lowerBoundsPosition_all['knee_adduction_l'] = [-30 * np.pi / 180]
            lowerBoundsPosition_all['knee_adduction_r'] = [-30 * np.pi / 180]        
            lowerBoundsPosition_all['ankle_angle_l'] = [-50 * np.pi / 180]
            lowerBoundsPosition_all['ankle_angle_r'] = [-50 * np.pi / 180]        
            lowerBoundsPosition_all['subtalar_angle_l'] = [-35 * np.pi / 180] 
            lowerBoundsPosition_all['subtalar_angle_r'] = [-35 * np.pi / 180]        
            lowerBoundsPosition_all['mtp_angle_l'] = [-45 * np.pi / 180]
            lowerBoundsPosition_all['mtp_angle_r'] = [-45 * np.pi / 180]        
        
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
            # Special cases
            if joint == 'mtp_angle_l' or joint == 'mtp_angle_r':
                upperBoundsPosition[joint] = [5 * np.pi / 180]
                lowerBoundsPosition[joint] = [-45 * np.pi / 180]
                
            # Scaling                       
            s = pd.concat([abs(upperBoundsPosition[joint]), 
                           abs(lowerBoundsPosition[joint])]).max(level=0)
            scalingPosition.insert(count, joint, s)
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
            # Special cases
            if joint == 'mtp_angle_l' or joint == 'mtp_angle_r':
                upperBoundsVelocity[joint] = [50]
                lowerBoundsVelocity[joint] = [-50]

            # Scaling                       
            s = pd.concat([abs(upperBoundsVelocity[joint]), 
                           abs(lowerBoundsVelocity[joint])]).max(level=0)
            scalingVelocity.insert(count, joint, s)
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
            # Special cases
            if joint == 'mtp_angle_l' or joint == 'mtp_angle_r':
                upperBoundsAcceleration[joint] = [1000]
                lowerBoundsAcceleration[joint] = [-1000]
        
            # Scaling                       
            s = pd.concat([abs(upperBoundsAcceleration[joint]), 
                           abs(lowerBoundsAcceleration[joint])]).max(level=0)
            scalingAcceleration.insert(count, joint, s)
            upperBoundsAcceleration[joint] /= scalingAcceleration[joint]
            lowerBoundsAcceleration[joint] /= scalingAcceleration[joint]

        return (upperBoundsAcceleration, lowerBoundsAcceleration, 
                scalingAcceleration)
    
    def getBoundsAccelerationFiltered(self):
        self.splineQs()
        upperBoundsAcceleration = pd.DataFrame()   
        lowerBoundsAcceleration = pd.DataFrame() 
        scalingAcceleration = pd.DataFrame() 
        for count, joint in enumerate(self.joints):
            if (self.joints.count(joint[:-1] + 'l')) == 1:        
                ub = max(max(self.Qdotdots_spline_filter[joint[:-1] + 'l']), 
                          max(self.Qdotdots_spline_filter[joint[:-1] + 'r']))
                lb = min(min(self.Qdotdots_spline_filter[joint[:-1] + 'l']), 
                          min(self.Qdotdots_spline_filter[joint[:-1] + 'r']))                              
            else:
                ub = max(self.Qdotdots_spline_filter[joint])
                lb = min(self.Qdotdots_spline_filter[joint])
            r = abs(ub - lb)
            ub = ub + r
            lb = lb - r                        
            upperBoundsAcceleration.insert(count, joint, [ub])
            lowerBoundsAcceleration.insert(count, joint, [lb])
            # Special cases
            if joint == 'mtp_l' or joint == 'mtp_r':
                upperBoundsAcceleration[joint] = [500]
                lowerBoundsAcceleration[joint] = [-500]
        
            # Scaling                       
            s = pd.concat([abs(upperBoundsAcceleration[joint]), 
                           abs(lowerBoundsAcceleration[joint])]).max(level=0)
            scalingAcceleration.insert(count, joint, s)
            upperBoundsAcceleration[joint] /= scalingAcceleration[joint]
            lowerBoundsAcceleration[joint] /= scalingAcceleration[joint]

        return (upperBoundsAcceleration, lowerBoundsAcceleration, 
                scalingAcceleration)
    
    def getBoundsActivation(self, lb_activation=0.05):
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

            # Scaling                       
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

            # Scaling                       
            scalingForce.insert(count + len(self.muscles), 
                                          muscle[:-1] + 'l', s)   
            upperBoundsForce[
                    muscle[:-1] + 'l'] /= scalingForce[muscle[:-1] + 'l']
            lowerBoundsForce[
                    muscle[:-1] + 'l'] /= scalingForce[muscle[:-1] + 'l']
        
        return upperBoundsForce, lowerBoundsForce, scalingForce
    
    def getBoundsActivationDerivative(self):
        activationTimeConstant = 0.015
        deactivationTimeConstant = 0.06
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

            # Scaling                       
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
            
            # Scaling                       
            scalingForceDerivative.insert(count + len(self.muscles), 
                                                muscle[:-1] + 'l', s)  
            upperBoundsForceDerivative[muscle[:-1] + 'l'] /= (
                    scalingForceDerivative[muscle[:-1] + 'l'])
            lowerBoundsForceDerivative[muscle[:-1] + 'l'] /= (
                    scalingForceDerivative[muscle[:-1] + 'l']) 
        
        return (upperBoundsForceDerivative, lowerBoundsForceDerivative, 
                scalingForceDerivative)
    
    def getBoundsArmExcitation(self, armJoints):
        lb = [-1] 
        lb_vec = lb * len(armJoints)
        ub = [1]
        ub_vec = ub * len(armJoints)
        s = [150]
        s_vec = s * len(armJoints)
        upperBoundsArmExcitation = pd.DataFrame([ub_vec], 
                                                columns=armJoints)   
        lowerBoundsArmExcitation = pd.DataFrame([lb_vec], 
                                                columns=armJoints)            
        scalingArmExcitation = pd.DataFrame([s_vec], columns=armJoints)
        
        return (upperBoundsArmExcitation, lowerBoundsArmExcitation,
                scalingArmExcitation)
    
    def getBoundsArmActivation(self, armJoints):
        lb = [-1] 
        lb_vec = lb * len(armJoints)
        ub = [1]
        ub_vec = ub * len(armJoints)
        s = [150]
        s_vec = s * len(armJoints)
        upperBoundsArmActivation = pd.DataFrame([ub_vec], 
                                                columns=armJoints)   
        lowerBoundsArmActivation = pd.DataFrame([lb_vec], 
                                                columns=armJoints) 
        scalingArmActivation = pd.DataFrame([s_vec], columns=armJoints)                  
        
        return (upperBoundsArmActivation, lowerBoundsArmActivation, 
                scalingArmActivation)
        
    def getBoundsMtpExcitation(self, mtpJoints):
        lb = [-1] 
        lb_vec = lb * len(mtpJoints)
        ub = [1]
        ub_vec = ub * len(mtpJoints)
        s = [100]
        s_vec = s * len(mtpJoints)
        upperBoundsMtpExcitation = pd.DataFrame([ub_vec], 
                                                columns=mtpJoints)   
        lowerBoundsMtpExcitation = pd.DataFrame([lb_vec], 
                                                columns=mtpJoints)            
        scalingMtpExcitation = pd.DataFrame([s_vec], columns=mtpJoints)
        
        return (upperBoundsMtpExcitation, lowerBoundsMtpExcitation,
                scalingMtpExcitation)
    
    def getBoundsMtpActivation(self, mtpJoints):
        lb = [-1] 
        lb_vec = lb * len(mtpJoints)
        ub = [1]
        ub_vec = ub * len(mtpJoints)
        s = [100]
        s_vec = s * len(mtpJoints)
        upperBoundsMtpActivation = pd.DataFrame([ub_vec], 
                                                columns=mtpJoints)   
        lowerBoundsMtpActivation = pd.DataFrame([lb_vec], 
                                                columns=mtpJoints) 
        scalingMtpActivation = pd.DataFrame([s_vec], columns=mtpJoints)                  
        
        return (upperBoundsMtpActivation, lowerBoundsMtpActivation, 
                scalingMtpActivation)
    
    def getBoundsLumbarExcitation(self, lumbarJoints):
        lb = [-1] 
        lb_vec = lb * len(lumbarJoints)
        ub = [1]
        ub_vec = ub * len(lumbarJoints)
        s = [300]
        s_vec = s * len(lumbarJoints)
        upperBoundsLumbarExcitation = pd.DataFrame([ub_vec], 
                                                columns=lumbarJoints)   
        lowerBoundsLumbarExcitation = pd.DataFrame([lb_vec], 
                                                columns=lumbarJoints)            
        scalingLumbarExcitation = pd.DataFrame([s_vec], columns=lumbarJoints)
        
        return (upperBoundsLumbarExcitation, lowerBoundsLumbarExcitation,
                scalingLumbarExcitation)
    
    def getBoundsLumbarActivation(self, lumbarJoints):
        lb = [-1] 
        lb_vec = lb * len(lumbarJoints)
        ub = [1]
        ub_vec = ub * len(lumbarJoints)
        s = [300]
        s_vec = s * len(lumbarJoints)
        upperBoundsLumbarActivation = pd.DataFrame([ub_vec], 
                                                columns=lumbarJoints)   
        lowerBoundsLumbarActivation = pd.DataFrame([lb_vec], 
                                                columns=lumbarJoints) 
        scalingLumbarActivation = pd.DataFrame([s_vec], columns=lumbarJoints)                  
        
        return (upperBoundsLumbarActivation, lowerBoundsLumbarActivation, 
                scalingLumbarActivation)
    
    def getBoundsReserveActuators(self, joint, value):
        
        # max_value = {}
        # max_value['hip_flexion_l'] = 20
        # max_value['hip_flexion_r'] = 20
        # max_value['hip_adduction_l'] = 20
        # max_value['hip_adduction_r'] = 20
        # max_value['hip_rotation_l'] = 20
        # max_value['hip_rotation_r'] = 20
        
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
        upperBoundsOffset = pd.DataFrame([0.5 / scaling], columns=['offset_y']) 
        lowerBoundsOffset = pd.DataFrame([-0.5 / scaling], columns=['offset_y'])
        
        return upperBoundsOffset, lowerBoundsOffset
    
    def getBoundsContactParameters(self, NContactSpheres, 
                                   parameter_to_optimize):
        
        
        # lbRadius = 0.022        
        # ubRadius = 0.042
        
        lbRadius = 0.01        
        ubRadius = 0.04
        
        if NContactSpheres == 6:
            
            lbLocation_s1 = np.array([-0.01, -0.015]) - 0.005
            lbLocation_s2 = np.array([0.14,  -0.035]) - 0.005
            lbLocation_s3 = np.array([0.12,   0]) - 0.005
            lbLocation_s4 = np.array([0.04,  -0.03]) - 0.005
            lbLocation_s5 = np.array([0.0,   -0.03]) - 0.005
            lbLocation_s6 = np.array([0.0,   -0.02]) - 0.005
            # lbLocation_s5 = np.array([0.0,   -0.03])
            # lbLocation_s6 = np.array([0.05,   -0.02])
            
            ubLocation_s1 = np.array([0.05,  0.01]) + 0.005
            ubLocation_s2 = np.array([0.185, 0.01]) + 0.005
            ubLocation_s3 = np.array([0.165, 0.05]) + 0.005
            ubLocation_s4 = np.array([0.12,  0.03]) + 0.005
            ubLocation_s5 = np.array([0.07,  0.03]) + 0.005
            ubLocation_s6 = np.array([0.052, 0.05]) + 0.005
            # ubLocation_s5 = np.array([0.04,  0.03])
            # ubLocation_s6 = np.array([0.07, 0.02])
            
            # lbLocation_s1 = np.array([0.00215773306688716053,   -0.00434269152195360195]) - 0.03
            # lbLocation_s2 = np.array([0.16841223157345971972,   -0.03258850869005603529]) - 0.03
            # lbLocation_s3 = np.array([0.15095065283989317351,    0.05860493716970469752]) - 0.03
            # lbLocation_s4 = np.array([0.07517351958454182581,    0.02992219727974926649]) - 0.03
            # lbLocation_s5 = np.array([0.06809743951165971032,   -0.02129214951175864221]) - 0.03
            # lbLocation_s6 = np.array([0.05107307963374478621,    0.07020500618327656095]) - 0.03
            
            # ubLocation_s1 = np.array([0.00215773306688716053,   -0.00434269152195360195]) + 0.03
            # ubLocation_s2 = np.array([0.16841223157345971972,   -0.03258850869005603529]) + 0.03
            # ubLocation_s3 = np.array([0.15095065283989317351,    0.05860493716970469752]) + 0.03
            # ubLocation_s4 = np.array([0.07517351958454182581,    0.02992219727974926649]) + 0.03
            # ubLocation_s5 = np.array([0.06809743951165971032,   -0.02129214951175864221]) + 0.03
            # ubLocation_s6 = np.array([0.05107307963374478621,    0.07020500618327656095]) + 0.03
        
            if parameter_to_optimize == 'option1':  
        
                lbContactParameters_unsc = np.concatenate((lbLocation_s1, lbLocation_s2, lbLocation_s3, lbLocation_s4, lbLocation_s5, lbLocation_s6, [lbRadius], [lbRadius], [lbRadius], [lbRadius], [lbRadius], [lbRadius]))
                ubContactParameters_unsc = np.concatenate((ubLocation_s1, ubLocation_s2, ubLocation_s3, ubLocation_s4, ubLocation_s5, ubLocation_s6, [ubRadius], [ubRadius], [ubRadius], [ubRadius], [ubRadius], [ubRadius]))
        
        scalingContactParameters_v = 1 / (ubContactParameters_unsc - lbContactParameters_unsc)
        scalingContactParameters_r = 0.5 - ubContactParameters_unsc / (ubContactParameters_unsc - lbContactParameters_unsc)
        
        lowerBoundsContactParameters = -0.5 * np.ones((1, len(lbContactParameters_unsc)))
        upperBoundsContactParameters = 0.5 * np.ones((1, len(ubContactParameters_unsc)))
        
        return upperBoundsContactParameters, lowerBoundsContactParameters, scalingContactParameters_v, scalingContactParameters_r
    
    def getBoundsGR(self, GR, headers):
        upperBoundsGR = pd.DataFrame()   
        lowerBoundsGR = pd.DataFrame() 
        scalingGR = pd.DataFrame() 
        for count, header in enumerate(headers):             
            if (header[0] == 'R' or header[0] == 'L'):        
                ub = max(max(GR['R' + header[1:]]), 
                         max(GR['L' + header[1:]]))
                lb = min(min(GR['R' + header[1:]]), 
                         min(GR['L' + header[1:]]))                              
            else:
                raise ValueError("Problem bounds GR")
            r = abs(ub - lb)
            ub = ub + r
            lb = lb - r                        
            upperBoundsGR.insert(count, header, [ub])
            lowerBoundsGR.insert(count, header, [lb]) 
                
            # Scaling                       
            s = pd.concat([abs(upperBoundsGR[header]), 
                           abs(lowerBoundsGR[header])]).max(level=0)
            scalingGR.insert(count, header, s)
            lowerBoundsGR[header] /= scalingGR[header]
            upperBoundsGR[header] /= scalingGR[header]
                
        return upperBoundsGR, lowerBoundsGR, scalingGR
    
    # def getBoundsFinalTime(self):
    #     upperBoundsFinalTime = pd.DataFrame([1], columns=['time'])   
    #     lowerBoundsFinalTime = pd.DataFrame([0.1], columns=['time'])  
        
    #     return upperBoundsFinalTime, lowerBoundsFinalTime