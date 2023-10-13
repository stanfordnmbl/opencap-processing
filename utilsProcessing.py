'''
    ---------------------------------------------------------------------------
    OpenCap processing: utilsProcessing.py
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
pathFile = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(os.path.join(pathFile, 'ActivityAnalyses'))

import logging
import opensim
import numpy as np
from scipy import signal
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from utils import storage_to_dataframe, download_trial, get_trial_id
from utilsTRC import TRCFile, numpy2TRC

def lowPassFilter(time, data, lowpass_cutoff_frequency, order=4):
    
    fs = 1/np.round(np.mean(np.diff(time)),16)
    wn = lowpass_cutoff_frequency/(fs/2)
    sos = signal.butter(order/2, wn, btype='low', output='sos')
    dataFilt = signal.sosfiltfilt(sos, data, axis=0)

    return dataFilt

# %% Segment gait
def segment_gait(session_id, trial_name, data_folder, gait_cycles_from_end=0):
    
    # Segmentation is done in the gait_analysis class
    from gait_analysis import gait_analysis  
    
    gait = gait_analysis(os.path.join(data_folder,session_id), trial_name,
                         n_gait_cycles=-1)
    heelstrikeTimes = gait.gaitEvents['ipsilateralTime'][gait_cycles_from_end,(0,2)].tolist()
    
    return heelstrikeTimes, gait

# %% Segment squats.
def segment_squats(ikFilePath, pelvis_ty=None, timeVec=None, visualize=False,
                  filter_pelvis_ty=True, cutoff_frequency=4, height=.2):
    
    # TODO: eventually, this belongs in a squat_analysis class and should take
    # the form of segment_gait
    
    # Extract pelvis_ty if not given.
    if pelvis_ty is None and timeVec is None:
        ikResults = storage_to_dataframe(ikFilePath,headers={'pelvis_ty'})
        timeVec = ikResults['time']
        if filter_pelvis_ty:
            from utilsOpenSimAD import filterNumpyArray
            pelvis_ty = filterNumpyArray(
                ikResults['pelvis_ty'].to_numpy(), timeVec.to_numpy(), 
                cutoff_frequency=cutoff_frequency)
        else:
            pelvis_ty = ikResults['pelvis_ty']    
    dt = timeVec[1] - timeVec[0]

    # Identify minimums.
    pelvSignal = np.array(-pelvis_ty - np.min(-pelvis_ty))
    pelvSignalPos = np.array(pelvis_ty - np.min(pelvis_ty))
    idxMinPelvTy,_ = signal.find_peaks(pelvSignal,distance=.7/dt,height=height)
    
    # Find the max adjacent to all of the minimums.
    minIdxOld = 0
    startFinishInds = []
    for i, minIdx in enumerate(idxMinPelvTy):
        if i<len(idxMinPelvTy)-1:
            nextIdx = idxMinPelvTy[i+1]
        else:
            nextIdx = len(pelvSignalPos)
        startIdx = np.argmax(pelvSignalPos[minIdxOld:minIdx]) + minIdxOld
        endIdx = np.argmax(pelvSignalPos[minIdx:nextIdx]) + minIdx
        startFinishInds.append([startIdx,endIdx])
        minIdxOld = np.copy(minIdx)
    startFinishTimes = [timeVec[i].tolist() for i in startFinishInds]
    
    if visualize:
        plt.figure()     
        plt.plot(-pelvSignal)
        for c_v, val in enumerate(startFinishInds):
            plt.plot(val, -pelvSignal[val], marker='o', markerfacecolor='k',
                     markeredgecolor='none', linestyle='none',
                     label='Squatting phase')
            if c_v == 0:
                plt.legend()
        plt.xlabel('Frames')
        plt.ylabel('Position [m]')
        plt.title('Vertical pelvis position')
        plt.draw()
    
    return startFinishTimes

# %% Segment sit-to-stands.
'''
 Three time intervals are returned:
     - risingTimes: rising phase.
     - risingTimesDelayedStart: rising phase from delayed start to exclude
        time interval when there is contact with the chair.
     - risingSittingTimesDelayedStartPeriodicEnd: rising and sitting phases
         from delayed start to corresponding periodic end in terms of
         vertical pelvis position.     
'''
def segment_STS(ikFilePath, pelvis_ty=None, timeVec=None, velSeated=0.3,
               velStanding=0.15, visualize=False, filter_pelvis_ty=True, 
               cutoff_frequency=4, delay=0.1):
    
    # TODO: eventually, this belongs in a sts_analysis class and should take
    # the form of segment_gait
    
    # Extract pelvis_ty if not given.
    if pelvis_ty is None and timeVec is None:
        ikResults = storage_to_dataframe(ikFilePath,headers={'pelvis_ty'})
        timeVec = ikResults['time']
        if filter_pelvis_ty:
            from utilsOpenSimAD import filterNumpyArray
            pelvis_ty = filterNumpyArray(
                ikResults['pelvis_ty'].to_numpy(), timeVec.to_numpy(), 
                cutoff_frequency=cutoff_frequency)
        else:
            pelvis_ty = ikResults['pelvis_ty']
    dt = timeVec[1] - timeVec[0]
    
    # Identify minimum.
    pelvSignal = np.array(pelvis_ty - np.min(pelvis_ty))
    pelvVel = np.diff(pelvSignal,append=0)/dt
    idxMaxPelvTy,_ = signal.find_peaks(pelvSignal,distance=.9/dt,height=.2,
                                prominence=.2)
    
    # Find the max adjacent to all of the minimums.
    maxIdxOld = 0
    startFinishInds = []
    for i, maxIdx in enumerate(idxMaxPelvTy):     
        # Find velocity peak to left of pelv_ty peak.
        vels = pelvVel[maxIdxOld:maxIdx]
        velPeak,peakVals = signal.find_peaks(vels,distance=.9/dt,height=.2) 
        velPeak = velPeak[np.argmax(peakVals['peak_heights'])] + maxIdxOld        
        velsLeftOfPeak = np.flip(pelvVel[maxIdxOld:velPeak])
        velsRightOfPeak = pelvVel[velPeak:]        
        # Trace left off the pelv_ty peak and find first index where
        # velocity<velSeated m/s.
        slowingIndLeft = np.argwhere(velsLeftOfPeak<velSeated)[0]
        startIdx = velPeak - slowingIndLeft
        slowingIndRight = np.argwhere(velsRightOfPeak<velStanding)[0]
        endIdx = velPeak + slowingIndRight
        startFinishInds.append([startIdx[0],endIdx[0]])
        maxIdxOld = np.copy(maxIdx)
    risingTimes = [timeVec[i].tolist() for i in startFinishInds]  
        
    # We add a delay to make sure we do not simulate part of the motion
    # involving chair contact; this is not modeled.        
    sf = 1/np.round(np.mean(np.round(timeVec.to_numpy()[1:] - 
                                     timeVec.to_numpy()[:-1],2)),16)
    startFinishIndsDelay = []
    for i in startFinishInds:
        c_i = []
        for c_j, j in enumerate(i):
            if c_j == 0:
                c_i.append(j + int(delay*sf))
            else:
                c_i.append(j)
        startFinishIndsDelay.append(c_i)
    risingTimesDelayedStart = [
        timeVec[i].tolist() for i in startFinishIndsDelay]
    
    # Segment periodic STS by identifying when the pelvis_ty value from the
    # standing phase best matches that from the sitting phase.
    startFinishIndsDelayPeriodic = []
    for val in startFinishIndsDelay:
        pelvVal_up = pelvSignal[val[0]]
        # Find next index when pelvis_ty is lower than this value.
        val_down = (np.argwhere(pelvSignal[val[0]+1:] < pelvVal_up)[0][0])
        # Add trimmed part.
        val_down += (val[0]+1)
        # Select val_down or val_down-1 based on best match with pelvVal_up.
        if (np.abs(pelvSignal[val_down] - pelvVal_up) > 
            np.abs(pelvSignal[val_down-1] - pelvVal_up)):
            val_down -= 1
        startFinishIndsDelayPeriodic.append([val[0], val_down])
    risingSittingTimesDelayedStartPeriodicEnd = [
        timeVec[i].tolist() for i in startFinishIndsDelayPeriodic]
    
    if visualize:        
        plt.figure()     
        plt.plot(pelvSignal)
        for c_v, val in enumerate(startFinishInds):
            plt.plot(val, pelvSignal[val], marker='o', markerfacecolor='k',
                     markeredgecolor='none', linestyle='none', 
                     label='Rising phase')
            val2 = startFinishIndsDelay[c_v][0]
            plt.plot(val2, pelvSignal[val2], marker='o',
                     markerfacecolor='r', markeredgecolor='none',
                     linestyle='none', label='Delayed start')
            val3 = startFinishIndsDelayPeriodic[c_v][1]
            plt.plot(val3, pelvSignal[val3], marker='o',
                     markerfacecolor='g', markeredgecolor='none',
                     linestyle='none', 
                     label='Periodic end corresponding to delayed start')
            if c_v == 0:
                plt.legend()
        plt.xlabel('Frames')
        plt.ylabel('Position [m]')
        plt.title('Vertical pelvis position')
        plt.tight_layout()
        plt.draw()
    
    return (risingTimes, risingTimesDelayedStart, 
            risingSittingTimesDelayedStartPeriodicEnd)

# %% Generate model with adjusted muscle wrapping to prevent unrealistic
# wrapping giving rise to bad muscle-tendon lengths and moment arms. Changes
# are made for the gmax1, iliacus, and psoas. Changes are documented in
# modelAdjustment.log.
def adjust_muscle_wrapping(
        baseDir, dataDir, poseDetector='DefaultPD', 
        cameraSetup='DefaultModel', OpenSimModel="LaiUhlrich2022",
        overwrite=False):
    
    # Paths
    osDir = os.path.join(dataDir, 'OpenSimData')
    pathModelFolder = os.path.join(osDir, 'Model')
    
    # We changed the OpenSim model name after some time:
    # from LaiArnoldModified2017_poly_withArms_weldHand to LaiUhlrich2022.
    # This is a hack for backward compatibility.
    if OpenSimModel == 'LaiArnoldModified2017_poly_withArms_weldHand':
        unscaledModelName = 'LaiUhlrich2022'
    else:
        unscaledModelName = OpenSimModel
    
    pathUnscaledModel = os.path.join(baseDir, 'OpenSimPipeline', 'Models',
                                     unscaledModelName + '.osim')
    pathScaledModel = os.path.join(pathModelFolder,
                                   OpenSimModel + '_scaled.osim')
    pathOutputModel = os.path.join(pathModelFolder,
                                   OpenSimModel + '_scaled_adjusted.osim')
    
    if overwrite is False and os.path.exists(pathOutputModel):
        return
    else:
        print('Adjust muscle wrapping surfaces.')
        
    # Set up logging.
    logPath = os.path.join(pathModelFolder,'modelAdjustment.log')
    if os.path.exists(logPath):
        os.remove(logPath)
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.shutdown()
    logging.basicConfig(filename=logPath,format='%(message)s',
                        level=logging.INFO)
    
    # Load models.
    opensim.Logger.setLevelString('error')
    unscaledModel = opensim.Model(pathUnscaledModel)
    scaledModel = opensim.Model(pathScaledModel)    
    scaledBodySet = scaledModel.getBodySet()
    
    # Poses that often cause problems.
    pose_gmax = [
        [['hip_flexion_r',90],['hip_adduction_r',-26], ['hip_rotation_r',40]]]    
    coord_gmax = 'hip_flexion_r'
    
    # generic model doesn't wrap beyond 32deg abd.
    pose_hipFlexors = [
        [['hip_flexion_r',-30],['hip_adduction_r',-32],['hip_rotation_r',-36]],
        [['hip_flexion_r',-30],['hip_adduction_r',-50],['hip_rotation_r',0]],
        [['hip_flexion_r',-30],['hip_adduction_r',30],['hip_rotation_r',0]]] 
    coord_hipFlexors = 'hip_flexion_r'
    
    # Gmax1 - shrink wrap cyl radius.
    momentArmsGmax_unscaled = getMomentArms(
        unscaledModel,pose_gmax,'glmax1_r',coord_gmax)
    momentArmsGmax_scaled = getMomentArms(
        scaledModel,pose_gmax,'glmax1_r',coord_gmax)    
    # Get wrapping surface.
    pelvis = scaledBodySet.get('pelvis')
    gmaxWrap = opensim.WrapCylinder.safeDownCast(
        pelvis.getWrapObjectSet().get('Gmax1_at_pelvis_r'))
    radius = gmaxWrap.get_radius()
    originalRadius = np.copy(radius)
    
    for iPose,(momentArmGmax_scaled,momentArmGmax_unscaled) in enumerate(zip(momentArmsGmax_scaled,momentArmsGmax_unscaled)): 
        if np.abs(momentArmGmax_scaled) < np.max([0.5* np.abs(momentArmGmax_unscaled), 0.008]): # This constant came from 100 scaled models
            originalBadMomentArm = np.copy(momentArmGmax_scaled)            
            while np.abs(momentArmGmax_scaled) <= np.abs(originalBadMomentArm) and radius > 0.002:
                gmaxWrap.set_radius(radius-0.002) 
                momentArmGmax_scaled = getMomentArms(scaledModel,pose_gmax,'glmax1_r',coord_gmax)[iPose]
                radius = gmaxWrap.get_radius()                
            if radius > 0.5*originalRadius:
                outputStr = '-For pose #{}, scaled gmax1 moment arm was {:.3f}. Unscaled was {:.3f}. Reduced R&L wrap radius from {:.3f} to {:.3f}, which increased the moment arm back to {:.3f}.'.format(
                              iPose, originalBadMomentArm,momentArmGmax_unscaled,
                              originalRadius,radius,momentArmGmax_scaled)
                print(outputStr)
                logging.info(outputStr)
                # Set the left side as well.
                opensim.WrapCylinder.safeDownCast(pelvis.getWrapObjectSet().get('Gmax1_at_pelvis_l')).set_radius(radius)        
            else:
                outputStr = '-For pose #{}, couldn''t restore glmax1 moment arm by shrinking radius by 1/2. Model unchanged.'.format(iPose)
                print(outputStr)
                logging.info(outputStr)
                gmaxWrap.set_radius(float(originalRadius))        
            scaledModel.initSystem()       
        else:
            outputStr = '-For pose #{}, scaled gmax1 moment arm was {:.3f}. Unscaled was {:.3f}. No adjustements made.'.format(
                         iPose,np.abs(momentArmGmax_scaled),np.abs(momentArmGmax_unscaled))
            print(outputStr)
            logging.info(outputStr)
    
    # Iliacus - change path points to engage wrap cylinder.
    momentArms_unscaled = getMomentArms(
        unscaledModel,pose_hipFlexors,'iliacus_r',coord_hipFlexors)
    momentArms_scaled = getMomentArms(
        scaledModel,pose_hipFlexors,'iliacus_r',coord_hipFlexors)    
    # Get path point locations.
    muscle = scaledModel.getMuscles().get('iliacus_r')
    pathPoints = muscle.get_GeometryPath().getPathPointSet()
    point1 = opensim.PathPoint.safeDownCast(pathPoints.get(1))
    loc1Vec = point1.get_location()
    point2 = opensim.PathPoint.safeDownCast(pathPoints.get(2))
    loc2Vec = point2.get_location()    
    original_loc1 = [loc1Vec[i] for i in range(3)]
    original_loc2 = [loc2Vec[i] for i in range(3)]
    # Get wrap cyl.        
    wrapCyl = opensim.WrapCylinder.safeDownCast(
        pelvis.getWrapObjectSet().get('IL_at_brim_r'))
    radius = wrapCyl.get_radius()
    originalRadius = np.copy(radius)
    previousRadius = np.copy(radius)
    
    for iPose,(momentArm_scaled,momentArm_unscaled) in enumerate(zip(momentArms_scaled,momentArms_unscaled)):
        if np.abs(momentArm_scaled) < np.max([0.7* np.abs(momentArm_unscaled) , 0.015]):             
            # Get path point locations.
            muscle = scaledModel.getMuscles().get('iliacus_r')
            pathPoints = muscle.get_GeometryPath().getPathPointSet()
            point1 = opensim.PathPoint.safeDownCast(pathPoints.get(1))
            loc1Vec = point1.get_location()
            point2 = opensim.PathPoint.safeDownCast(pathPoints.get(2))
            loc2Vec = point2.get_location()            
            originalBadMomentArm = np.copy(momentArm_scaled)                           
            while np.abs(momentArm_scaled) <= np.max([0.7* np.abs(momentArm_unscaled) , 0.015]) and (np.abs(loc1Vec[0]-original_loc1[0]) < 0.015 and np.abs(loc2Vec[1]-original_loc2[1]) <0.015):
                loc1Vec[0] += 0.002 # Move the 1st (pelvis) path point forward
                loc2Vec[1] -= 0.002 # move the 2nd (femur) path point down
                point1.set_location(loc1Vec)
                point2.set_location(loc2Vec)        
                momentArm_scaled = getMomentArms(scaledModel,pose_hipFlexors,'iliacus_r',coord_hipFlexors)[iPose]          
            while np.abs(momentArm_scaled) <= np.max([0.7* np.abs(momentArm_unscaled) , 0.015]) and radius>0.7*originalRadius: # above approach did not succeed, drop the cyl radius some
                wrapCyl.set_radius(radius-0.002) 
                momentArm_scaled = getMomentArms(scaledModel,pose_hipFlexors,'iliacus_r',coord_hipFlexors)[iPose]
                pelvis = scaledBodySet.get('pelvis')
                radius = wrapCyl.get_radius()
            if np.abs(momentArm_scaled) > np.max([0.7* np.abs(momentArm_unscaled) , 0.015]): # succeeded
                # Set the left side as well.
                muscle = scaledModel.getMuscles().get('iliacus_l')
                pathPoints = muscle.get_GeometryPath().getPathPointSet()        
                point1 = opensim.PathPoint.safeDownCast(pathPoints.get(1))
                loc1Vec_l = point1.get_location()
                loc1Vec_l[0] = loc1Vec[0]
                point1.set_location(loc1Vec_l)                
                point2 = opensim.PathPoint.safeDownCast(pathPoints.get(2))
                loc2Vec_l = point2.get_location()
                loc2Vec_l[1] = loc2Vec[1]
                point2.set_location(loc2Vec_l)                
                if radius<previousRadius:
                    radiusStr = ', and after moving points by 1.5±0.2cm wasn''t enough, reduced R&L iliacus wrap radius from {:.3f} to {:.3f}'.format(
                    originalRadius,radius)
                    # set the left side as well.
                    opensim.WrapCylinder.safeDownCast(pelvis.getWrapObjectSet().get('IL_at_brim_l')).set_radius(radius)
                else:
                    radiusStr = ''
                previousRadius = np.copy(radius)    
                outputStr = '-For pose #{}, moved iliacus pelvis path point xpos forward from {:.3f} to {:.3f}, and femur iliacus path point ypos down from {:.3f} to {:.3f}'.format(
                    iPose,original_loc1[0],loc1Vec[0],original_loc2[1],loc2Vec[1]) + radiusStr + '. Restored moment arm from {:.3f} to {:.3f}.'.format(
                      originalBadMomentArm,momentArm_scaled)
                print(outputStr)
                logging.info(outputStr)
            else:
                outputStr = '-For pose #{}, couldn''t restore iliacus moment arm by moving path points by 2cm. Model unchanged.'.format(iPose)
                print(outputStr)
                logging.info(outputStr)                
                point1.set_location(original_loc1)
                point2.set_location(original_loc2)            
            scaledModel.initSystem()
        else:
            outputStr = '-For pose #{}, scaled iliacus moment arm was {:.3f}. Unscaled was {:.3f}. No adjustements made.'.format(
                  iPose,np.abs(momentArm_scaled),np.abs(momentArm_unscaled))
            print(outputStr)
            logging.info(outputStr)
    
    # Psoas - change path points to engage wrap cylinder.
    momentArms_unscaled = getMomentArms(
        unscaledModel,pose_hipFlexors,'psoas_r',coord_hipFlexors)
    momentArms_scaled = getMomentArms(
        scaledModel,pose_hipFlexors,'psoas_r',coord_hipFlexors)    
    # Get path point locations 
    muscle = scaledModel.getMuscles().get('psoas_r')
    pathPoints = muscle.get_GeometryPath().getPathPointSet()
    point1 = opensim.PathPoint.safeDownCast(pathPoints.get(1))
    loc1Vec = point1.get_location()
    point2 = opensim.PathPoint.safeDownCast(pathPoints.get(2))
    loc2Vec = point2.get_location()    
    original_loc1 = [loc1Vec[i] for i in range(3)]
    original_loc2 = [loc2Vec[i] for i in range(3)]
    # Get wrap cyl         
    wrapCyl = opensim.WrapCylinder.safeDownCast(
        pelvis.getWrapObjectSet().get('PS_at_brim_r'))
    radius = wrapCyl.get_radius()
    originalRadius = np.copy(radius)
    previousRadius = np.copy(radius)
    
    for iPose,(momentArm_scaled,momentArm_unscaled) in enumerate(zip(momentArms_scaled,momentArms_unscaled)):
        if np.abs(momentArm_scaled) < np.max([0.7* np.abs(momentArm_unscaled), 0.015]):            
            # Get path point locations.
            muscle = scaledModel.getMuscles().get('psoas_r')
            pathPoints = muscle.get_GeometryPath().getPathPointSet()
            point1 = opensim.PathPoint.safeDownCast(pathPoints.get(1))
            loc1Vec = point1.get_location()
            point2 = opensim.PathPoint.safeDownCast(pathPoints.get(2))
            loc2Vec = point2.get_location()
            originalBadMomentArm = np.copy(momentArm_scaled)               
            while np.abs(momentArm_scaled) <= np.max([0.7* np.abs(momentArm_unscaled), 0.015]) and (np.abs(loc1Vec[0]-original_loc1[0]) < 0.015 and np.abs(loc2Vec[1]-original_loc2[1]) < 0.015):
                loc1Vec[0] += 0.002 # Move the 1st (pelvis) path point forward
                loc2Vec[1] -= 0.002 # move the 2nd (femur) path point down
                point1.set_location(loc1Vec)
                point2.set_location(loc2Vec)        
                momentArm_scaled = getMomentArms(scaledModel,pose_hipFlexors,'psoas_r',coord_hipFlexors)[iPose]            
            while np.abs(momentArm_scaled) <= np.max([0.7* np.abs(momentArm_unscaled) , 0.015]) and radius>0.7*originalRadius: #above approach did not succeed, drop the cyl radius some
                wrapCyl.set_radius(radius-0.002) 
                momentArm_scaled = getMomentArms(scaledModel,pose_hipFlexors,'psoas_r',coord_hipFlexors)[iPose]
                pelvis = scaledBodySet.get('pelvis')
                radius = wrapCyl.get_radius()
            if np.abs(momentArm_scaled) > np.max([0.7* np.abs(momentArm_unscaled) , 0.015]): # succeeded
                # set the left side as well.
                muscle = scaledModel.getMuscles().get('psoas_l')
                pathPoints = muscle.get_GeometryPath().getPathPointSet()        
                point1 = opensim.PathPoint.safeDownCast(pathPoints.get(1))
                loc1Vec_l = point1.get_location()
                loc1Vec_l[0] = loc1Vec[0]
                point1.set_location(loc1Vec_l)
                point2 = opensim.PathPoint.safeDownCast(pathPoints.get(2))
                loc2Vec_l = point2.get_location()
                loc2Vec_l[1] = loc2Vec[1]
                point2.set_location(loc2Vec_l)
                if radius<previousRadius:
                    radiusStr = ', and after moving points by 1.5±0.2cm wasn''t enough, reduced R&L psoas wrap radius from {:.3f} to {:.3f}'.format(
                    originalRadius,radius)
                    # set the left side as well.
                    opensim.WrapCylinder.safeDownCast(pelvis.getWrapObjectSet().get('PS_at_brim_l')).set_radius(radius)
                else:
                    radiusStr = ''
                previousRadius = np.copy(radius)   
                outputStr = '-For pose #{}, moved psoas pelvis path point xpos forward from {:.3f} to {:.3f}, and femur psoas path point ypos down from {:.3f} to {:.3f}'.format(
                    iPose,original_loc1[0],loc1Vec[0],original_loc2[1],loc2Vec[1]) + radiusStr + '. Restored moment arm from {:.3f} to {:.3f}.'.format(
                      originalBadMomentArm,momentArm_scaled)
                print(outputStr)
                logging.info(outputStr)                   
            else:
                outputStr = '-For pose #{}, couldn''t restore psoas moment arm by moving path points by 2cm. Model unchanged.'.format(iPose)
                print(outputStr)
                logging.info(outputStr)                
                point1.set_location(opensim.Vec3(original_loc1))
                point2.set_location(opensim.Vec3(original_loc2))          
            scaledModel.initSystem()           
        else:
            outputStr = '-For pose #{}, scaled psoas moment arm was {:.3f}. Unscaled was {:.3f}. No adjustements made.'.format(
                  iPose,np.abs(momentArm_scaled),np.abs(momentArm_unscaled))
            print(outputStr)
            logging.info(outputStr)
    
    scaledModel.printToXML(pathOutputModel)
    logging.shutdown()
    
# %% Pose the models and get moment arms.
def getMomentArms(model, poses, muscleName, coordinateForMomentArm):
    state = model.initSystem()
    coords = model.getCoordinateSet()
    muscleSet = model.getMuscles()
    coordForMA = coords.get(coordinateForMomentArm)
    momentArms = []
    for i, pose in enumerate(poses):        
        for coordVal in pose:
            coords.get(coordVal[0]).setValue(state,np.deg2rad(coordVal[1]))
        momentArms.append(
            muscleSet.get(muscleName).computeMomentArm(state,coordForMA))
        
    return momentArms

# %% Generate model with contacts.
def generate_model_with_contacts(
        dataDir, poseDetector='DefaultPD', cameraSetup='DefaultModel',
        OpenSimModel="LaiUhlrich2022", setPatellaMasstoZero=True, 
        overwrite=False, contact_configuration='generic'):
    
    # %% Process settings.
    osDir = os.path.join(dataDir, 'OpenSimData')
    # pathModelFolder = os.path.join(osDir, poseDetector, cameraSetup, 'Model')
    pathModelFolder = os.path.join(osDir, 'Model')
    suffix_MA = '_adjusted'
    outputModelFileName = (OpenSimModel + "_scaled" + suffix_MA)
    pathOutputFiles = os.path.join(pathModelFolder, outputModelFileName)    
    if contact_configuration == 'generic':
        suffix_contact = '_contacts'
    elif contact_configuration == 'dhondt2023':
        suffix_contact = '_contacts_dhondt2023'
    pathOutputModel = pathOutputFiles + suffix_contact + ".osim"
    
    if overwrite is False and os.path.exists(pathOutputModel):
        return
    else:
        print('Add foot-ground contacts.')
        
    # %% Add contact spheres to the scaled model.
    # The parameters of the foot-ground contacts are based on previous work. We
    # scale the contact sphere locations based on foot dimensions.
    if contact_configuration == 'generic':
        reference_contact_spheres = {
            "s1_r": {"radius": 0.032, "location": np.array([0.0019011578840796601,   -0.01,  -0.00382630379623308]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_r"},
            "s2_r": {"radius": 0.032, "location": np.array([0.14838639994206301,     -0.01,  -0.028713422052654002]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_r"},
            "s3_r": {"radius": 0.032, "location": np.array([0.13300117060705099,     -0.01,  0.051636247344956601]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_r"},
            "s4_r": {"radius": 0.032, "location": np.array([0.066234666199163503,    -0.01,  0.026364160674169801]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_r"},
            "s5_r": {"radius": 0.032, "location": np.array([0.059999999999999998,    -0.01,  -0.018760308461917698]), "orientation": np.array([0, 0, 0]), "socket_frame": "toes_r" },
            "s6_r": {"radius": 0.032, "location": np.array([0.044999999999999998,    -0.01,  0.061856956754965199]), "orientation": np.array([0, 0, 0]), "socket_frame": "toes_r" },
            "s1_l": {"radius": 0.032, "location": np.array([0.0019011578840796601,   -0.01,  0.00382630379623308]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_l"},
            "s2_l": {"radius": 0.032, "location": np.array([0.14838639994206301,     -0.01,  0.028713422052654002]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_l"},
            "s3_l": {"radius": 0.032, "location": np.array([0.13300117060705099,     -0.01,  -0.051636247344956601]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_l"},
            "s4_l": {"radius": 0.032, "location": np.array([0.066234666199163503,    -0.01,  -0.026364160674169801]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_l"},
            "s5_l": {"radius": 0.032, "location": np.array([0.059999999999999998,    -0.01,  0.018760308461917698]), "orientation": np.array([0, 0, 0]), "socket_frame": "toes_l" },
            "s6_l": {"radius": 0.032, "location": np.array([0.044999999999999998,    -0.01,  -0.061856956754965199]), "orientation": np.array([0, 0, 0]), "socket_frame": "toes_l" }}      
        reference_scale_factors = {"calcn_r": np.array([0.91392399999999996, 0.91392399999999996, 0.91392399999999996]),
                                "toes_r":  np.array([0.91392399999999996, 0.91392399999999996, 0.91392399999999996]),
                                "calcn_l": np.array([0.91392399999999996, 0.91392399999999996, 0.91392399999999996]),
                                "toes_l":  np.array([0.91392399999999996, 0.91392399999999996, 0.91392399999999996])}
    
    elif contact_configuration == 'dhondt2023':
        reference_contact_spheres = {
            "s1_r": {"radius": 0.032, "location": np.array([0.01, 0.0069229175108780888, -0.0049972000000000003]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_r"},
            "s2_r": {"radius": 0.032, "location": np.array([0.059999999999999998, 0.01192291751087809, 0.020001000000000001]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_r"},
            "s3_r": {"radius": 0.023, "location": np.array([0.16420673660730933, 0.0021994834115391451, 0.020558107002]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_r"},
            "s4_r": {"radius": 0.021, "location": np.array([0.16420673660730933, 0.0021994834115391451, -0.010624592998000001]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_r"},
            "s5_r": {"radius": 0.016, "location": np.array([0.053154, -0.0015385412445609557, -0.0034172999999999999]), "orientation": np.array([0, 0, 0]), "socket_frame": "toes_r" },
            "s1_l": {"radius": 0.032, "location": np.array([0.01, 0.0069229175108780888, 0.0049972000000000003]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_l"},
            "s2_l": {"radius": 0.032, "location": np.array([0.059999999999999998, 0.01192291751087809, -0.020001000000000001]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_l"},
            "s3_l": {"radius": 0.023, "location": np.array([0.16420673660730933, 0.0021994834115391451, -0.020558107002]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_l"},
            "s4_l": {"radius": 0.021, "location": np.array([0.16420673660730933, 0.0021994834115391451, 0.010624592998000001]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_l"},
            "s5_l": {"radius": 0.016, "location": np.array([0.053154, -0.0015385412445609557, 0.0034172999999999999]), "orientation": np.array([0, 0, 0]), "socket_frame": "toes_l" }}
        reference_scale_factors = {"calcn_r": np.array([0.91392399999999996, 0.91392399999999996, 0.91392399999999996]),
                                "toes_r":  np.array([0.91392399999999996, 0.91392399999999996, 0.91392399999999996]),
                                "calcn_l": np.array([0.91392399999999996, 0.91392399999999996, 0.91392399999999996]),
                                "toes_l":  np.array([0.91392399999999996, 0.91392399999999996, 0.91392399999999996])}

    reference_contact_half_space = {"name": "floor", "location": np.array([0, 0, 0]),"orientation": np.array([0, 0, -np.pi/2]), "frame": "ground"}

    if contact_configuration == 'generic':
        stiffness = 1000000
    elif contact_configuration == 'dhondt2023':
        stiffness = 10000000
    dissipation = 2.0
    static_friction = 0.8
    dynamic_friction = 0.8
    viscous_friction = 0.5
    transition_velocity = 0.2
    
    # Add contact spheres and SmoothSphereHalfSpaceForces.
    opensim.Logger.setLevelString('error')
    model = opensim.Model(pathOutputFiles + ".osim")   
    bodySet = model.get_BodySet()
    
    # ContactHalfSpace.
    if reference_contact_half_space["frame"] == "ground":
        contact_half_space_frame = model.get_ground()
    else:
        raise ValueError('Not yet supported.')    
    contactHalfSpace = opensim.ContactHalfSpace(
        opensim.Vec3(reference_contact_half_space["location"]),
        opensim.Vec3(reference_contact_half_space["orientation"]),
        contact_half_space_frame, reference_contact_half_space["name"])
    contactHalfSpace.connectSocket_frame(contact_half_space_frame)
    model.addContactGeometry(contactHalfSpace)
    
    # ContactSpheres and SmoothSphereHalfSpaceForces.
    for ref_contact_sphere in reference_contact_spheres:    
        # ContactSpheres.
        body = bodySet.get(reference_contact_spheres[ref_contact_sphere]["socket_frame"])
        # Scale location based on attached_geometry scale_factors.      
        # We don't scale the y_position.
        attached_geometry = body.get_attached_geometry(0)
        c_scale_factors = attached_geometry.get_scale_factors().to_numpy() 
        c_ref_scale_factors = reference_scale_factors[reference_contact_spheres[ref_contact_sphere]["socket_frame"]]
        scale_factors = c_ref_scale_factors / c_scale_factors        
        scale_factors[1] = 1        
        scaled_location = reference_contact_spheres[ref_contact_sphere]["location"] / scale_factors
        c_contactSphere = opensim.ContactSphere(
            reference_contact_spheres[ref_contact_sphere]["radius"],
            opensim.Vec3(scaled_location), body, ref_contact_sphere)
        c_contactSphere.connectSocket_frame(body)
        model.addContactGeometry(c_contactSphere)
        
        # SmoothSphereHalfSpaceForces.
        SmoothSphereHalfSpaceForce = opensim.SmoothSphereHalfSpaceForce(
            "SmoothSphereHalfSpaceForce_" + ref_contact_sphere, 
            c_contactSphere, contactHalfSpace)
        SmoothSphereHalfSpaceForce.set_stiffness(stiffness)
        SmoothSphereHalfSpaceForce.set_dissipation(dissipation)
        SmoothSphereHalfSpaceForce.set_static_friction(static_friction)
        SmoothSphereHalfSpaceForce.set_dynamic_friction(dynamic_friction)
        SmoothSphereHalfSpaceForce.set_viscous_friction(viscous_friction)
        SmoothSphereHalfSpaceForce.set_transition_velocity(transition_velocity)        
        SmoothSphereHalfSpaceForce.connectSocket_half_space(contactHalfSpace)
        SmoothSphereHalfSpaceForce.connectSocket_sphere(c_contactSphere)
        model.addForce(SmoothSphereHalfSpaceForce)
    
    # We do not use the patella in the dynamic simulations. The reason is that
    # the patella only matters for the muscle-tendon lengths and moment arms,
    # but since we approximate those with polynomials, the patella is useless.
    # We therefore remove it, since otherwise we would have to deal with
    # kinematic constraints that would make things unecessarily complicated.
    # We remove it when building the external function, and here we set its
    # mass to zero such that we can make an apple-to-apple comparison when
    # checking that the outputs from the external function match the results
    # from ID ran with the model (with a mass set to 0, the patella will not
    # influence ID).
    if setPatellaMasstoZero:
        for i in range(bodySet.getSize()):        
            c_body = bodySet.get(i)
            c_body_name = c_body.getName()            
            if (c_body_name == 'patella_l' or c_body_name == 'patella_r'):
                c_body.set_mass(0.)
                c_body.set_inertia(opensim.Vec6(0))
        
    model.finalizeConnections
    model.initSystem()
    model.printToXML(pathOutputModel)
    
# %% Align marker data with ground.
# When the checkerboard is not perfectly aligned with the ground, then it might
# look like if the subject is going uphill or downhill. This function computes
# the angle between the checkerboard and the ground, and then rotates the
# marker data such that the checkerboard is aligned with the ground. 
def align_markers_with_ground_3(sessionDir, trialName,
                                referenceMarker='Neck',
                                suffixOutputFileName='aligned',
                                lowpass_cutoff_frequency_for_marker_values=-1,
                                addOffset=True, visualize=False, angle=None,
                                select_window=[]):

    pathTRCFile = os.path.join(sessionDir, 'MarkerData', trialName + '.trc')
    trc_file = TRCFile(pathTRCFile)
    time = trc_file.time

    # Find index in time vector corresponding to the start and end of the trial
    # as defined in select_window. Start is given by first entry in select_window
    # and end is given by the second entry. If the entry is -1 then the start or
    # end of the trial is used. If select_window is empty then skip this step.
    use_select_window  = False
    if len(select_window) > 0:
        if select_window[0] == -1:
            start = 0
        else:
            start = np.argmin(np.abs(time-select_window[0]))
        if select_window[1] == -1:
            end = len(time)
        else:
            end = np.argmin(np.abs(time-select_window[1]))+1

        markers = trc_file.marker_names
        marker_data = np.zeros((len(time), 3*len(markers)))
        # marker_data[:,0] = time
        for i, marker in enumerate(markers):
            marker_data[:,3*i:3*i+3] = trc_file.marker(marker)
        marker_data_adj = marker_data[start:end,:]

        pathTRCFile_out = os.path.join(
            sessionDir, 'MarkerData', 
            trialName + '_{}.trc'.format('trimmed'))

        with open(pathTRCFile_out,"w") as f:
            numpy2TRC(f, marker_data_adj, markers, trc_file.camera_rate, time[start])
            
        trc_file = TRCFile(pathTRCFile_out)
        time = trc_file.time
        
        use_select_window = True

    # Extract data from reference markers.
    m = trc_file.marker(referenceMarker)
    if lowpass_cutoff_frequency_for_marker_values > 0:
        m = lowPassFilter(time, m, lowpass_cutoff_frequency_for_marker_values)
        
    # spline = interpolate.InterpolatedUnivariateSpline(time, m[:,1], k=3)
    # splineD1 = spline.derivative(n=1)
    # mid_m_speed = splineD1(time)
        
    if use_select_window:
        cutEnd = False
    else:
        cutEnd = True
    if cutEnd:
        sf = trc_file.camera_rate
        end_offset = -int(0.3*sf)
    else:
        end_offset = len(time)
        
    cutStart = True
    if cutStart:
        sf = trc_file.camera_rate
        start_offset = -int(2.3*sf)
    else:
        start_offset = len(time)
        
    # trim m and time
    m = m[start_offset:end_offset,:]
    time = time[start_offset:end_offset]

        
    if visualize:
        plt.figure()
        plt.plot(time, m[:,1])
        # plt.plot(time[:end_offset], mid_m_speed[:end_offset])
        # plt.plot(time, center_of_mass_speeds['y'])
        # plt.plot(time, mid_m_speed_filt)
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Vertical position of {}'.format(referenceMarker))
        plt.show()
        
        
        # fig, ax1 = plt.subplots()
        # plt.plot(time[:end_offset], m[:end_offset, 1], label='Position')
        # ax1.set_xlabel('Time (s)')
        # ax1.set_ylabel('Position (m)', color='tab:blue')
        # plt.title('Vertical position of {}'.format(referenceMarker))
        # ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # ax2 = ax1.twinx()
        # plt.plot(time[:end_offset], mid_m_speed[:end_offset], 'r', label='Speed')
        # ax2.set_ylabel('Speed', color='tab:red')
        # ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # fig.tight_layout()  # Ensures that the labels do not overlap
        
        # plt.show()
        
        
        
    # spline = interpolate.InterpolatedUnivariateSpline(time, mid_m[:,1], k=3)
    # splineD1 = spline.derivative(n=1)
    # mid_m_speed = splineD1(time)
    
    if angle is None:

        # We assume peak vertical speeds should match across gait cycles.
        peaks, _ = signal.find_peaks(m[:,1], distance=20, width=10, prominence=0.02)
        
        if peaks.shape[0] == 0:
            raise ValueError("No peaks detected")
        
        # time[peaks]
        
        # The beginning of the trial is not always great, we here select the gait
        # cycles that have similar durations (+/-20%).
        # diff_peaks = np.diff([peaks])    
        # for i in range(len(diff_peaks[0])-2, len(diff_peaks[0])-4, -1):
        #     if np.abs(diff_peaks[0][i]-diff_peaks[0][i+1]) > 0.2*diff_peaks[0][i+1]:
        #         break
        
        # Extract marker data at first and last peaks.
        pos_start = m[peaks[0], :]
        pos_end = m[peaks[-1], :]
        
        # Calculate the original vector.
        original_vector = pos_end - pos_start
    
        # The reference vector is the vector along the x-axis.
        reference_vector = np.array([1, 0, 0])
        
        # Calculate the dot product of the two vectors.
        dot_product = np.dot(reference_vector, original_vector)
        
        # Calculate the magnitudes (lengths) of the vectors.
        magnitude_A = np.linalg.norm(reference_vector)
        magnitude_B = np.linalg.norm(original_vector)
        
        # Calculate the angle between the two vectors in radians.
        angle_rad = np.arccos(dot_product / (magnitude_A * magnitude_B))
        
        # Convert the angle from radians to degrees if needed.
        angle_deg = np.degrees(angle_rad)
        
    else:
        
        angle_deg = angle
    
    # Rotate marker data about the z-axis.
    trc_file.rotate('z', angle_deg)
    
    print('Angle between reference vector and original vector: {:.2f} deg'.format(angle_deg))
    
    if addOffset:
        # Compute offset
        markers = trc_file.marker_names
        offset = float('inf')
        # Cut off last 0.5s to avoid issues with the end of the trial.
        
        for marker in markers:
            if '_study' in marker:
                min_y_pos = np.min(trc_file.marker(marker)[start_offset:end_offset,1])
                if min_y_pos < offset:
                    offset = min_y_pos                
        # Subtract offset
        trc_file.offset('y', -offset)
    # trc_file.offset('y', -(pos_end[1]-0.03))       

    # Print a new trc file.
    pathTRCFile_out = os.path.join(
        sessionDir, 'MarkerData', 
        trialName + '_{}.trc'.format(suffixOutputFileName))
    trc_file.write(pathTRCFile_out)

    return pathTRCFile_out

# %% Align marker data with ground.
# When the checkerboard is not perfectly aligned with the ground, then it might
# look like if the subject is going uphill or downhill. This function computes
# the angle between the checkerboard and the ground, and then rotates the
# marker data such that the checkerboard is aligned with the ground. 
def align_markers_with_ground_2(sessionDir, trialName,
                                referenceMarker='r_calc_study',
                                suffixOutputFileName='aligned',
                                lowpass_cutoff_frequency_for_marker_values=-1,
                                addOffset=True, visualize=False):

    pathTRCFile = os.path.join(sessionDir, 'MarkerData', trialName + '.trc')
    trc_file = TRCFile(pathTRCFile)
    time = trc_file.time

    # Extract data from reference markers.
    m = trc_file.marker(referenceMarker)
    if lowpass_cutoff_frequency_for_marker_values > 0:
        m = lowPassFilter(time, m, lowpass_cutoff_frequency_for_marker_values)
        
    cutEnd = True
    if cutEnd:
        sf = trc_file.camera_rate
        end_offset = -int(0.5*sf)
    else:
        end_offset = len(time)

        
    if visualize:
        plt.figure()
        plt.plot(-m[:end_offset,1])
        # plt.plot(time, center_of_mass_speeds['y'])
        # plt.plot(time, mid_m_speed_filt)
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.title('Speed of mid marker')
        plt.show()
        
        
        
    # spline = interpolate.InterpolatedUnivariateSpline(time, mid_m[:,1], k=3)
    # splineD1 = spline.derivative(n=1)
    # mid_m_speed = splineD1(time)

    # We assume peak vertical speeds should match across gait cycles.
    peaks, _ = signal.find_peaks(-m[:end_offset,1], distance=30, 
                                  width=15, prominence=0.02)
    
    # The beginning of the trial is not always great, we here select the gait
    # cycles that have similar durations (+/-20%).
    diff_peaks = np.diff([peaks])    
    for i in range(len(diff_peaks[0])-2, len(diff_peaks[0])-4, -1):
        if np.abs(diff_peaks[0][i]-diff_peaks[0][i+1]) > 0.2*diff_peaks[0][i+1]:
            break
    
    # Extract marker data at first and last peaks.
    pos_start = m[peaks[i+1], :]
    pos_end = m[peaks[-1], :]
    
    # Calculate the original vector.
    original_vector = pos_end - pos_start

    # The reference vector is the vector along the x-axis.
    reference_vector = np.array([1, 0, 0])
    
    # Calculate the dot product of the two vectors.
    dot_product = np.dot(reference_vector, original_vector)
    
    # Calculate the magnitudes (lengths) of the vectors.
    magnitude_A = np.linalg.norm(reference_vector)
    magnitude_B = np.linalg.norm(original_vector)
    
    # Calculate the angle between the two vectors in radians.
    angle_rad = np.arccos(dot_product / (magnitude_A * magnitude_B))
    
    # Convert the angle from radians to degrees if needed.
    angle_deg = np.degrees(angle_rad)
    
    # Rotate marker data about the z-axis.
    trc_file.rotate('z', angle_deg)
    
    print('Angle between reference vector and original vector: {:.2f} deg'.format(angle_deg))
    
    # if addOffset:
    #     # Compute offset
    #     markers = trc_file.marker_names
    #     offset = float('inf')
    #     # Cut off last 0.5s to avoid issues with the end of the trial.
        
    #     for marker in markers:
    #         min_y_pos = np.min(trc_file.marker(marker)[:end_offset,1])
    #         if min_y_pos < offset:
    #             offset = min_y_pos                
    #     # Subtract offset
    #     trc_file.offset('y', -offset)
    trc_file.offset('y', -(pos_end[1]-0.03))       

    # Print a new trc file.
    pathTRCFile_out = os.path.join(
        sessionDir, 'MarkerData', 
        trialName + '_{}.trc'.format(suffixOutputFileName))
    trc_file.write(pathTRCFile_out)

    return pathTRCFile_out


# def align_markers_with_ground(sessionDir, trialName, kinematics_obj,
#                               referenceMarker1='r.PSIS_study',
#                               referenceMarker2='L.PSIS_study',
#                               suffixOutputFileName='aligned',
#                               lowpass_cutoff_frequency_for_marker_values=-1,
#                               addOffset=True, visualize=False,
#                               ):

#     pathTRCFile = os.path.join(sessionDir, 'MarkerData', trialName + '.trc')
#     trc_file = TRCFile(pathTRCFile)
#     time = trc_file.time

#     '''
#     # Extract data from reference markers.
#     m1 = trc_file.marker(referenceMarker1)
#     m2 = trc_file.marker(referenceMarker2)
#     mid_m = (m1+m2)/2
#     if lowpass_cutoff_frequency_for_marker_values > 0:
#         mid_m = lowPassFilter(
#             time, mid_m, lowpass_cutoff_frequency_for_marker_values)
#     spline = interpolate.InterpolatedUnivariateSpline(time, mid_m[:,1], k=3)
#     splineD1 = spline.derivative(n=1)
#     mid_m_speed = splineD1(time)
#     mid_m_speed_filt = lowPassFilter(
#         time, mid_m_speed, lowpass_cutoff_frequency_for_marker_values)
#     '''
    
#     center_of_mass_values = kinematics_obj.get_center_of_mass_values(lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_marker_values)
#     # center_of_mass_speeds = kinematics_obj.get_center_of_mass_speeds(lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_marker_values)

#     cutEnd = True
#     if cutEnd:
#         sf = trc_file.camera_rate
#         end_offset = -int(0.5*sf)
#     else:
#         end_offset = len(time)

#     if visualize:
#         plt.figure()
#         plt.plot(time[:end_offset], center_of_mass_values['y'][:end_offset])
#         # plt.plot(time, center_of_mass_speeds['y'])
#         # plt.plot(time, mid_m_speed_filt)
#         plt.xlabel('Time (s)')
#         plt.ylabel('Speed (m/s)')
#         plt.title('Speed of mid marker')
#         plt.show()
        
    
        
#     mid_m = center_of_mass_values.to_numpy()[:,1:]

#     # We assume peak vertical speeds should match across gait cycles.
#     # peaks, _ = signal.find_peaks(mid_m_speed, distance=30, 
#     #                              width=10, prominence=0.05)
#     # peaks, _ = signal.find_peaks(center_of_mass_speeds['y'], distance=30, 
#     #                              width=10, prominence=0.1)
#     peaks, _ = signal.find_peaks(center_of_mass_values['y'][:end_offset], distance=30, 
#                                  width=10)
    
#     # The beginning of the trial is not always great, we here select the gait
#     # cycles that have similar durations (+/-20%).
#     diff_peaks = np.diff([peaks])    
#     for i in range(len(diff_peaks[0])-2, len(diff_peaks[0])-4, -1):
#         if np.abs(diff_peaks[0][i]-diff_peaks[0][i+1]) > 0.2*diff_peaks[0][i+1]:
#             break

#     pos_start = mid_m[peaks[-2], :]
#     pos_end = mid_m[peaks[-1], :]
    
#     # Calculate the original vector.
#     original_vector = pos_end - pos_start

#     # The reference vector is the vector along the x-axis.
#     reference_vector = np.array([1, 0, 0])
    
#     # Calculate the dot product of the two vectors.
#     dot_product = np.dot(reference_vector, original_vector)
    
#     # Calculate the magnitudes (lengths) of the vectors.
#     magnitude_A = np.linalg.norm(reference_vector)
#     magnitude_B = np.linalg.norm(original_vector)
    
#     # Calculate the angle between the two vectors in radians.
#     angle_rad = np.arccos(dot_product / (magnitude_A * magnitude_B))
    
#     # Convert the angle from radians to degrees if needed.
#     angle_deg = np.degrees(angle_rad)
    
#     print('Angle between reference vector and original vector: {:.2f} deg'.format(angle_deg))
    
#     # Extract marker data at first and last peaks.
#     # angles_deg = []
#     # count = 0
#     # for j in range(i+1, peaks.shape[0]-1):
    
#     #     pos_start = mid_m[peaks[j], :]
#     #     pos_end = mid_m[peaks[-1], :]
        
#     #     # Calculate the original vector.
#     #     original_vector = pos_end - pos_start
    
#     #     # The reference vector is the vector along the x-axis.
#     #     reference_vector = np.array([1, 0, 0])
        
#     #     # Calculate the dot product of the two vectors.
#     #     dot_product = np.dot(reference_vector, original_vector)
        
#     #     # Calculate the magnitudes (lengths) of the vectors.
#     #     magnitude_A = np.linalg.norm(reference_vector)
#     #     magnitude_B = np.linalg.norm(original_vector)
        
#     #     # Calculate the angle between the two vectors in radians.
#     #     angle_rad = np.arccos(dot_product / (magnitude_A * magnitude_B))
        
#     #     # Convert the angle from radians to degrees if needed.
#     #     angles_deg.append(np.degrees(angle_rad))
        
#     #     print('Angle between reference vector and original vector: {:.2f} deg'.format(angles_deg[count]))
#     #     count += 1
#     # angle_deg = np.mean(angles_deg)
#     # print('Final angle between reference vector and original vector: {:.2f} deg'.format(angle_deg))
    
#     # Rotate marker data about the z-axis.
#     trc_file.rotate('z', angle_deg)
    
    
#     if addOffset:
#         # Compute offset
#         markers = trc_file.marker_names
#         offset = float('inf')
#         # Cut off last 0.5s to avoid issues with the end of the trial.
        
#         for marker in markers:
#             min_y_pos = np.min(trc_file.marker(marker)[:end_offset,1])
#             if min_y_pos < offset:
#                 offset = min_y_pos                
#         # Subtract offset
#         trc_file.offset('y', -offset)       

#     # Print a new trc file.
#     pathTRCFile_out = os.path.join(
#         sessionDir, 'MarkerData', 
#         trialName + '_{}.trc'.format(suffixOutputFileName))
#     trc_file.write(pathTRCFile_out)

#     return pathTRCFile_out
