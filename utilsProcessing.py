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

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from utils import storage_to_dataframe

def lowPassFilter(time, data, lowpass_cutoff_frequency, order=4):
    
    fs = 1/np.round(np.mean(np.diff(time)),16)
    wn = lowpass_cutoff_frequency/(fs/2)
    sos = signal.butter(order/2, wn, btype='low', output='sos')
    dataFilt = signal.sosfiltfilt(sos, data, axis=0)

    return dataFilt

# %% Segment squats.
def segmentSquats(ikFilePath, pelvis_ty=None, timeVec=None, visualize=False,
                  filter_pelvis_ty=True, cutoff_frequency=4, height=.2):
    
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
def segmentSTS(ikFilePath, pelvis_ty=None, timeVec=None, velSeated=0.3,
               velStanding=0.15, visualize=False, filter_pelvis_ty=True, 
               cutoff_frequency=4, delay=0.1):
    
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
