# -*- coding: utf-8 -*-
"""
Created on Wed May  4 22:26:46 2022

@author: suhlr

This example downloads data from the OpenCap server and plots center of
mass trajectory and estimated vertical ground reaction force.

Model
This uses the OpenSim model described by Rajagopal et al. 2016. 

Data
The included data was collected by the OpenCap team and has been made publicly
available with the permission of the volunteer. It can be viewed at:
https://app.opencap.ai/session/572543ce-2a73-4750-96b6-c3dedcbe11b7

Processing
Using the kinematics computed by OpenCap, we use the OpenSim API to compute 
the center of mass position, velocity, and acceleration.

Requirements
You need an opencap account to run this demo. If you do not have an account yet,
you can create one at app.opencap.ai. You need an account both to record data
and to use this tutorial with the publicly available data.
"""

#%% Imports

import sys
import os
sys.path.append(os.path.abspath('./..'))

import numpy as np
import matplotlib.pyplot as plt
import utilsProcessing as up


#%% User inputs

# Enter session and trial names

# session id comes from end of url in app.opencap.ai/session/[session_id]
session_id = "6fc75175-7a66-4fe0-819d-c5f2ee8672dc"
# specify trial names in a list, otherwise use None to process all trials in a session
specificTrialNames = None 

# Example session
# session_id = "572543ce-2a73-4750-96b6-c3dedcbe11b7"
# specificTrialNames = ['jump_littleKneeFlexion','jump_moreKneeFlexion','jump_countermovement'] 

dataFolder =  os.path.join("./Data",session_id)


#%% Processing

# Download kinematics
specificTrialNames = up.downloadKinematics(session_id, folder=dataFolder, 
                                            trialNames=specificTrialNames)

# Compute center of mass position, velocity, acceleration
COMtrajectories = up.calcCenterOfMassTrajectory(dataFolder,coordinateFilterFreq=6,
                                                 COMFilterFreq=6,trialNames=specificTrialNames)


# %% Plot center of mass position 
# TODO make a nice plotting utility

def getCol(trialDict,colName):
    return trialDict['data'][:,trialDict['fields'].index(colName)]

   
try:
    plt.close(plt.figure(1))
except:
    pass

SMALL_SIZE=8
MEDIUM_SIZE=10
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=6)    # legend fontsize
    

fig1 = plt.figure(1,figsize=(6,6),dpi=300)
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)
for trial in COMtrajectories:
    idxPeakVel = np.argmax(getCol(trial,'vel_y'))
    x = getCol(trial,'time')
    x = x-x[idxPeakVel]
    y = getCol(trial,'pos_y')
    y = y-y[0]
    ax1.plot(x,y,label=trial['name'])

    
    x = getCol(trial,'time')
    y = getCol(trial,'vel_y')
    x = x-x[idxPeakVel]
    ax2.plot(x,y,label=trial['name'])
    

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Center of Mass Position (m)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Center of Mass Velocity (m/s)')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

box = ax1.get_position()
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          fancybox=True, ncol=5)



# %% Forces from accelerations. F=ma
 
fig2 = plt.figure(2,figsize=(6,4),dpi=300)
gravity = 9.81

for trial in COMtrajectories:
    idxPeakVel = np.argmax(getCol(trial,'vel_y'))
    x = getCol(trial,'time')
    x = x-x[idxPeakVel]
    COM_acc_y = getCol(trial,'acc_y')
    f_y = (COM_acc_y+gravity) / gravity #vGRF exp. in bodyweights: vGRF = m(a+g)/mg
    plt.plot(x,f_y,label=trial['name'])


ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Time (s)')
plt.ylabel('Vertical Ground Reaction Force (bodyweight)')

box = ax1.get_position()
ax.legend(loc='uper center', bbox_to_anchor=(0.5, 1.2),
          fancybox=True, ncol=5)
plt.tight_layout()
