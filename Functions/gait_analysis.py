# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:23:12 2023

this function performs kinematic gait analysis

@authors: Scott Uhlrich & Antoine Falisse
"""

import os
import sys
sys.path.append("..")

import utilsKinematics
# import utils

def gait_analysis(trial_id,nGaitCycles=1):
    
    # session_id = utils.get_trial_json(trial_id)['session']
    # TODO DELETE
    trialName = 'walk'
    session_id = '03284efb-2244-4a48-aec9-abc34afdffc8'
    
    # Local data dir -> will be deleted with lambda instance
    dataDir = os.path.abspath(os.path.join('../Data',session_id))
    os.makedirs(dataDir,exist_ok=True)
    
    # download data
    # trialName = utils.download_trial(trial_id,dataDir,session_id=session_id)
    
    # init gait analysis
    gait = utilsKinematics.gait_analysis(dataDir, trialName, 
                 lowpass_cutoff_frequency_for_coordinate_values=-1,
                 n_gait_cycles=1)
    
    
    scalar_names = {'gait_speed','stride_length'}
            
    gaitScalars = gait.get_scalars(scalar_names)
    
    # post results
    
    # TODO temp
    return gaitScalars
    

# TODO delete. Testing as script
gaitScalars = gait_analysis('bf181007-d0f3-4395-8dc3-a0f0e5553761')

test = 1


