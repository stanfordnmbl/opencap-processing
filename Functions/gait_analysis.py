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
import utils

def gait_analysis(trial_id,scalar_names=None,n_gait_cycles=1,filterFreq=6):
    
    session_id = utils.get_trial_json(trial_id)['session']
    
    # Local data dir -> will be deleted with lambda instance
    sessionDir = os.path.join(os.path.abspath('../Data'),session_id)
    
    # download data
    trialName = utils.download_trial(trial_id,sessionDir,session_id=session_id) 
    
    # init gait analysis
    gait_r = utilsKinematics.gait_analysis(sessionDir, trialName, leg='r',
                 lowpass_cutoff_frequency_for_coordinate_values=filterFreq,
                 n_gait_cycles=n_gait_cycles)
    gait_l = utilsKinematics.gait_analysis(sessionDir, trialName, leg='l',
                 lowpass_cutoff_frequency_for_coordinate_values=filterFreq,
                 n_gait_cycles=n_gait_cycles)
        
    # compute scalars and get time-normalized kinematics
    gaitResults = {}
    gaitResults['scalars_r'] = gait_r.compute_scalars(scalar_names)
    gaitResults['curves_r'] = gait_r.get_coordinates_normalized_time()
    gaitResults['scalars_l'] = gait_l.compute_scalars(scalar_names)
    gaitResults['curves_l'] = gait_l.get_coordinates_normalized_time()
    
    # TODO post results to server
    
    # TODO delete return once posted to server
    return gaitResults
    

# TODO delete. Testing as script

# overground trial
trial_id = 'bf181007-d0f3-4395-8dc3-a0f0e5553761'

# treadmill trial 1.25m/s
# session_id = "4d5c3eb1-1a59-4ea1-9178-d3634610561c"
# trial_name = 'walk_1_25ms'
# trial_id = utils.getTrialId(session_id,trial_name)

scalar_names = {'gait_speed','stride_length','step_width'}

gaitResults = gait_analysis(trial_id,scalar_names=scalar_names,
                                       n_gait_cycles=4)




