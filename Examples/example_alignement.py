import os
import sys
baseDir = os.path.join(os.getcwd(), '..')
sys.path.append(baseDir)

from utils import download_trial, get_trial_id
from utilsProcessing import align_markers_with_ground
from utilsOpenSim import runIKTool

session_id = 'a08ec9d6-24f8-44f7-a59c-f603b7517e4d'
trial_name = '10mwrt_2'

dataFolder = os.path.join(baseDir, 'Data')

# Get trial id from name.
trial_id = get_trial_id(session_id,trial_name)    

# Set session path.
sessionDir = os.path.join(dataFolder, session_id)

# Download data.
trialName, modelName = download_trial(trial_id,sessionDir,session_id=session_id)

# Align markers with ground.
pathTRCFile_out = align_markers_with_ground(
    sessionDir, trialName, 
    referenceMarker1='r.PSIS_study', referenceMarker2='L.PSIS_study',
    suffixOutputFileName='aligned',
    lowpass_cutoff_frequency_for_marker_values=6)

# Run inverse kinematics.
pathGenericSetupFile = os.path.join(
    baseDir, 'OpenSimPipeline', 
    'InverseKinematics', 'Setup_InverseKinematics.xml')

pathScaledModel = os.path.join(sessionDir, 'OpenSimData', 'Model', modelName)
pathOutputFolder = os.path.join(sessionDir, 'OpenSimData', 'Kinematics')

runIKTool(pathGenericSetupFile, pathScaledModel, pathTRCFile_out,
          pathOutputFolder)
