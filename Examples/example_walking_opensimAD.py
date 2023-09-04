'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_walking_opensimAD.py
    ---------------------------------------------------------------------------
    Copyright 2023 Stanford University and the Authors
    
    Author(s): Antoine Falisse
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    
    This code makes use of CasADi, which is licensed under LGPL, Version 3.0;
    https://github.com/casadi/casadi/blob/master/LICENSE.txt.
    
    Install requirements:
        - Visit https://github.com/stanfordnmbl/opencap-processing for details.        
        - Third-party software packages:
            - Windows
                - Visual studio: https://visualstudio.microsoft.com/downloads/
                    - Make sure you install C++ support
                    - Code tested with community editions 2017-2019-2022
            - Linux
                - OpenBLAS libraries
                    - sudo apt-get install libopenblas-base
            
    Please contact us for any questions: https://www.opencap.ai/#contact

    This example shows how to run dynamic simulations of walking using
    data collected with OpenCap. The code uses OpenSimAD, a custom version
    of OpenSim that supports automatic differentiation (AD).

    This example is made of two sub-examples. The first one uses a torque-driven
    musculoskeletal model and the second one uses a muscle-driven
    musculoskeletal model. Please note that both examples are meant to
    demonstrate how to run dynamic simualtions and are not meant to be 
    biomechanically valid. We only made sure the simulations converged to
    solutions that were visually reasonable. You can find more examples of
    dynamic simulations using OpenSimAD in example_kinetics.py.
'''

# %% Select the example you want to run.
runTorqueDrivenProblem = True
runMuscleDrivenProblem = False
runComparison = False

# %% Directories, paths, and imports. You should not need to change anything.
import os
import sys
baseDir = os.path.join(os.getcwd(), '..')
sys.path.append(baseDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking
from utilsAuthentication import get_token
from utilsProcessing import segment_gait
from utils import get_trial_id, download_trial

# %% OpenCap authentication. Visit https://app.opencap.ai/login to create an
# account if you don't have one yet.
get_token(saveEnvPath=os.getcwd())

# %% Inputs common to both examples.

# Insert your session ID here. You can find the ID of all your sessions at
# https://app.opencap.ai/sessions.
# Visit https://app.opencap.ai/session/<session_id> to visualize the data of
# your session.
session_id = "4d5c3eb1-1a59-4ea1-9178-d3634610561c"

# Insert the name of the trial you want to simulate.
trial_name = 'walk_1_25ms'

# Insert the path to where you want the data to be downloaded.
dataFolder = os.path.join(baseDir, 'Data')

# Insert the type of activity you want to simulate. We have pre-defined settings
# for different activities (more details above). Visit 
# ./UtilsDynamicSimulations/OpenSimAD/settingsOpenSimAD.py to see all available
# activities and their settings. If your activity is not in the list, select
# 'other' to use generic settings or set your own settings.
motion_type = 'walking'

# Insert the time interval you want to simulate. It is recommended to simulate
# trials shorter than 2s (more details above). Set to [] to simulate full trial.
# We here selected a time window that corresponds to a full gait stride in order
# to use periodic constraints. You can use the gait segmentation function to
# automatically segment the gait cycle. Also insert the speed of the treadmill
# in m/s. A positive value indicates that the subject is moving forward. 
# You should ignore this parameter or set it to 0 if the trial was not measured
# on a treadmill. You can also use the gait segmenter to automatically extract
# the treadmill speed.
segmentation_method = 'manual'
if segmentation_method == 'automatic':
    # Download the trial.
    download_trial(get_trial_id(session_id,trial_name),
                   os.path.join(dataFolder,session_id),
                   session_id=session_id)    
    time_window, gaitObject = segment_gait(
        session_id, trial_name, dataFolder, gait_cycles_from_end=3)
    treadmill_speed = gaitObject.treadmillSpeed
else:
    time_window = [5.7333333, 6.9333333]
    treadmill_speed = 1.25
    
# %% Sub-example 1: walking simulation with torque-driven model.
# Insert a string to "name" you case.
case = 'torque_driven'

# Prepare inputs for dynamic simulation (this will be skipped if already done):
#   - Download data from OpenCap database
#   - Adjust wrapping surfaces
#   - Add foot-ground contacts
#   - Generate external function (OpenSimAD)
settings = processInputsOpenSimAD(
    baseDir, dataFolder, session_id, trial_name, motion_type, 
    time_window=time_window, treadmill_speed=treadmill_speed)

# Adjust settings for this example.
# Set the model to be torque-driven.
settings['torque_driven_model'] = True

# Adjust the weights of the objective function and remove the default
# muscle-related weigths. The objective function contains terms for tracking
# coordinate values (positionTrackingTerm), speeds (velocityTrackingTerm), and
# accelerations (accelerationTrackingTerm), as well as terms for minimizing
# excitations of the ideal torque actuators at the arms (armExcitationTerm),
# lumbar (lumbarExcitationTerm), and lower-extremity (coordinateExcitationTerm)
# joints. The objective function also contains a regularization term that
# minimizes the coordinate accelerations (jointAccelerationTerm).
settings['weights'] = {
    'positionTrackingTerm': 10,
    'velocityTrackingTerm': 10,
    'accelerationTrackingTerm': 50,
    'armExcitationTerm': 0.001,
    'lumbarExcitationTerm': 0.001,
    'coordinateExcitationTerm': 1,
    'jointAccelerationTerm': 0.001,}

# Add periodic constraints to the problem. This will constrain initial and
# final states of the problem to be the same. This is useful for obtaining
# faster convergence. Please note that the walking trial we selected might not
# be perfectly periodic. We here add periodic constraints to show how to do it.
# We here add periodic constraints for the coordinate values (coordinateValues)
# and coordinate speeds (coordinateSpeeds) of the lower-extremity joints
# (lowerLimbJoints). We also add periodic constraints for the activations of the
# ideal torque actuators at the lower-extremity (lowerLimbJointActivations) and
# lumbar (lumbarJointActivations) joints. 
settings['periodicConstraints'] = {
    'coordinateValues': ['lowerLimbJoints'],
    'coordinateSpeeds': ['lowerLimbJoints'],
    'lowerLimbJointActivations': ['all'],
    'lumbarJointActivations': ['all']}

# Filter the data to be tracked. We here filter the coordinate values (Qs) with
# a 6 Hz (cutoff_freq_Qs) low-pass filter, the coordinate speeds (Qds) with a 6
# Hz (cutoff_freq_Qds) low-pass filter, and the coordinate accelerations (Qdds)
# with a 6 Hz (cutoff_freq_Qdds) low-pass filter. We also compute the coordinate
# accelerations by first splining the coordinate speeds (splineQds=True) and
# then taking the first time derivative (default is to spline the coordinate
# values and then take the second time derivative).
settings['filter_Qs_toTrack'] = True
settings['cutoff_freq_Qs'] = 6
settings['filter_Qds_toTrack'] = True
settings['cutoff_freq_Qds'] = 6
settings['filter_Qdds_toTrack'] = True
settings['cutoff_freq_Qdds'] = 6
settings['splineQds'] = True

# We set the mesh density to 50. We recommend using a mesh density of 100 by
# default, but we here use a lower value to reduce the computation time.
settings['meshDensity'] = 50

# Run the dynamic simulation.
if runTorqueDrivenProblem:
    # Here are some reference numbers for convergence of the problem. Note that
    # it might vary depending on the machine you are using.
    #   - Windows (Windows 10):    converged in 99 iterations
    #   - macOS   (Monterey 12.2): converged in 110 iterations 
    #   - Linux   (Ubuntu 20.04):  converged in 109 iterations
    run_tracking(baseDir, dataFolder, session_id, settings, case=case)

    # Plot some results.
    plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings, [case])

# %% Sub-example 2: walking simulation with muscle-driven model.
# Insert a string to "name" you case.
case = 'muscle_driven'

# Prepare inputs for dynamic simulation (this will be skipped if already done):
#   - Download data from OpenCap database
#   - Adjust wrapping surfaces
#   - Add foot-ground contacts
#   - Generate external function (OpenSimAD)
settings = processInputsOpenSimAD(
    baseDir, dataFolder, session_id, trial_name, motion_type, 
    time_window=time_window, treadmill_speed=treadmill_speed)

# Add periodic constraints to the problem. This will constrain initial and
# final states of the problem to be the same. This is useful for obtaining
# faster convergence. Please note that the walking trial we selected might not
# be perfectly periodic. We here add periodic constraints to show how to do it.
# We here add periodic constraints for the coordinate values (coordinateValues)
# and coordinate speeds (coordinateSpeeds) of the lower-extremity joints
# (lowerLimbJoints). We also add periodic constraints for the activations and
# forces of all muscles acuating the lower-extremity (muscleActivationsForces)
# and for activations of the ideal torque actuators at the lumbar
# (lumbarJointActivations) joints. 
settings['periodicConstraints'] = {
    'coordinateValues': ['lowerLimbJoints'],
    'coordinateSpeeds': ['lowerLimbJoints'],
    'muscleActivationsForces': ['all'],
    'lumbarJointActivations': ['all']}

# Filter the data to be tracked. We here filter the coordinate values (Qs) with
# a 6 Hz (cutoff_freq_Qs) low-pass filter, the coordinate speeds (Qds) with a 6
# Hz (cutoff_freq_Qds) low-pass filter, and the coordinate accelerations (Qdds)
# with a 6 Hz (cutoff_freq_Qdds) low-pass filter. We also compute the coordinate
# accelerations by first splining the coordinate speeds (splineQds=True) and
# then taking the first time derivative (default is to spline the coordinate
# values and then take the second time derivative).
settings['filter_Qs_toTrack'] = True
settings['cutoff_freq_Qs'] = 6
settings['filter_Qds_toTrack'] = True
settings['cutoff_freq_Qds'] = 6
settings['filter_Qdds_toTrack'] = True
settings['cutoff_freq_Qdds'] = 6
settings['splineQds'] = True

# We set the mesh density to 50. We recommend using a mesh density of 100 by
# default, but we here use a lower value to reduce the computation time.
settings['meshDensity'] = 50

# Run the dynamic simulation.
if runMuscleDrivenProblem:
    # Here are some reference numbers for convergence of the problem. Note that
    # it might vary depending on the machine you are using.
    #   - Windows (Windows 10):    converged in 586 iterations
    #   - macOS   (Monterey 12.2): converged in 499 iterations
    #   - Linux   (Ubuntu 20.04):  converged in 645 iterations
    run_tracking(baseDir, dataFolder, session_id, settings, case=case)

    # Plot some results.
    plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings, [case])

# %% Comparison torque-driven vs. muscle-driven model.
if runComparison:
    plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings,
                        ['torque_driven', 'muscle_driven'])
