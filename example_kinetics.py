'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_kinetics.py
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
    
    This code make use of CasADi, which is licensed under LGPL, Version 3.0;
    https://github.com/casadi/casadi/blob/master/LICENSE.txt.
    
    Install requirements:
        CMake: https://cmake.org/download/
        Visual studio: https://visualstudio.microsoft.com/downloads/
            - Make sure you install C++ support
            - This code was tested with Visual Studio Community 2017-2019-2022
            
    Please contact us for any questions: https://www.opencap.ai/#contact
'''

# %% Directories, paths, and imports. You should not need to change anything.
import os
import sys

baseDir = os.getcwd()
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(baseDir)
sys.path.append(opensimADDir)

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsDC
from mainOpenSimAD import run_tracking

# %% User inputs.
'''
Please provide:
    
    session_id:     This is a 36 character-long string. You can find the ID of
                    all your sessions at https://app.opencap.ai/sessions.
                    
    trial_name:     This is the name of the trial you want to simulate. You can
                    find all trial names after loading a session.
                    
    motion_type:    This is the type of activity you want to simulate. Options
                    are 'running', 'walking', 'drop_jump', 'sit-to-stand', 
                    'squats', and 'other'. We provide pre-defined settings that
                    worked well for this set of activities (except for other).
                    If your activity is different, select 'other' to use
                    generic settings or set your own settings in
                    settingsOpenSimAD.
                    
    time_window:    This is the time interval you want to simulate. It is
                    recommended to simulate trials shorter than 2s. Set to []
                    to simulate full trial. For 'squats' or 'sit_to_stand', we
                    built segmenters to separate the different repetitions. In
                    such case, instead of providing the time_window, you can
                    provide the index of the repetition (see below) and the
                    time_window will be automatically computed.
                    
    repetition:     Only if motion_type is 'sit_to_stand' or 'squats'. This
                    is the index of the repetition you want to simulate (0 is 
                    first). There is no need to set the time_window. 
                    
    case:           This is a string that will be appended to the file names
                    of the results. Dynamic simulations are optimization
                    problems, and it is common to have to play with some
                    settings to get the problem to converge or converge to a
                    meaningful solution. It is useful to keep track of which
                    solution corresponsd to which settings; you can then easily
                    compare results generated with different settings.
                    
    (optional)
    treadmill_speed:This an optional parameter that indicates the speed of
                    the treadmill in m/s. A positive value indicates that the
                    subject is moving forward. You should ignore this parameter
                    or set it to 0 if the trial was not measured on a
                    treadmill. By default, treadmill_speed=0.
    
See example inputs below for different activities. Please note that we did not
verify the biomechanical validity of the results; we only made sure the
simulations converge to kinematic solutions that were visually reasonable.

Please contact us for any questions: https://www.opencap.ai/#contact
'''

# We provide a few examples for overground and treadmill activities.
# Select which example you would like to run.
session_type = 'overground' # Options are 'overground' and 'treadmill'.
case = '0'
# Predefined settings.
if session_type == 'overground':
    session_id = "ef516897-f0b2-493e-9927-3022db2d2ac3"
    trial_name = 'Squats' # Options are 'Gait', 'Squats', 'DJ', 'STS'.
    if trial_name == 'Gait': # Walking example
        motion_type = 'walking'
        time_window = [2.5, 4.0]
    elif trial_name == 'Squats': # Squat example
        motion_type = 'squats'
        repetition = 1
    elif trial_name == 'DJ': # Drop jump example
        motion_type = 'drop_jump'
        time_window = [1.55, 2.35]
    elif trial_name == 'STS': # Sit-to-stand example
        motion_type = 'sit_to_stand'
        repetition = 0
elif session_type == 'treadmill':
    session_id = "2174d76b-2646-4099-b11e-6ccdc96d82bf"
    trial_name = 'running_natural_backwards1'
    if trial_name == 'running_natural_backwards1': # Running example
        motion_type = 'running'
        time_window = [6.6, 7.2]
        treadmill_speed = 2.67
# Set to True to solve the optimal control problem.
solveProblem = True
# Set to True to analyze the results of the optimal control problem. If you
# solved the problem already, and only want to analyze/process the results, you
# can set solveProblem to False and run this script with analyzeResults set to
# True. This is useful if you do additional post-processing but do not need to
# re-run the problem.
analyzeResults = True

# %% Paths.
dataFolder =  os.path.join(baseDir, 'Data')

# %% Setup. 
if not 'time_window' in locals():
    time_window = None
if not 'repetition' in locals():
    repetition = None
if not 'treadmill_speed' in locals():
    treadmill_speed = 0
settings = processInputsOpenSimAD(baseDir, dataFolder, session_id, trial_name, 
                                  motion_type, time_window, repetition,
                                  treadmill_speed)

# %% Simulation.
run_tracking(baseDir, dataFolder, session_id, settings, case=case, 
             solveProblem=solveProblem, analyzeResults=analyzeResults)

# %% Plots.
plotResultsDC(dataFolder, session_id, trial_name, cases=[case], rep=repetition)
