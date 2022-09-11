'''
Install requirements:
    Third-party software:
        CMake: https://cmake.org/download/
        Visual studio: https://visualstudio.microsoft.com/downloads/
            - Make sure you install C++ support
            - This code was tested with Visual Studio Community 2017
        CasADi: https://web.casadi.org/get/
            - You can just 'pip install casadi' with Anaconda
            
Please contact us for any questions: https://www.opencap.ai/#contact
'''

# %% Directories, paths, and imports. You should not need to change anything.
import os
baseDir = os.getcwd()
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
# jrDir = os.path.join(baseDir, 'opensimPipeline', 'JointReaction')

import sys
sys.path.append(baseDir)
sys.path.append(opensimADDir)
# sys.path.append(jrDir)

from utils import download_kinematics, storage_to_numpy, import_metadata
from utilsProcessing import segmentSquats, segmentSTS, adjustMuscleWrapping, generateModelWithContacts
from utilsOpenSimAD import generateExternalFunction, plotResultsDC
from settingsOpenSimAD import get_default_setup, get_trial_setup
from mainOpenSimAD import run_tracking

# %% User inputs.
'''
Please provide:
    session_id:     This is a 36 character-long string. You can find the ID of
                    all your sessions at https://app.opencap.ai/sessions.
    trial_name:     This is the name of the trial you want to simulate. You can
                    find all trial names after loading a session.
    motion_type:    Options are 'running', 'walking', 'drop_jump', 'sts', 
                    'squats', and 'other'. This is the type of activity you 
                    want to simulate. We provide settings/formulations that 
                    worked well for different types of activities. We have 
                    pre-defined settings for running, walking, drop_jump, sts, 
                    and squats. If your activity is different, please select
                    'other'.
    time_window:    This is the time interval you want to simulate. It is
                    recommended to simulate trials shorter than 2s. Set to []
                    to simulate full trial. For squats or sit-to-stands, we
                    built segmenters to separate the different repetitions. In
                    such case, instead of providing the time_window, you can
                    provide the index of the repetition (0 is first) and the
                    time_window will be automatically computed.
    repetition:     Only for motion_type='sts' and motion_type='squats'. This
                    is the index of the repetition you want to simulate.
    case:           This is a string that will be appended to the file names
                    of the results. Dynamic simulations are optimization
                    problems, and it is common to have to play with some
                    settings to get the problem to converge or converge to a
                    meaningful solution. It is useful to keep track of what
                    solutions correspond to what settings and this is what this
                    variable is doing. You can then easily compare results
                    generated from different sets of settings.
    compiler:       This is the name of the compiler/build system generator
                    you want to use. Our framework involves building/compiling
                    libraries and you will need to have a compiler for that.
                    On windows, we use Visual Studio Generators. You can find
                    a list of generators here: https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html.
                    TODO: Win64 or x64. 
                    With Visual Studio 2017 64 bits, the generator name is:
                    'Visual Studio 15 2017 Win64'.
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

# Examples.
###############################################################################
# Overground session - uncomment below to run.
# session_id = "ef516897-f0b2-493e-9927-3022db2d2ac3"
# trial_name = 'Gait' # Examples include 'Gait', 'Squats', 'DJ', 'STS'.
# if trial_name == 'Gait': # Walking example
#     motion_type = 'walking'
#     time_window = [2.5, 4.0]
# elif trial_name == 'Squats': # Squat example
#     motion_type = 'squats'
#     repetition = 1
# elif trial_name == 'DJ': # Drop jump example
#     motion_type = 'drop_jump'
#     time_window = [1.55, 2.35]
# elif trial_name == 'STS': # Sit-to-stand example
#     motion_type = 'sts'
#     repetition = 0
###############################################################################
# Treadmill session - uncomment below to run.
session_id = "2174d76b-2646-4099-b11e-6ccdc96d82bf"
trial_name = 'running_natural_backwards1'
if trial_name == 'running_natural_backwards1': # Running example
    motion_type = 'running'
    time_window = [6.6, 7.2]
    treadmill_speed = 2.67
###############################################################################

case = '0'

# Set compiler/build system generator.
compiler = "Visual Studio 15 2017 Win64"

# Set to True to solve the optimal control problem.
solveProblem = True
# Set to True to analyze the results of the optimal control problem. If you
# solved the problem already, and only want to analyze/process the results, you
# can set solveProblem to False and run this script with analyzeResults set to
# True.
analyzeResults = True

# %% Paths.
dataFolder =  os.path.join(baseDir, 'Data')
sessionFolder =  os.path.join(dataFolder, session_id)

# %% Setup.
# Processing inputs.
if not 'treadmill_speed' in locals():
    treadmill_speed = 0
    treadmill = False
else:
    treadmill = True
# Download kinematics and model.
_ = download_kinematics(session_id, folder=sessionFolder, trialNames=[trial_name])

# Prepare inputs for dynamic simulations.
# Adjust muscle wrapping.
adjustMuscleWrapping(baseDir, dataFolder, session_id, overwrite=False)
# Add foot-ground contacts to musculoskeletal model.
generateModelWithContacts(dataFolder, session_id, overwrite=False)
# Generate external function.
generateExternalFunction(baseDir, dataFolder, session_id, compiler=compiler,
                         overwrite=False, treadmill=treadmill)

# Get settings.
default_settings = get_default_setup(motion_type)
settings = get_trial_setup(default_settings, motion_type, trial_name)
# # Add time to settings if not specified.
pathMotionFile = os.path.join(sessionFolder, 'OpenSimData', 'Kinematics',
                              trial_name + '.mot')
if not 'repetition' in locals():
    repetition = None
if not 'time_window' in locals(): 
    if motion_type == 'squats':
        times_window = segmentSquats(pathMotionFile, visualize=True)
    elif motion_type == 'sts':
        _, _, times_window = segmentSTS(pathMotionFile, visualize=True)
    time_window = times_window[repetition]
else:
    if not time_window:
        motion_file = storage_to_numpy(pathMotionFile)
        time_window = [motion_file['time'][0], motion_file['time'][-1]]    
settings['trials'][trial_name]['timeInterval'] = time_window

# # Get demographics.
metadata = import_metadata(os.path.join(sessionFolder, 'sessionMetadata.yaml'))
demographics = {}
demographics['mass_kg'] = metadata['mass_kg']
demographics['height_m'] = metadata['height_m']

# %% Simulation.
run_tracking(baseDir, dataFolder, session_id, settings, demographics,
              case=case, solveProblem=solveProblem,
              analyzeResults=analyzeResults, rep=repetition, 
              treadmill_speed=treadmill_speed)

# %% Plots.
plotResultsDC(dataFolder, session_id, trial_name, cases=[case], rep=repetition)
