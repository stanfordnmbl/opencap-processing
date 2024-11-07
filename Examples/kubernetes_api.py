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
runTorqueDrivenProblem = False
runMuscleDrivenProblem = True
runComparison = False

# %% Directories, paths, and imports. You should not need to change anything.
import os
import sys


baseDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(baseDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking
from utilsAuthentication import get_token
from utilsProcessing import segment_gait
from utils import get_trial_id, download_trial
from utilsKineticsOpenSimAD import kineticsOpenSimAD
from utilsPlotting import plot_dataframe
import glob
from tqdm import tqdm
import json
import time
import numpy as np 


import argparse

# %% Argument parsing
parser = argparse.ArgumentParser(description='Process some inputs for OpenSimAD.')
parser.add_argument('--subject', type=str, required=True, help='Subject identifier')
parser.add_argument('--mot-dir', type=str, required=True, help='Directory containing .mot files')
parser.add_argument('--segments', type=str, required=True, help='Segment file path')

args = parser.parse_args()

subject = args.subject
mot_dir = args.mot_dir
segments_file = args.segments



# %% OpenCap authentication. Visit https://app.opencap.ai/login to create an
# account if you don't have one yet.
get_token(saveEnvPath=os.getcwd())

# %% Inputs common to both examples.

# Insert your session ID here. You can find the ID of all your sessions at
# https://app.opencap.ai/sessions.
# Visit https://app.opencap.ai/session/<session_id> to visualize the data of
# your session.

# json_file = json.load(open("../Data/OpenSim/sqt_sessions.json"))

# mcs_files = ["349e4383-da38-4138-8371-9a5fed63a56a","015b7571-9f0b-4db4-a854-68e57640640d","c613945f-1570-4011-93a4-8c8c6408e2cf","dfda5c67-a512-4ca2-a4b3-6a7e22599732","7562e3c0-dea8-46f8-bc8b-ed9d0f002a77","275561c0-5d50-4675-9df1-733390cd572f","0e10a4e3-a93f-4b4d-9519-d9287d1d74eb","a5e5d4cd-524c-4905-af85-99678e1239c8","dd215900-9827-4ae6-a07d-543b8648b1da","3d1207bf-192b-486a-b509-d11ca90851d7","c28e768f-6e2b-4726-8919-c05b0af61e4a","fb6e8f87-a1cc-48b4-8217-4e8b160602bf","e6b10bbf-4e00-4ac0-aade-68bc1447de3e","d66330dc-7884-4915-9dbb-0520932294c4","0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45","2345d831-6038-412e-84a9-971bc04da597","0a959024-3371-478a-96da-bf17b1da15a9","ef656fe8-27e7-428a-84a9-deb868da053d","c08f1d89-c843-4878-8406-b6f9798a558e","d2020b0e-6d41-4759-87f0-5c158f6ab86a","8dc21218-8338-4fd4-8164-f6f122dc33d9"]
session_id = subject

if os.path.isfile(segments_file):
    segments_data = np.load(segments_file,allow_pickle=True).item()
    print("Segments data loaded:", segments_data)    
else: 
    print("No Segment found")    

for mot_file in tqdm(segments_data):
    for i, segment in enumerate(segments_data[mot_file]):
        # segment = segments_data[trial_name]
        prev_time = time.time()

        trial_name = f'{mot_file}_segment_{i+1}'
        print("Running on session:", session_id, "trial:",trial_name, "segment:", segment)
        
        try:
        
            folder_path = os.path.join(baseDir,'Data',session_id,'OpenSimData','Dynamics',trial_name)
            
            if os.path.exists(os.path.join(folder_path, 'optimaltrajectories.npy')):
                
                continue


            orig_path = os.path.join(mot_dir,mot_file + '.mot')
            new_path = os.path.join(baseDir,'Data',session_id,'OpenSimData','Kinematics',f'{trial_name}.mot')
            os.system(f"cp {orig_path} {new_path}")    
            
            print("Running command:", f"cp {orig_path} {new_path}")

            # session_id = "f78f774c-a705-4875-8e36-9c7184cc95ef" #"4d5c3eb1-1a59-4ea1-9178-d3634610561c"

            # # Insert the name of the trial you want to simulate.
            # trial_name = '' #'walk_1_25ms'

            # Insert the path to where you want the data to be downloaded.
            dataFolder = os.path.join(baseDir, 'Data')

            # Insert the type of activity you want to simulate. We have pre-defined settings
            # for different activities (more details above). Visit 
            # ./UtilsDynamicSimulations/OpenSimAD/settingsOpenSimAD.py to see all available
            # activities and their settings. If your activity is not in the list, select
            # 'other' to use generic settings or set your own settings.
            motion_type = 'squats' #'walking'

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
                time_window = [segment[0]/60, segment[1]/60] #[5.7333333, 6.9333333]
                treadmill_speed = 0 # 1.25
            print("Time window:", time_window)
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
            settings['meshDensity'] = 60 #50

            # Run the dynamic simulation.
            if runTorqueDrivenProblem:
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
            settings['meshDensity'] = 60 #50

            # Run the dynamic simulation.
            if runMuscleDrivenProblem:
                run_tracking(baseDir, dataFolder, session_id, settings, case=case)
                # Plot some results.
                plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings, [case])

                # Retrieve results from the optimal solution using utilsKineticsOpenSimAD.
                opt_sol_obj = kineticsOpenSimAD(dataFolder, session_id, trial_name, case)
                # Extract and plot muscle forces.
                muscle_forces = opt_sol_obj.get_muscle_forces()
                plot_dataframe(
                    dataframes = [muscle_forces],
                    xlabel = 'Time (s)',
                    ylabel = 'Force (N)',
                    title = 'Muscle forces',
                    labels = [trial_name])

            # %% Comparison torque-driven vs. muscle-driven model.
            if runComparison:
                plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings,
                                    ['torque_driven', 'muscle_driven'])

            end_time = time.time()
            
            print("Total time taken:", end_time-prev_time)

        except Exception as e:
            print(e)
            continue