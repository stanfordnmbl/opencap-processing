'''
    ---------------------------------------------------------------------------
    OpenCap processing: example.py
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
import utilsKinematics
from utils import download_kinematics

# %% User inputs.
# Specify session id; see end of url in app.opencap.ai/session/<session_id>.
session_id = "6fc75175-7a66-4fe0-819d-c5f2ee8672dc"

# Specify trial names in a list; use None to process all trials in a session.
specific_trial_names = None

# Specify where to download the data.
data_folder = os.path.join("./Data", session_id)

# %% Download data.
trial_names = download_kinematics(session_id, folder=data_folder, trialNames=specific_trial_names)

# %% Process data.
kinematics, coordinates, muscle_tendon_lengths, moment_arms, center_of_mass = {}, {}, {}, {}, {}
coordinates['values'], coordinates['speeds'], coordinates['accelerations'] = {}, {}, {}
center_of_mass['values'], center_of_mass['speeds'], center_of_mass['accelerations'] = {}, {}, {}

for trial_name in trial_names:
    # Create object from class kinematics.
    kinematics[trial_name] = utilsKinematics.kinematics(data_folder, trial_name, lowpass_cutoff_frequency_for_coordinate_values=10)
    
    # Get coordinate values, speeds, and accelerations.
    coordinates['values'][trial_name] = kinematics[trial_name].get_coordinate_values(in_degrees=True) # already filtered
    coordinates['speeds'][trial_name] = kinematics[trial_name].get_coordinate_speeds(in_degrees=True, lowpass_cutoff_frequency=10)
    coordinates['accelerations'][trial_name] = kinematics[trial_name].get_coordinate_accelerations(in_degrees=True, lowpass_cutoff_frequency=10)
    
    # Get muscle-tendon lengths and moment arms.
    muscle_tendon_lengths[trial_name] = kinematics[trial_name].get_muscle_tendon_lengths()
    moment_arms[trial_name] = kinematics[trial_name].get_moment_arms()
    
    # Get center of mass values, speeds, and accelerations.
    center_of_mass['values'][trial_name] = kinematics[trial_name].get_center_of_mass_values(lowpass_cutoff_frequency=10)
    center_of_mass['speeds'][trial_name] = kinematics[trial_name].get_center_of_mass_speeds(lowpass_cutoff_frequency=10)
    center_of_mass['accelerations'][trial_name] = kinematics[trial_name].get_center_of_mass_accelerations(lowpass_cutoff_frequency=10)
    
    
# %% Print as csv: example.
output_csv_dir = os.path.join(data_folder, 'OpenSimData', 'Kinematics', 'Outputs')
os.makedirs(output_csv_dir, exist_ok=True)
output_csv_path = os.path.join(output_csv_dir, 'coordinate_speeds_{}.csv'.format(trial_names[0]))
coordinates['speeds'][trial_names[0]].to_csv(output_csv_path)
