'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_get_sessions_subject.py
    ---------------------------------------------------------------------------
    Copyright 2024 Stanford University and the Authors
    
    Author(s): Antoine Falisse
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
                
    Please contact us for any questions: https://www.opencap.ai/#contact

    This example shows how to retrieve all session IDs from a subject, given the
    subject's name. The example assumes that you have already authenticated
    with OpenCap. If you haven't, please run createAuthenticationEnvFile.py
    first.

'''
import sys
sys.path.append("../")

from utils import get_user_subjects, get_subject_sessions

# Insert the name of the subject you are interested in.
subject_name = 'my_subject_name'

# Get list with all your subjects.
subjects = get_user_subjects()

# Get subject IDs. There could be multiple subjects with the same name.
subject_ids = [subject['id'] for subject in subjects if subject['name'] == subject_name]
print ("We found {} subjects with the name {}".format(len(subject_ids), subject_name))

# Get session IDs from subject(s).
session_ids = []
for subject_id in subject_ids:
    session_ids.append(get_subject_sessions(subject_id))
