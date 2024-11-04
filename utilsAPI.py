'''
    ---------------------------------------------------------------------------
    OpenCap processing: utilsAPI.py
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

from decouple import config

def get_api_url():
    if 'API_URL' not in globals():
        global API_URL
        try: # look in environment file
            API_URL = config("API_URL")
        except: # default
            API_URL = "https://api.opencap.ai/"    
    if API_URL[-1] != '/':
        API_URL= API_URL + '/'

    return API_URL
