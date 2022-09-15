'''
    ---------------------------------------------------------------------------
    OpenCap processing: utilsAuthentication.py
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

import requests
from decouple import config
import getpass
import os
import maskpass
from utilsAPI import get_api_url

API_URL = get_api_url()

def get_token():
           
    if 'API_TOKEN' not in globals():
    
        try: # look in environment file
            token = config("API_TOKEN")              
            
        except:
            try:
                # If spyder, use maskpass
                isSpyder = 'SPY_PYTHONPATH' in os.environ
                
                if isSpyder:
                    un = maskpass.advpass(prompt="Enter Username\n")
                    pw = maskpass.advpass(prompt="Enter Password\n")
                else:
                    un = getpass.getpass(prompt='Username: ', stream=None)
                    pw = getpass.getpass(prompt='Password: ', stream=None)
                
                data = {"username":un,"password":pw}
                resp = requests.post(API_URL + 'login/',data=data).json()
                token = resp['token']
                
                print('Login successful.')
                
            except:
                raise Exception('Login failed.')
        
        global API_TOKEN
        API_TOKEN = token
    else:
        token = API_TOKEN
    
    return token
