import requests
from decouple import config
import getpass
from utilsAPI import get_api_url

API_URL = get_api_url()

def get_token():           
    if 'API_TOKEN' not in globals():    
        try: # look in environment file
            token = config("API_TOKEN")
        except:
            try:
                un = getpass.getpass(prompt='Username: ', stream=None)
                pw = getpass.getpass(prompt='Password: ', stream=None)                
                data = {"username":un,"password":pw}
                resp = requests.post(API_URL+' login/',data=data).json()
                token = resp['token']                
                print('Login successful.')                
            except:
                raise Exception('Login failed.')        
        global API_TOKEN
        API_TOKEN = token
    else:
        token = API_TOKEN    
    return token
