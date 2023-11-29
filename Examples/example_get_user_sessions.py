import sys
sys.path.append("..")

from utils import get_user_sessions
from utilsAPI import get_api_url

API_URL = get_api_url()

my_users = {    
    
    }

my_sessions_all, my_sessions_valid, my_sessions_invalid = {}, {}, {}
for my_user in my_users:
    # my_sessions_valid[my_user], my_sessions_invalid[my_user] = [], [], []
    my_users[my_user]['sessions_valid'] = get_user_sessions(user_token=my_users[my_user]['token'])

    # for session in my_users[my_user]['sessions_all']:
    #     my_sessions_all[my_user].append(session['id'])

    # for session in my_users[my_user]['sessions_valid']:
    #     my_sessions_valid[my_user].append(session['id'])

    # for session in my_sessions_all[my_user]:
    #     if session not in my_sessions_valid[my_user]:
    #         my_sessions_invalid[my_user].append(session)

    
        
    

    
    
    
