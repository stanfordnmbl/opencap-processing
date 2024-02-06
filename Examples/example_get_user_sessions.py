import sys
sys.path.append("..")

from utils import get_user_sessions, get_user_sessions_all
from utilsAPI import get_api_url

API_URL = get_api_url()

my_users = {   
    }

my_sessions_all, my_sessions_valid, my_sessions_invalid = {}, {}, {}
for my_user in my_users:
    my_sessions_all[my_user], my_sessions_valid[my_user], my_sessions_invalid[my_user] = [], [], []
    my_users[my_user]['sessions_all'] = get_user_sessions_all(user_token=my_users[my_user]['token'])
    # Contains sessions not belonging to use (public sessions)
    to_remove = []
    for count, session in enumerate(my_users[my_user]['sessions_all']):
        if not session['user'] == my_users[my_user]['user']:
            to_remove.append(count)
    for count, index in enumerate(to_remove):
        del my_users[my_user]['sessions_all'][index-count]
    my_users[my_user]['sessions_valid'] = get_user_sessions(user_token=my_users[my_user]['token'])

    for session in my_users[my_user]['sessions_all']:
        my_sessions_all[my_user].append(session['id'])

    for session in my_users[my_user]['sessions_valid']:
        my_sessions_valid[my_user].append(session['id'])

    for session in my_sessions_all[my_user]:
        if session not in my_sessions_valid[my_user]:
            my_sessions_invalid[my_user].append(session)

    
        
    

    
    
    