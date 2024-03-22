import os
import sys
sys.path.append("..")
import numpy as np
import json

from utils import get_trial_id, get_trial_json, download_file, post_video_to_trial, delete_video_from_trial

data_folder = os.path.join("../Data/Parker")
os.makedirs(data_folder, exist_ok=True)


# bad_data_trim_needed = \
# {'057d10da-34c7-4fb7-a127-6040010dde06': {'name': 'brooke_sit',
#                                           'session': '057d10da-34c7-4fb7-a127-6040010dde06'},
# #  '18f57fa8-41a0-4d8d-b7a3-d0c838516f24': {'name': 'brooke',
# #                                           'session': '18f57fa8-41a0-4d8d-b7a3-d0c838516f24'},
# #  '4c46fbb5-4ab8-49ff-b90d-5e8a38c5bcff': {'name': 'brooke',
# #                                           'session': '4c46fbb5-4ab8-49ff-b90d-5e8a38c5bcff'},
# #  '8bf259f7-000f-4474-9ad4-255949bdaf71': {'name': 'brooke',
# #                                           'session': '8bf259f7-000f-4474-9ad4-255949bdaf71'}
#                                           }

bad_data_trim_needed = \
{
  '54d4868a-9351-4844-b56c-c0c24bcb24e3': {'name': 'test',
                                           'start': 0.2, 'end': 1},
 # 'f5ec71e8-7898-4c51-993b-897014a3e8e3': {'name': '10mwrt',
 #                                          'start': 0, 'end': 4},
#  '18f57fa8-41a0-4d8d-b7a3-d0c838516f24': {'name': 'brooke',
#                                           'session': '18f57fa8-41a0-4d8d-b7a3-d0c838516f24'},
#  '4c46fbb5-4ab8-49ff-b90d-5e8a38c5bcff': {'name': 'brooke',
#                                           'session': '4c46fbb5-4ab8-49ff-b90d-5e8a38c5bcff'},
#  '8bf259f7-000f-4474-9ad4-255949bdaf71': {'name': 'brooke',
#                                           'session': '8bf259f7-000f-4474-9ad4-255949bdaf71'}
                                          }

def get_video_info(video_file):
    # Define the ffprobe command to get frame rate
    frame_rate_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 {video_file}'
    # Execute the ffprobe command to get frame rate
    frame_rate_output = os.popen(frame_rate_cmd).read().strip()

    # Define the ffprobe command to get number of frames
    frame_count_cmd = f'ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 {video_file}'
    # Execute the ffprobe command to get number of frames
    frame_count_output = os.popen(frame_count_cmd).read().strip()

    # Parse the output and convert to appropriate data types
    frame_rate = eval(frame_rate_output) if frame_rate_output else None
    frame_count = int(frame_count_output) if frame_count_output else None

    return frame_rate, frame_count
    
def trim_video_by_frames(video_path, start_frame, frame_duration, trimmed_video_path):
    # Build the ffmpeg command to trim without re-encoding
    ffmpeg_cmd = f'ffmpeg -i {video_path} -ss {start_frame / frame_rate} -frames:v {frame_duration} -c:v copy -c:a copy {trimmed_video_path}'
    
    os.system(ffmpeg_cmd)


for count, session_id in enumerate(bad_data_trim_needed):
    trial_name = bad_data_trim_needed[session_id]['name']
    trial_id = get_trial_id(session_id, trial_name)
    trial_json = get_trial_json(trial_id)
    for k, video in enumerate(trial_json["videos"]):

        # Download the video
        videoDir = os.path.join(data_folder, session_id, trial_id, "Videos", video['id'], "InputMedia")
        os.makedirs(videoDir, exist_ok=True)
        video_path = os.path.join(videoDir, trial_id + ".mov")
        download_file(video["video"], video_path)

        # Trim the video
        frame_rate, frame_count = get_video_info(video_path)
        video_duration = frame_count / frame_rate        
        n_frame_trim_start = int(np.floor(bad_data_trim_needed[session_id]['start'] * frame_rate))
        desired_num_frames = int(np.floor((bad_data_trim_needed[session_id]['end'] - bad_data_trim_needed[session_id]['start']) * frame_rate))
        trimmedVideoDir = os.path.join(data_folder, session_id, trial_id, "Videos", video['id'], "TrimmedInputMedia")
        os.makedirs(trimmedVideoDir, exist_ok=True)
        trimmed_video_path = os.path.join(trimmedVideoDir, trial_id + ".mov")
        trim_video_by_frames(video_path, n_frame_trim_start, desired_num_frames, trimmed_video_path)

        # Post the trimmed video
        post_video_to_trial(trimmed_video_path,trial_id,video['device_id'],json.dumps(video['parameters']))
        
        # Delete the old video
        delete_video_from_trial(video['id'])


# all_bad_data = \
# {'002ea0b9-a2b6-4983-a876-275e3087ef1e': {'name': 'brooke',
#                                           'session': '002ea0b9-a2b6-4983-a876-275e3087ef1e'},
#  '057d10da-34c7-4fb7-a127-6040010dde06': {'name': 'brooke_sit',
#                                           'session': '057d10da-34c7-4fb7-a127-6040010dde06'},
#  '07d80cd7-3634-4485-b3a7-a9d592cf82b0': {'name': 'Brooke',
#                                           'session': '07d80cd7-3634-4485-b3a7-a9d592cf82b0'},
#  '109eabdf-e50d-459e-8917-c856338a720b': {'name': '10mwrt',
#                                           'session': '109eabdf-e50d-459e-8917-c856338a720b'},
#  '18f57fa8-41a0-4d8d-b7a3-d0c838516f24': {'name': 'brooke',
#                                           'session': '18f57fa8-41a0-4d8d-b7a3-d0c838516f24'},
#  '19788965-c4a4-481e-a687-f2055b6d85cb': {'name': 'stairs_descend',
#                                           'session': '19788965-c4a4-481e-a687-f2055b6d85cb'},
#  '1fc22b0c-1ef8-4b6d-b83b-85b12634905d': {'name': '10mwrt_1',
#                                           'session': '1fc22b0c-1ef8-4b6d-b83b-85b12634905d'},
#  '22fab7ce-b219-4c5d-84a4-e780816d46ea': {'name': 'tug_line_1',
#                                           'session': '22fab7ce-b219-4c5d-84a4-e780816d46ea'},
#  '25d785ae-f49e-4531-b14c-86d6d5e9d144': {'name': 'toe_stand',
#                                           'session': '25d785ae-f49e-4531-b14c-86d6d5e9d144'},
#  '25ebd0a3-e8ce-4a7a-ad91-0c1d94753b69': {'name': 'jump',
#                                           'session': '25ebd0a3-e8ce-4a7a-ad91-0c1d94753b69'},
#  '28ec73c6-025b-4ea4-b844-803e8205bf1a': {'name': 'brooke',
#                                           'session': '28ec73c6-025b-4ea4-b844-803e8205bf1a'},
#  '2aff7769-ee45-4e3d-a8cc-266d5b886a10': {'name': 'toe_stand',
#                                           'session': '2aff7769-ee45-4e3d-a8cc-266d5b886a10'},
#  '2f6340cd-d4ad-4ef1-b8c3-d1296821c78f': {'name': 'tug_cone',
#                                           'session': '2f6340cd-d4ad-4ef1-b8c3-d1296821c78f'},
#  '316ac7bf-12a8-4b1b-84db-56a9a7f9d70b': {'name': '5xsts',
#                                           'session': '316ac7bf-12a8-4b1b-84db-56a9a7f9d70b'},
#  '31b45d11-f909-42e5-b404-3451b4b6d238': {'name': 'TUG',
#                                           'session': '31b45d11-f909-42e5-b404-3451b4b6d238'},
#  '34f2095f-d076-4a3d-b094-75f968a93c21': {'name': 'jump',
#                                           'session': '34f2095f-d076-4a3d-b094-75f968a93c21'},
#  '38a56b81-3e2c-4854-93a4-333b66c8e0f1': {'name': 'toe_stand',
#                                           'session': '38a56b81-3e2c-4854-93a4-333b66c8e0f1'},
#  '3a681925-50b5-470e-a44a-c1cab743f58b': {'name': '10mwt',
#                                           'session': '3a681925-50b5-470e-a44a-c1cab743f58b'},
#  '41599e7a-8eea-4b03-b5ef-57502504e879': {'name': 'toe_stand',
#                                           'session': '41599e7a-8eea-4b03-b5ef-57502504e879'},
#  '43db734e-13fb-4d6d-9223-09f967dd40f0': {'name': 'toe_stand',
#                                           'session': '43db734e-13fb-4d6d-9223-09f967dd40f0'},
#  '4769f868-d487-4e0d-bda6-98eb6751b40a': {'name': 'jump',
#                                           'session': '4769f868-d487-4e0d-bda6-98eb6751b40a'},
#  '4c46fbb5-4ab8-49ff-b90d-5e8a38c5bcff': {'name': 'brooke',
#                                           'session': '4c46fbb5-4ab8-49ff-b90d-5e8a38c5bcff'},
#  '4d0cfa49-baa9-4e49-8b34-2bc6727f6052': {'name': '10mwt',
#                                           'session': '4d0cfa49-baa9-4e49-8b34-2bc6727f6052'},
#  '51003106-cdec-40f7-9204-0e21953bb4a7': {'name': '10mwt',
#                                           'session': '51003106-cdec-40f7-9204-0e21953bb4a7'},
#  '55566ced-e1be-4789-b265-b0168087a402': {'name': '10mwrt_1',
#                                           'session': '55566ced-e1be-4789-b265-b0168087a402'},
#  '59f3fc4e-674f-43cc-9fdc-e9a2cf0a109e': {'name': 'tug_line_1',
#                                           'session': '59f3fc4e-674f-43cc-9fdc-e9a2cf0a109e'},
#  '5a3dd4e7-7293-46e6-9c0c-b77e2511860d': {'name': '10mwrt',
#                                           'session': '5a3dd4e7-7293-46e6-9c0c-b77e2511860d'},
#  '621e5ae8-226b-4fc6-bbd9-df448328ff1f': {'name': 'jump_1',
#                                           'session': '621e5ae8-226b-4fc6-bbd9-df448328ff1f'},
#  '623a4fe2-bc1e-4248-920a-e35416d1350d': {'name': '10mwrt',
#                                           'session': '623a4fe2-bc1e-4248-920a-e35416d1350d'},
#  '6537ecc8-e44b-4405-a9fe-c09ede5e0550': {'name': '5xsts',
#                                           'session': '6537ecc8-e44b-4405-a9fe-c09ede5e0550'},
#  '65aa878e-4396-4a9c-a9a0-7a1e2c708f83': {'name': 'ARM_ROM',
#                                           'session': '65aa878e-4396-4a9c-a9a0-7a1e2c708f83'},
#  '68cf5eec-1d9b-4630-8585-c9094d2de82c': {'name': 'TUGCone',
#                                           'session': '68cf5eec-1d9b-4630-8585-c9094d2de82c'},
#  '6de88f3d-b84a-4ee5-8003-a5fbb994278e': {'name': '10mwt',
#                                           'session': '6de88f3d-b84a-4ee5-8003-a5fbb994278e'},
#  '715504d6-c823-4632-8c79-b7639bfdcf4a': {'name': 'tug_cone',
#                                           'session': '715504d6-c823-4632-8c79-b7639bfdcf4a'},
#  '725e0d28-d043-44bd-9f3e-0e58338a6bc4': {'name': '10mwrt',
#                                           'session': '725e0d28-d043-44bd-9f3e-0e58338a6bc4'},
#  '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86': {'name': 'TUGfast',
#                                           'session': '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86'},
#  '76b201f8-1950-414e-a48a-97bf932c61bc': {'name': '10mwt',
#                                           'session': '76b201f8-1950-414e-a48a-97bf932c61bc'},
#  '773c33cd-a12d-46d2-af17-35fa7b4e83bd': {'name': 'tug_cone',
#                                           'session': '773c33cd-a12d-46d2-af17-35fa7b4e83bd'},
#  '78d9fbfe-04e0-4766-ba15-198e246d5e9c': {'name': '10mwt',
#                                           'session': '78d9fbfe-04e0-4766-ba15-198e246d5e9c'},
#  '7d9b3d14-2672-458a-94df-d35c7b5cdcf5': {'name': '5xsts',
#                                           'session': '7d9b3d14-2672-458a-94df-d35c7b5cdcf5'},
#  '822926eb-1c7d-4298-84c5-7ead6909387b': {'name': 'tug_cone_2',
#                                           'session': '822926eb-1c7d-4298-84c5-7ead6909387b'},
#  '8963b080-d316-4183-8928-47bebcac70b1': {'name': '10mwt',
#                                           'session': '8963b080-d316-4183-8928-47bebcac70b1'},
#  '8b7a734d-2874-42f7-94a2-c6026ec35d99': {'name': '10mwrt',
#                                           'session': '8b7a734d-2874-42f7-94a2-c6026ec35d99'},
#  '8bf259f7-000f-4474-9ad4-255949bdaf71': {'name': 'brooke',
#                                           'session': '8bf259f7-000f-4474-9ad4-255949bdaf71'},
#  '8d2f7856-3996-462f-bb2b-f020e7053a90': {'name': 'Brooke',
#                                           'session': '8d2f7856-3996-462f-bb2b-f020e7053a90'},
#  '90e3af8d-3aac-4d97-ac5a-70f5c4e911ae': {'name': 'tug_cone',
#                                           'session': '90e3af8d-3aac-4d97-ac5a-70f5c4e911ae'},
#  '9108903d-26b3-479a-ad49-955aebb868c7': {'name': 'TUGfast',
#                                           'session': '9108903d-26b3-479a-ad49-955aebb868c7'},
#  '9808e3d7-b2a1-4864-a3c7-7e6549a5dd36': {'name': '10mwt',
#                                           'session': '9808e3d7-b2a1-4864-a3c7-7e6549a5dd36'},
#  '9871c398-0d02-450b-86a6-0d8c6b27b26d': {'name': '5xsts',
#                                           'session': '9871c398-0d02-450b-86a6-0d8c6b27b26d'},
#  '98fc0617-1f5e-44ba-92c9-a32f09899295': {'name': 'brooke',
#                                           'session': '98fc0617-1f5e-44ba-92c9-a32f09899295'},
#  '9c4666be-57f4-4a6b-9c51-7e394f4a53e1': {'name': 'tug_line',
#                                           'session': '9c4666be-57f4-4a6b-9c51-7e394f4a53e1'},
#  '9ce91ceb-2bf8-4544-ae22-3da5ad3cb9a1': {'name': '10mwrt',
#                                           'session': '9ce91ceb-2bf8-4544-ae22-3da5ad3cb9a1'},
#  '9e5d7c34-d9fc-43f7-85e9-d3e5def7ef3a': {'name': '10mrt',
#                                           'session': '9e5d7c34-d9fc-43f7-85e9-d3e5def7ef3a'},
#  'a08ec9d6-24f8-44f7-a59c-f603b7517e4d': {'name': 'tug_cone',
#                                           'session': 'a08ec9d6-24f8-44f7-a59c-f603b7517e4d'},
#  'a0d8b67a-89b0-45a7-a634-348ef2bcca5a': {'name': '10mwt',
#                                           'session': 'a0d8b67a-89b0-45a7-a634-348ef2bcca5a'},
#  'a4679c46-2f38-4cc5-86b4-bdd3f01791ab': {'name': 'brooke',
#                                           'session': 'a4679c46-2f38-4cc5-86b4-bdd3f01791ab'},
#  'a7675b05-88da-4bb0-b27b-5357c3ec5807': {'name': 'tug_cone',
#                                           'session': 'a7675b05-88da-4bb0-b27b-5357c3ec5807'},
#  'aca92056-3f67-403c-8d6a-513055274ffe': {'name': '5xsts',
#                                           'session': 'aca92056-3f67-403c-8d6a-513055274ffe'},
#  'ace248c9-61a6-4c41-93eb-83edd823de0c': {'name': 'toe_stand',
#                                           'session': 'ace248c9-61a6-4c41-93eb-83edd823de0c'},
#  'b233300d-2c6d-4735-aadb-c597da014629': {'name': 'ELBOWROM',
#                                           'session': 'b233300d-2c6d-4735-aadb-c597da014629'},
#  'ba7d94c8-dccb-486d-90d5-b3dcc237bfce': {'name': '10mwrt',
#                                           'session': 'ba7d94c8-dccb-486d-90d5-b3dcc237bfce'},
#  'bb12568e-2303-4f87-95e4-baec67f2c023': {'name': 'tug_cone',
#                                           'session': 'bb12568e-2303-4f87-95e4-baec67f2c023'},
#  'bc4dd6dc-d9d2-4345-b130-d42bb1c4d673': {'name': '5TSTS',
#                                           'session': 'bc4dd6dc-d9d2-4345-b130-d42bb1c4d673'},
#  'bca0aad8-c129-4a62-bef3-b5de1659df5e': {'name': 'tug_cone',
#                                           'session': 'bca0aad8-c129-4a62-bef3-b5de1659df5e'},
#  'c2001341-2624-4595-91d1-9af78ed76421': {'name': '10mwt_1',
#                                           'session': 'c2001341-2624-4595-91d1-9af78ed76421'},
#  'ca174c27-d0fe-4edd-8799-9ef3ea053086': {'name': 'JUMP',
#                                           'session': 'ca174c27-d0fe-4edd-8799-9ef3ea053086'},
#  'ca44b787-96c1-4ed3-8151-89b4bd4310c4': {'name': 'toe_stand',
#                                           'session': 'ca44b787-96c1-4ed3-8151-89b4bd4310c4'},
#  'cbb0e8bc-8ed0-4af5-a20d-0a54650a3375': {'name': '5TSTS',
#                                           'session': 'cbb0e8bc-8ed0-4af5-a20d-0a54650a3375'},
#  'cda2db6e-b268-42ee-99d0-9cc358e893d1': {'name': '10mwrt',
#                                           'session': 'cda2db6e-b268-42ee-99d0-9cc358e893d1'},
#  'cdb103d7-e1a7-4f67-bf74-e335f79b471c': {'name': '10MWT_2',
#                                           'session': 'cdb103d7-e1a7-4f67-bf74-e335f79b471c'},
#  'ce0fdd88-53b6-4974-92fb-a720b93d1623': {'name': '5xsts',
#                                           'session': 'ce0fdd88-53b6-4974-92fb-a720b93d1623'},
#  'd10ea979-2a29-4bd7-a3ac-d475b5878134': {'name': 'toe_stand',
#                                           'session': 'd10ea979-2a29-4bd7-a3ac-d475b5878134'},
#  'd37407d6-83a8-44f8-b5b2-c651db8edd9f': {'name': 'jump',
#                                           'session': 'd37407d6-83a8-44f8-b5b2-c651db8edd9f'},
#  'd9538893-e95c-4e28-960e-dd4aa0760942': {'name': 'tug_line',
#                                           'session': 'd9538893-e95c-4e28-960e-dd4aa0760942'},
#  'dac69d8a-0790-48ea-9296-66b5903d9ba8': {'name': '10mwrt_2',
#                                           'session': 'dac69d8a-0790-48ea-9296-66b5903d9ba8'},
#  'dc497821-4eac-49e0-87ac-c088d1245edb': {'name': '10mwt_1',
#                                           'session': 'dc497821-4eac-49e0-87ac-c088d1245edb'},
#  'dcfddd1b-39f9-4d0e-80c7-b68caa612a1e': {'name': 'curls',
#                                           'session': 'dcfddd1b-39f9-4d0e-80c7-b68caa612a1e'},
#  'de9b2159-afd0-4e11-8259-8f7d4e2e8ca5': {'name': '10MWRT',
#                                           'session': 'de9b2159-afd0-4e11-8259-8f7d4e2e8ca5'},
#  'df982665-8e59-4691-88f8-611ccbfe7f4a': {'name': 'brooke',
#                                           'session': 'df982665-8e59-4691-88f8-611ccbfe7f4a'},
#  'e4d2328b-6f01-4f4e-bea3-64076357de64': {'name': '10mwrt',
#                                           'session': 'e4d2328b-6f01-4f4e-bea3-64076357de64'},
#  'e91690d8-1a0d-4031-a7ad-3fb1e23d7c36': {'name': '10mwt',
#                                           'session': 'e91690d8-1a0d-4031-a7ad-3fb1e23d7c36'},
#  'ead8a2fc-c926-464f-9965-d155a62eead8': {'name': 'curls',
#                                           'session': 'ead8a2fc-c926-464f-9965-d155a62eead8'},
#  'ee86be93-b1da-4080-b445-176f6071e734': {'name': '10mwt_2',
#                                           'session': 'ee86be93-b1da-4080-b445-176f6071e734'},
#  'f249a8d6-9965-4240-94c3-e42a6b4bccf6': {'name': 'brooke',
#                                           'session': 'f249a8d6-9965-4240-94c3-e42a6b4bccf6'},
#  'f5e4cd75-e3c2-40ee-9596-ae767fa63e85': {'name': 'jump',
#                                           'session': 'f5e4cd75-e3c2-40ee-9596-ae767fa63e85'},
#  'f5ec71e8-7898-4c51-993b-897014a3e8e3': {'name': '10mwrt',
#                                           'session': 'f5ec71e8-7898-4c51-993b-897014a3e8e3'},
#  'fa856382-bf8d-411a-b086-a2d74fab9d1b': {'name': '10mrt',
#                                           'session': 'fa856382-bf8d-411a-b086-a2d74fab9d1b'},
#  'faa79e6d-f90e-4399-8b68-9e9f2dd00027': {'name': 'HOP',
#                                           'session': 'faa79e6d-f90e-4399-8b68-9e9f2dd00027'}}



# bad_data_server_failed = \
# {'31b45d11-f909-42e5-b404-3451b4b6d238': {'name': 'TUG',
#                                           'session': '31b45d11-f909-42e5-b404-3451b4b6d238'},
#  '41599e7a-8eea-4b03-b5ef-57502504e879': {'name': 'toe_stand',
#                                           'session': '41599e7a-8eea-4b03-b5ef-57502504e879'},
#  '4769f868-d487-4e0d-bda6-98eb6751b40a': {'name': 'jump',
#                                           'session': '4769f868-d487-4e0d-bda6-98eb6751b40a'},
#  '621e5ae8-226b-4fc6-bbd9-df448328ff1f': {'name': 'jump_1',
#                                           'session': '621e5ae8-226b-4fc6-bbd9-df448328ff1f'},
#  '6537ecc8-e44b-4405-a9fe-c09ede5e0550': {'name': '5xsts',
#                                           'session': '6537ecc8-e44b-4405-a9fe-c09ede5e0550'},
#  '76b201f8-1950-414e-a48a-97bf932c61bc': {'name': '10mwt',
#                                           'session': '76b201f8-1950-414e-a48a-97bf932c61bc'},
#  '8d2f7856-3996-462f-bb2b-f020e7053a90': {'name': 'Brooke',
#                                           'session': '8d2f7856-3996-462f-bb2b-f020e7053a90'},
#  '90e3af8d-3aac-4d97-ac5a-70f5c4e911ae': {'name': '10mwt',
#                                           'session': '90e3af8d-3aac-4d97-ac5a-70f5c4e911ae'},
#  '9108903d-26b3-479a-ad49-955aebb868c7': {'name': 'Curl',
#                                           'session': '9108903d-26b3-479a-ad49-955aebb868c7'},
#  '9ce91ceb-2bf8-4544-ae22-3da5ad3cb9a1': {'name': '10mwrt',
#                                           'session': '9ce91ceb-2bf8-4544-ae22-3da5ad3cb9a1'},
#  'b233300d-2c6d-4735-aadb-c597da014629': {'name': 'ELBOWROM',
#                                           'session': 'b233300d-2c6d-4735-aadb-c597da014629'},
#  'c2001341-2624-4595-91d1-9af78ed76421': {'name': '10mwt_1',
#                                           'session': 'c2001341-2624-4595-91d1-9af78ed76421'},
#  'ca174c27-d0fe-4edd-8799-9ef3ea053086': {'name': 'JUMP',
#                                           'session': 'ca174c27-d0fe-4edd-8799-9ef3ea053086'},
#  'cda2db6e-b268-42ee-99d0-9cc358e893d1': {'name': '10mwrt',
#                                           'session': 'cda2db6e-b268-42ee-99d0-9cc358e893d1'},
#  'e4d2328b-6f01-4f4e-bea3-64076357de64': {'name': '10mwrt',
#                                           'session': 'e4d2328b-6f01-4f4e-bea3-64076357de64'},
#  'ee86be93-b1da-4080-b445-176f6071e734': {'name': '10mwt_2',
#                                           'session': 'ee86be93-b1da-4080-b445-176f6071e734'},
#  'f5e4cd75-e3c2-40ee-9596-ae767fa63e85': {'name': 'jump',
#                                           'session': 'f5e4cd75-e3c2-40ee-9596-ae767fa63e85'},
#  'f5ec71e8-7898-4c51-993b-897014a3e8e3': {'name': '10mwrt',
#                                           'session': 'f5ec71e8-7898-4c51-993b-897014a3e8e3'}}

