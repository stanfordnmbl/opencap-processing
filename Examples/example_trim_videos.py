import os
import sys
sys.path.append("..")
import numpy as np
import json
import shutil
import traceback

from utils import get_trial_id, get_trial_json, download_file, post_video_to_trial, delete_video_from_trial, delete_results, set_trial_status

data_folder = os.path.join("../Data/Parker")
os.makedirs(data_folder, exist_ok=True)

# Test
# all_bad_data = \
# {'19feaf9b-67ae-4652-bcdd-b42a447c3d31_000': {'session': '19feaf9b-67ae-4652-bcdd-b42a447c3d31',
#   'name': 'test',
#   'start': 0,
#   'end': 2}, # nothing obvious
# }

sessions_to_exclude = [
    '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86_001',
    '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86_002',
    '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86_003',
    '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86_004',
    '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86_005',
    '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86_006',
    '9108903d-26b3-479a-ad49-955aebb868c7_068',
    '8d2f7856-3996-462f-bb2b-f020e7053a90_070',
    'ca174c27-d0fe-4edd-8799-9ef3ea053086_071',
    'ca174c27-d0fe-4edd-8799-9ef3ea053086_072',
    'ca174c27-d0fe-4edd-8799-9ef3ea053086_073',
    '9ce91ceb-2bf8-4544-ae22-3da5ad3cb9a1_087'

]

# subject cannot perform activity: 25ebd0a3-e8ce-4a7a-ad91-0c1d94753b69_113

all_bad_data = \
{'dcfddd1b-39f9-4d0e-80c7-b68caa612a1e_000': {'session': 'dcfddd1b-39f9-4d0e-80c7-b68caa612a1e',
  'name': 'arm_rom',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86_001': {'session': '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86',
  'name': '10mrt',
  'start': np.nan,
  'end': np.nan}, # cut off FOV in first few seconds of one side
 '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86_002': {'session': '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86',
  'name': '10mwt',
  'start': 7,
  'end': np.nan}, # TO REPROCESS
 '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86_003': {'session': '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86',
  'name': '5xSTS',
  'start': np.nan,
  'end': np.nan}, # both videos cut off right before 6th rep is finished
 '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86_004': {'session': '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86',
  'name': 'Curl',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86_005': {'session': '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86',
  'name': 'SHLD_ROM',
  'start': np.nan,
  'end': np.nan}, # person in the background of one frame 
 '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86_006': {'name': 'TUGfast',
  'session': '74cd2ec6-de3c-4a0e-a8c9-44edbffe0b86',
  'start': np.nan,
  'end': np.nan}, # coming very close to the end
 'dac69d8a-0790-48ea-9296-66b5903d9ba8_007': {'session': 'dac69d8a-0790-48ea-9296-66b5903d9ba8',
  'name': '10mwt',
  'start': 0,
  'end': 15}, # TO REPROCESS
 'd9538893-e95c-4e28-960e-dd4aa0760942_008': {'session': 'd9538893-e95c-4e28-960e-dd4aa0760942',
  'name': 'brooke',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 'd9538893-e95c-4e28-960e-dd4aa0760942_009': {'session': 'd9538893-e95c-4e28-960e-dd4aa0760942',
  'name': 'tug_cone',
  'start': 3,
  'end': np.nan}, # TO REPROCESS
 'd9538893-e95c-4e28-960e-dd4aa0760942_010': {'session': 'd9538893-e95c-4e28-960e-dd4aa0760942',
  'name': '10mwt',
  'start': np.nan,
  'end': np.nan},
 'd9538893-e95c-4e28-960e-dd4aa0760942_011': {'session': 'd9538893-e95c-4e28-960e-dd4aa0760942',
  'name': '10mwrt',
  'start': np.nan,
  'end': 7}, # TO REPROCESS
 'd9538893-e95c-4e28-960e-dd4aa0760942_012': {'name': 'tug_line',
  'session': 'd9538893-e95c-4e28-960e-dd4aa0760942',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '8963b080-d316-4183-8928-47bebcac70b1_013': {'name': '10mwt',
  'session': '8963b080-d316-4183-8928-47bebcac70b1',
  'start': 0,
  'end': 9}, # TO REPROCESS, not sure it will help
 'b233300d-2c6d-4735-aadb-c597da014629_014': {'session': 'b233300d-2c6d-4735-aadb-c597da014629',
  'name': '10MRT',
  'start': 7,
  'end': np.nan}, # # TO REPROCESS
 'fa856382-bf8d-411a-b086-a2d74fab9d1b_015': {'name': '10mrt',
  'session': 'fa856382-bf8d-411a-b086-a2d74fab9d1b',
  'start': np.nan,
  'end': 4}, # TO REPROCESS, person in background
 '9ce91ceb-2bf8-4544-ae22-3da5ad3cb9a1_016': {'session': '9ce91ceb-2bf8-4544-ae22-3da5ad3cb9a1',
  'name': '10mwrt_1',
  'start': np.nan,
  'end': np.nan}, # VIDEOS DIFFERENT LENGTHS, one side is 1 second truncated
 '9c4666be-57f4-4a6b-9c51-7e394f4a53e1_017': {'name': 'tug_line',
  'session': '9c4666be-57f4-4a6b-9c51-7e394f4a53e1',
  'start': 2,
  'end': 10}, # TO REPROCESS
 'f249a8d6-9965-4240-94c3-e42a6b4bccf6_018': {'name': 'brooke',
  'session': 'f249a8d6-9965-4240-94c3-e42a6b4bccf6',
  'start': np.nan,
  'end': np.nan}, # person in background
 '43db734e-13fb-4d6d-9223-09f967dd40f0_019': {'name': 'toe_stand',
  'session': '43db734e-13fb-4d6d-9223-09f967dd40f0',
  'start': np.nan,
  'end': np.nan}, # nothing obvious, v different lengths
 '8b7a734d-2874-42f7-94a2-c6026ec35d99_020': {'name': '10mwrt',
  'session': '8b7a734d-2874-42f7-94a2-c6026ec35d99',
  'start': 3,
  'end': np.nan}, # fast run
 'ead8a2fc-c926-464f-9965-d155a62eead8_021': {'name': 'curls',
  'session': 'ead8a2fc-c926-464f-9965-d155a62eead8',
  'start': 14,
  'end': np.nan}, # TO REPROCESS [VIDEOS DIFFERENT LENGTHS]
 'dc497821-4eac-49e0-87ac-c088d1245edb_022': {'name': '10mwt_1',
  'session': 'dc497821-4eac-49e0-87ac-c088d1245edb',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 'a08ec9d6-24f8-44f7-a59c-f603b7517e4d_023': {'session': 'a08ec9d6-24f8-44f7-a59c-f603b7517e4d',
  'name': '10mwrt_2',
  'start': np.nan,
  'end': np.nan}, # nothing obvious, some glare from overhead lights
 '38a56b81-3e2c-4854-93a4-333b66c8e0f1_024': {'name': 'toe_stand',
  'session': '38a56b81-3e2c-4854-93a4-333b66c8e0f1',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '9108903d-26b3-479a-ad49-955aebb868c7_025': {'session': '9108903d-26b3-479a-ad49-955aebb868c7',
  'name': '5TSTS',
  'start': 10,
  'end': 27}, # TO REPROCESS
 '002ea0b9-a2b6-4983-a876-275e3087ef1e_026': {'name': 'brooke',
  'session': '002ea0b9-a2b6-4983-a876-275e3087ef1e',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 'b233300d-2c6d-4735-aadb-c597da014629_027': {'session': 'b233300d-2c6d-4735-aadb-c597da014629',
  'name': 'TUG',
  'start': np.nan,
  'end': np.nan}, # exiting FOV
 '68cf5eec-1d9b-4630-8585-c9094d2de82c_028': {'name': 'TUGCone',
  'session': '68cf5eec-1d9b-4630-8585-c9094d2de82c',
  'start': np.nan,
  'end': np.nan}, # leaving the fov
 'e91690d8-1a0d-4031-a7ad-3fb1e23d7c36_029': {'session': 'e91690d8-1a0d-4031-a7ad-3fb1e23d7c36',
  'name': '10mwrt',
  'start': np.nan,
  'end': 6}, # TO REPROCESS
 '3a681925-50b5-470e-a44a-c1cab743f58b_030': {'name': '10mwt',
  'session': '3a681925-50b5-470e-a44a-c1cab743f58b',
  'start': np.nan,
  'end': 9}, # nothing obvious, quality seems bad (clinic)
 '773c33cd-a12d-46d2-af17-35fa7b4e83bd_031': {'session': '773c33cd-a12d-46d2-af17-35fa7b4e83bd',
  'name': '10mrt',
  'start': np.nan,
  'end': np.nan}, # person in the background
 '725e0d28-d043-44bd-9f3e-0e58338a6bc4_032': {'name': '10mwrt',
  'session': '725e0d28-d043-44bd-9f3e-0e58338a6bc4',
  'start': 0,
  'end': 10}, # TO REPROCESS
 'ace248c9-61a6-4c41-93eb-83edd823de0c_033': {'name': 'toe_stand',
  'session': 'ace248c9-61a6-4c41-93eb-83edd823de0c',
  'start': np.nan,
  'end': 38}, # TO REPROCESS
 'cdb103d7-e1a7-4f67-bf74-e335f79b471c_034': {'name': '10MWT_2',
  'session': 'cdb103d7-e1a7-4f67-bf74-e335f79b471c',
  'start': np.nan,
  'end': 13}, # TO REPROCESS
 'cbb0e8bc-8ed0-4af5-a20d-0a54650a3375_035': {'name': '5TSTS',
  'session': 'cbb0e8bc-8ed0-4af5-a20d-0a54650a3375',
  'start': np.nan,
  'end': np.nan}, # person in background
 '59f3fc4e-674f-43cc-9fdc-e9a2cf0a109e_036': {'name': 'tug_line_1',
  'session': '59f3fc4e-674f-43cc-9fdc-e9a2cf0a109e',
  'start': np.nan,
  'end': np.nan}, # nothing obvious, quality seems bad (clinic)
 'ce0fdd88-53b6-4974-92fb-a720b93d1623_037': {'name': '5xsts',
  'session': 'ce0fdd88-53b6-4974-92fb-a720b93d1623',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 'a0d8b67a-89b0-45a7-a634-348ef2bcca5a_038': {'name': '10mwt',
  'session': 'a0d8b67a-89b0-45a7-a634-348ef2bcca5a',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '623a4fe2-bc1e-4248-920a-e35416d1350d_039': {'name': '10mwrt',
  'session': '623a4fe2-bc1e-4248-920a-e35416d1350d',
  'start': np.nan,
  'end': np.nan}, # fast run
 'd10ea979-2a29-4bd7-a3ac-d475b5878134_040': {'session': 'd10ea979-2a29-4bd7-a3ac-d475b5878134',
  'name': 'jump',
  'start': 0.1,
  'end': np.nan}, # TODO REPROCESS
 'dcfddd1b-39f9-4d0e-80c7-b68caa612a1e_041': {'name': 'curls',
  'session': 'dcfddd1b-39f9-4d0e-80c7-b68caa612a1e',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '9808e3d7-b2a1-4864-a3c7-7e6549a5dd36_042': {'name': '10mwt',
  'session': '9808e3d7-b2a1-4864-a3c7-7e6549a5dd36',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 'ba7d94c8-dccb-486d-90d5-b3dcc237bfce_043': {'name': '10mwrt',
  'session': 'ba7d94c8-dccb-486d-90d5-b3dcc237bfce',
  'start': 3,
  'end': 10}, # TO REPROCESS
 '90e3af8d-3aac-4d97-ac5a-70f5c4e911ae_044': {'session': '90e3af8d-3aac-4d97-ac5a-70f5c4e911ae',
  'name': 'brooke',
  'start': np.nan,
  'end': np.nan}, # video appears to start mid-task
 'dac69d8a-0790-48ea-9296-66b5903d9ba8_045': {'name': '10mwrt_2',
  'session': 'dac69d8a-0790-48ea-9296-66b5903d9ba8',
  'start': np.nan,
  'end': 13}, # TO REPROCESS
 'b233300d-2c6d-4735-aadb-c597da014629_046': {'session': 'b233300d-2c6d-4735-aadb-c597da014629',
  'name': 'SHLDRAROM',
  'start': np.nan,
  'end': np.nan}, # arms exit FOV
 'd10ea979-2a29-4bd7-a3ac-d475b5878134_047': {'session': 'd10ea979-2a29-4bd7-a3ac-d475b5878134',
  'name': 'arm_rom',
  'start': np.nan,
  'end': np.nan}, # hands almost exit fov
 'd10ea979-2a29-4bd7-a3ac-d475b5878134_048': {'session': 'd10ea979-2a29-4bd7-a3ac-d475b5878134',
  'name': 'curls',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 'd10ea979-2a29-4bd7-a3ac-d475b5878134_049': {'name': 'toe_stand',
  'session': 'd10ea979-2a29-4bd7-a3ac-d475b5878134',
  'start': np.nan,
  'end': np.nan}, # nothing obvious 
 '19788965-c4a4-481e-a687-f2055b6d85cb_050': {'session': '19788965-c4a4-481e-a687-f2055b6d85cb',
  'name': 'stairs_ascend',
  'start': np.nan,
  'end': np.nan}, # participant turns around after the task ends
 'de9b2159-afd0-4e11-8259-8f7d4e2e8ca5_051': {'session': 'de9b2159-afd0-4e11-8259-8f7d4e2e8ca5',
  'name': 'TUG_LINE',
  'start': np.nan,
  'end': np.nan}, # exits FOV and does not return to chair
 '22fab7ce-b219-4c5d-84a4-e780816d46ea_052': {'name': 'tug_line_1',
  'session': '22fab7ce-b219-4c5d-84a4-e780816d46ea',
  'start': np.nan,
  'end': np.nan}, # exiting the fov
 '8d2f7856-3996-462f-bb2b-f020e7053a90_053': {'session': '8d2f7856-3996-462f-bb2b-f020e7053a90',
  'name': '10MWT',
  'start': np.nan,
  'end': np.nan}, # nothing obvious, a bit blurry in the beginning of one frame
 '316ac7bf-12a8-4b1b-84db-56a9a7f9d70b_054': {'name': '5xsts',
  'session': '316ac7bf-12a8-4b1b-84db-56a9a7f9d70b',
  'start': np.nan,
  'end': np.nan}, # nothing obvious, quality seems bad (clinic)
 '8d2f7856-3996-462f-bb2b-f020e7053a90_055': {'session': '8d2f7856-3996-462f-bb2b-f020e7053a90',
  'name': '10MRT',
  'start': np.nan,
  'end': np.nan}, # nothing obvious, a decent amount of glare from the overhead lights
 '19788965-c4a4-481e-a687-f2055b6d85cb_056': {'name': 'stairs_descend',
  'session': '19788965-c4a4-481e-a687-f2055b6d85cb',
  'start': np.nan,
  'end': np.nan}, # ignore
 'a4679c46-2f38-4cc5-86b4-bdd3f01791ab_057': {'name': 'brooke',
  'session': 'a4679c46-2f38-4cc5-86b4-bdd3f01791ab',
  'start': np.nan,
  'end': np.nan}, # cuts off the beginning of the task, he appears to do the task twice
 'df982665-8e59-4691-88f8-611ccbfe7f4a_058': {'name': 'brooke',
  'session': 'df982665-8e59-4691-88f8-611ccbfe7f4a',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '9871c398-0d02-450b-86a6-0d8c6b27b26d_059': {'name': '5xsts',
  'session': '9871c398-0d02-450b-86a6-0d8c6b27b26d',
  'start': np.nan,
  'end': np.nan}, # VIDEOS DIFFERENT LENGTHS
 '2aff7769-ee45-4e3d-a8cc-266d5b886a10_060': {'name': 'toe_stand',
  'session': '2aff7769-ee45-4e3d-a8cc-266d5b886a10',
  'start': np.nan,
  'end': np.nan}, # nothing obvious except wearing a dress
 '057d10da-34c7-4fb7-a127-6040010dde06_061': {'name': 'brooke_sit',
  'session': '057d10da-34c7-4fb7-a127-6040010dde06',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '8bf259f7-000f-4474-9ad4-255949bdaf71_062': {'name': 'brooke',
  'session': '8bf259f7-000f-4474-9ad4-255949bdaf71',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '18f57fa8-41a0-4d8d-b7a3-d0c838516f24_063': {'name': 'brooke',
  'session': '18f57fa8-41a0-4d8d-b7a3-d0c838516f24',
  'start': np.nan,
  'end': np.nan}, # hands hidden
 '4c46fbb5-4ab8-49ff-b90d-5e8a38c5bcff_064': {'name': 'brooke',
  'session': '4c46fbb5-4ab8-49ff-b90d-5e8a38c5bcff',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '55566ced-e1be-4789-b265-b0168087a402_065': {'name': '10mwrt_1',
  'session': '55566ced-e1be-4789-b265-b0168087a402',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '822926eb-1c7d-4298-84c5-7ead6909387b_066': {'session': '822926eb-1c7d-4298-84c5-7ead6909387b',
  'name': 'tug_cone_1',
  'start': 4,
  'end': np.nan}, # TO REPROCESS, people in the background
 '822926eb-1c7d-4298-84c5-7ead6909387b_067': {'name': 'tug_cone_2',
  'session': '822926eb-1c7d-4298-84c5-7ead6909387b',
  'start': 9,
  'end': np.nan}, # TO REPROCESS, people in the background
 '9108903d-26b3-479a-ad49-955aebb868c7_068': {'session': '9108903d-26b3-479a-ad49-955aebb868c7',
  'name': 'Curl',
  'start': np.nan,
  'end': np.nan}, # folder not on drive
 '8d2f7856-3996-462f-bb2b-f020e7053a90_069': {'session': '8d2f7856-3996-462f-bb2b-f020e7053a90',
  'name': 'TUGCone',
  'start': np.nan,
  'end': np.nan}, # exiting the fov in one video
 '8d2f7856-3996-462f-bb2b-f020e7053a90_070': {'name': 'Brooke',
  'session': '8d2f7856-3996-462f-bb2b-f020e7053a90',
  'start': np.nan,
  'end': np.nan}, # missing video
 'ca174c27-d0fe-4edd-8799-9ef3ea053086_071': {'session': 'ca174c27-d0fe-4edd-8799-9ef3ea053086',
  'name': 'TIPTOE',
  'start': np.nan,
  'end': np.nan}, # [videos missing?]
 'ca174c27-d0fe-4edd-8799-9ef3ea053086_072': {'session': 'ca174c27-d0fe-4edd-8799-9ef3ea053086',
  'name': 'TIPTOEV2',
  'start': np.nan,
  'end': np.nan}, # only one video; exiting fov
 'ca174c27-d0fe-4edd-8799-9ef3ea053086_073': {'name': 'JUMP',
  'session': 'ca174c27-d0fe-4edd-8799-9ef3ea053086',
  'start': np.nan,
  'end': np.nan}, # only one video
 '31b45d11-f909-42e5-b404-3451b4b6d238_074': {'name': 'TUG',
  'session': '31b45d11-f909-42e5-b404-3451b4b6d238',
  'start': np.nan,
  'end': np.nan}, # Leaving the fov
 'b233300d-2c6d-4735-aadb-c597da014629_075': {'name': 'ELBOWROM',
  'session': 'b233300d-2c6d-4735-aadb-c597da014629',
  'start': 1.5,
  'end': np.nan}, # TO REPROCESS person in background
 'c2001341-2624-4595-91d1-9af78ed76421_076': {'name': '10mwt_1',
  'session': 'c2001341-2624-4595-91d1-9af78ed76421',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 'f5ec71e8-7898-4c51-993b-897014a3e8e3_077': {'name': '10mwrt',
  'session': 'f5ec71e8-7898-4c51-993b-897014a3e8e3',
  'start': np.nan,
  'end': 6}, # TO REPROCESS
 'ee86be93-b1da-4080-b445-176f6071e734_078': {'session': 'ee86be93-b1da-4080-b445-176f6071e734',
  'name': '10mwrt',
  'start': np.nan,
  'end': 7}, # TO REPROCESS
 '76b201f8-1950-414e-a48a-97bf932c61bc_079': {'name': '10mwt',
  'session': '76b201f8-1950-414e-a48a-97bf932c61bc',
  'start': 0,
  'end': 10}, # TO REPROCESS
 '90e3af8d-3aac-4d97-ac5a-70f5c4e911ae_080': {'session': '90e3af8d-3aac-4d97-ac5a-70f5c4e911ae',
  'name': '10mwt',
  'start': np.nan,
  'end': 10}, # VIDEOS DIFFERENT LENGTHS; one side has 12 extra seconds on the backend
 '4769f868-d487-4e0d-bda6-98eb6751b40a_081': {'session': '4769f868-d487-4e0d-bda6-98eb6751b40a',
  'name': 'toe_stand',
  'start': np.nan,
  'end': np.nan}, # VIDEOS DIFFERENT LENGTHS, one side is 4 seconds truncated
 '4769f868-d487-4e0d-bda6-98eb6751b40a_082': {'name': 'jump',
  'session': '4769f868-d487-4e0d-bda6-98eb6751b40a',
  'start': np.nan,
  'end': np.nan}, # 98eb6751b40a_082
 '41599e7a-8eea-4b03-b5ef-57502504e879_083': {'name': 'toe_stand',
  'session': '41599e7a-8eea-4b03-b5ef-57502504e879',
  'start': 0,
  'end': 33}, # TO REPROCESS
 'e4d2328b-6f01-4f4e-bea3-64076357de64_084': {'name': '10mwrt',
  'session': 'e4d2328b-6f01-4f4e-bea3-64076357de64',
  'start': 4,
  'end': 10}, # TO REPROCESS
 'f5e4cd75-e3c2-40ee-9596-ae767fa63e85_085': {'name': 'jump',
  'session': 'f5e4cd75-e3c2-40ee-9596-ae767fa63e85',
  'start': 3,
  'end': 9}, # TO REPROCESS
 'ee86be93-b1da-4080-b445-176f6071e734_086': {'name': '10mwt_2',
  'session': 'ee86be93-b1da-4080-b445-176f6071e734',
  'start': np.nan,
  'end': 10}, # TO REPROCESS
 '9ce91ceb-2bf8-4544-ae22-3da5ad3cb9a1_087': {'session': '9ce91ceb-2bf8-4544-ae22-3da5ad3cb9a1',
  'name': '10mwrt',
  'start': np.nan,
  'end': np.nan}, # folder not on drive
 '6537ecc8-e44b-4405-a9fe-c09ede5e0550_088': {'name': '5xsts',
  'session': '6537ecc8-e44b-4405-a9fe-c09ede5e0550',
  'start': 0,
  'end': 15}, # TO REPROCESS
 'cda2db6e-b268-42ee-99d0-9cc358e893d1_089': {'name': '10mwrt',
  'session': 'cda2db6e-b268-42ee-99d0-9cc358e893d1',
  'start': 5,
  'end': np.nan}, # TO REPROCESS REDO
 '621e5ae8-226b-4fc6-bbd9-df448328ff1f_090': {'name': 'jump_1',
  'session': '621e5ae8-226b-4fc6-bbd9-df448328ff1f',
  'start': np.nan,
  'end': np.nan}, # no jump recorded
  '6de88f3d-b84a-4ee5-8003-a5fbb994278e_091': {'name': '10mwt',
  'session': '6de88f3d-b84a-4ee5-8003-a5fbb994278e',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 'd37407d6-83a8-44f8-b5b2-c651db8edd9f_092': {'name': 'jump',
  'session': 'd37407d6-83a8-44f8-b5b2-c651db8edd9f',
  'start': np.nan,
  'end': np.nan}, # hands almost exit fov
 '4d0cfa49-baa9-4e49-8b34-2bc6727f6052_093': {'name': '10mwt',
  'session': '4d0cfa49-baa9-4e49-8b34-2bc6727f6052',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 'e91690d8-1a0d-4031-a7ad-3fb1e23d7c36_094': {'name': '10mwt',
  'session': 'e91690d8-1a0d-4031-a7ad-3fb1e23d7c36',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '5a3dd4e7-7293-46e6-9c0c-b77e2511860d_095': {'name': '10mwrt',
  'session': '5a3dd4e7-7293-46e6-9c0c-b77e2511860d',
  'start': np.nan,
  'end': np.nan}, # bad synchronization
 'bc4dd6dc-d9d2-4345-b130-d42bb1c4d673_096': {'session': 'bc4dd6dc-d9d2-4345-b130-d42bb1c4d673',
  'name': '10MRT',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 'bc4dd6dc-d9d2-4345-b130-d42bb1c4d673_097': {'session': 'bc4dd6dc-d9d2-4345-b130-d42bb1c4d673',
  'name': '10MWT',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 'de9b2159-afd0-4e11-8259-8f7d4e2e8ca5_098': {'name': '10MWRT',
  'session': 'de9b2159-afd0-4e11-8259-8f7d4e2e8ca5',
  'start': np.nan,
  'end': np.nan}, # person in background
 'bc4dd6dc-d9d2-4345-b130-d42bb1c4d673_099': {'session': 'bc4dd6dc-d9d2-4345-b130-d42bb1c4d673',
  'name': 'TugLineRun',
  'start': np.nan,
  'end': np.nan}, # exits FOV
 '07d80cd7-3634-4485-b3a7-a9d592cf82b0_100': {'name': 'Brooke',
  'session': '07d80cd7-3634-4485-b3a7-a9d592cf82b0',
  'start': np.nan,
  'end': np.nan}, # nothing obvious, quality seems bad (clinic)
 'faa79e6d-f90e-4399-8b68-9e9f2dd00027_101': {'name': 'HOP',
  'session': 'faa79e6d-f90e-4399-8b68-9e9f2dd00027',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '34f2095f-d076-4a3d-b094-75f968a93c21_102': {'name': 'jump',
  'session': '34f2095f-d076-4a3d-b094-75f968a93c21',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 'ca44b787-96c1-4ed3-8151-89b4bd4310c4_103': {'name': 'toe_stand',
  'session': 'ca44b787-96c1-4ed3-8151-89b4bd4310c4',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '2f6340cd-d4ad-4ef1-b8c3-d1296821c78f_104': {'name': 'tug_cone',
  'session': '2f6340cd-d4ad-4ef1-b8c3-d1296821c78f',
  'start': np.nan,
  'end': np.nan}, # nothing obvious, maybe Tina in the fov
 'a08ec9d6-24f8-44f7-a59c-f603b7517e4d_105': {'name': 'tug_cone',
  'session': 'a08ec9d6-24f8-44f7-a59c-f603b7517e4d',
  'start': np.nan,
  'end': np.nan}, # person in the background
 '78d9fbfe-04e0-4766-ba15-198e246d5e9c_106': {'name': '10mwt',
  'session': '78d9fbfe-04e0-4766-ba15-198e246d5e9c',
  'start': np.nan,
  'end': np.nan}, # people in the background, should have been handled, nothing else obvious
 '773c33cd-a12d-46d2-af17-35fa7b4e83bd_107': {'name': 'tug_cone',
  'session': '773c33cd-a12d-46d2-af17-35fa7b4e83bd',
  'start': np.nan,
  'end': np.nan}, # people in the background, should have been handled, nothing else obvious
 '715504d6-c823-4632-8c79-b7639bfdcf4a_108': {'name': 'tug_cone',
  'session': '715504d6-c823-4632-8c79-b7639bfdcf4a',
  'start': np.nan,
  'end': np.nan}, # people in the background of one frame
 'bb12568e-2303-4f87-95e4-baec67f2c023_109': {'name': 'tug_cone',
  'session': 'bb12568e-2303-4f87-95e4-baec67f2c023',
  'start': 5,
  'end': np.nan}, # TO REPROCESS people in foreground and background
 'a7675b05-88da-4bb0-b27b-5357c3ec5807_110': {'name': 'tug_cone',
  'session': 'a7675b05-88da-4bb0-b27b-5357c3ec5807',
  'start': 4,
  'end': np.nan}, # TODO REPROCESS; person in frame
 '7d9b3d14-2672-458a-94df-d35c7b5cdcf5_111': {'name': '5xsts',
  'session': '7d9b3d14-2672-458a-94df-d35c7b5cdcf5',
  'start': np.nan,
  'end': np.nan}, # nothing obvious, quality seems bad (clinic)
 '9108903d-26b3-479a-ad49-955aebb868c7_112': {'name': 'TUGfast',
  'session': '9108903d-26b3-479a-ad49-955aebb868c7',
  'start': 8,
  'end': 17}, # TO REPROCESS
 '25ebd0a3-e8ce-4a7a-ad91-0c1d94753b69_113': {'name': 'jump',
  'session': '25ebd0a3-e8ce-4a7a-ad91-0c1d94753b69',
  'start': 2,
  'end': np.nan}, # TO REPROCESS
 '90e3af8d-3aac-4d97-ac5a-70f5c4e911ae_114': {'name': 'tug_cone',
  'session': '90e3af8d-3aac-4d97-ac5a-70f5c4e911ae',
  'start': np.nan,
  'end': np.nan}, # people in the background, should have been handled, nothing else obvious
 'bc4dd6dc-d9d2-4345-b130-d42bb1c4d673_115': {'name': '5TSTS',
  'session': 'bc4dd6dc-d9d2-4345-b130-d42bb1c4d673',
  'start': np.nan,
  'end': np.nan}, # person in background
 '65aa878e-4396-4a9c-a9a0-7a1e2c708f83_116': {'name': 'ARM_ROM',
  'session': '65aa878e-4396-4a9c-a9a0-7a1e2c708f83',
  'start': np.nan,
  'end': np.nan}, # nothing obvious, quality seems bad (clinic) 
 '1fc22b0c-1ef8-4b6d-b83b-85b12634905d_117': {'name': '10mwrt_1',
  'session': '1fc22b0c-1ef8-4b6d-b83b-85b12634905d',
  'start': 3,
  'end': np.nan}, # TO REPROCESS
 '28ec73c6-025b-4ea4-b844-803e8205bf1a_118': {'name': 'brooke',
  'session': '28ec73c6-025b-4ea4-b844-803e8205bf1a',
  'start': np.nan,
  'end': np.nan}, # hands hidden
 '51003106-cdec-40f7-9204-0e21953bb4a7_119': {'name': '10mwt',
  'session': '51003106-cdec-40f7-9204-0e21953bb4a7',
  'start': np.nan,
  'end': np.nan}, # nothing obvious
 '109eabdf-e50d-459e-8917-c856338a720b_120': {'name': '10mwrt',
  'session': '109eabdf-e50d-459e-8917-c856338a720b',
  'start': 1,
  'end': np.nan}, # TO REPROCESS, not sure that will help
 'aca92056-3f67-403c-8d6a-513055274ffe_121': {'session': 'aca92056-3f67-403c-8d6a-513055274ffe',
  'name': 'brooke',
  'start': np.nan,
  'end': 7}, # VIDEOS DIFFERENT LENGTHS; one side is truncated
 'aca92056-3f67-403c-8d6a-513055274ffe_122': {'session': 'aca92056-3f67-403c-8d6a-513055274ffe',
  'name': 'tug_cone',
  'start': np.nan,
  'end': np.nan}, # VIDEOS DIFFERENT LENGTHS; one side is truncated
 'aca92056-3f67-403c-8d6a-513055274ffe_123': {'name': '5xsts',
  'session': 'aca92056-3f67-403c-8d6a-513055274ffe',
  'start': np.nan,
  'end': np.nan}, # VIDEOS DIFFERENT LENGTHS; one side is truncated
 '25d785ae-f49e-4531-b14c-86d6d5e9d144_124': {'name': 'toe_stand',
  'session': '25d785ae-f49e-4531-b14c-86d6d5e9d144',
  'start': 0,
  'end': 30.5}, # TO REPROCESS REDO
 '9e5d7c34-d9fc-43f7-85e9-d3e5def7ef3a_125': {'name': '10mrt',
  'session': '9e5d7c34-d9fc-43f7-85e9-d3e5def7ef3a',
  'start': np.nan,
  'end': np.nan}, # VIDEOS DIFFERENT LENGTHS, one side is 1 second truncated
 'bca0aad8-c129-4a62-bef3-b5de1659df5e_126': {'name': 'tug_cone',
  'session': 'bca0aad8-c129-4a62-bef3-b5de1659df5e',
  'start': np.nan,
  'end': 9}, # TO REPROCESS; person in background
 '98fc0617-1f5e-44ba-92c9-a32f09899295_127': {'name': 'brooke',
  'session': '98fc0617-1f5e-44ba-92c9-a32f09899295',
  'start': 10,
  'end': np.nan} # TO REPROCESS
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
    
def trim_video_by_frames(video_path, start_frame, frame_duration, frame_rate, trimmed_video_path):
    # Build the ffmpeg command to trim without re-encoding
    ffmpeg_cmd = f'ffmpeg -i {video_path} -ss {start_frame / frame_rate} -frames:v {frame_duration} -c:v copy {trimmed_video_path}'

    os.system(ffmpeg_cmd)

for count, session_id_all in enumerate(all_bad_data):
    
    if session_id_all in sessions_to_exclude:
        continue

    # if count < 90:
    #     continue
    
    # if count > 89:
    #     continue

    # if count != 124:
    #     continue

    session_id = session_id_all.split('_')[0]
    trial_name = all_bad_data[session_id_all]['name']
    trial_id = get_trial_id(session_id, trial_name)
    trial_json = get_trial_json(trial_id)
    for k, video in enumerate(trial_json["videos"]):

        # Download the video
        videoDir = os.path.join(data_folder, session_id_all, trial_id, "Videos", video['id'], "InputMedia")
        os.makedirs(videoDir, exist_ok=True)
        video_path = os.path.join(videoDir, trial_id + ".mov")
        # download_file(video["video"], video_path)

        trimmedVideoDir = os.path.join(data_folder, session_id_all, trial_id, "Videos", video['id'], "Trimmed")
        os.makedirs(trimmedVideoDir, exist_ok=True)
        
        file_name = os.path.basename(video_path)
        trimmed_video_path = os.path.join(trimmedVideoDir, file_name)

        # Trim the video
        frame_rate, frame_count = get_video_info(video_path)
        video_duration = frame_count / frame_rate

        start = all_bad_data[session_id_all]['start']
        end = all_bad_data[session_id_all]['end']
        # if both are nan, skip
        reprocess_trial = True
        if np.isnan(start) and np.isnan(end):
            # Copy video to trimmed video directory
            # shutil.copy2(video_path, trimmed_video_path)
            reprocess_trial = False
            continue
        # if end is not nan and start is nan, set start to 0
        if np.isnan(start):
            start = 0
        # if start is not nan and end is nan, set end to video duration
        if np.isnan(end):
            end = video_duration
        # if end is more than video duration, set end to video duration
        if end > video_duration:
            end = video_duration

        n_frame_trim_start = int(np.floor(start * frame_rate))
        desired_num_frames = int(np.floor((end - start) * frame_rate))
        
        trimmed_video_path = os.path.join(trimmedVideoDir, trial_id + ".mov")
        # trim_video_by_frames(video_path, n_frame_trim_start, desired_num_frames, frame_rate, trimmed_video_path)

        # Post the trimmed video
        post_video_to_trial(trimmed_video_path,trial_id,video['device_id'],json.dumps(video['parameters']))
        
        # Delete the old video
        delete_video_from_trial(video['id'])

    if reprocess_trial:
      # Delete the results of that trial
      delete_results(trial_id)

      # Change the status to stopped
      set_trial_status(trial_id, "stopped")
      test=1