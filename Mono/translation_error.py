# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:24:13 2024

@author: suhlr
"""

import sys
import os
import numpy as np
sys.path.append('../')
from utils import load_storage
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d

# colors
colors = {'mocap': np.array((218, 216, 214))/255,
'wham_opt': np.array((255, 131, 0))/255, # orange
'wham_no_opt': np.array((165, 11, 199))/255,# is purple
'video_2cam': np.array((0, 255, 255))/255 # is cyan
}

# load four .mot files
root_path = 'C:/Users/suhlr/Downloads/output/squats1/OpenSim/'
root_path_walking = 'C:/Users/suhlr/Downloads/output/walking4/case0/walking4_trimmed/OpenSim/'
figure_directory = 'G:/My Drive/Utah/Conferences_Talks/2024/ASB/Figures'

mot_paths = {'mocap':'Reference/Mocap/IK/squats1_compareToMonocular.mot',
             'video_2cam':'Reference/Video_2cam/IK/squats1_compareToMonocular.mot',
             'wham_opt':'IK/squats1_5/squats1_5_compareToMonocular.mot',
             'wham_no_opt':'IK/squats1_5_wham_result/squats1_5_wham_result_compareToMonocular.mot'
            }

grf_paths = {'mocap':'Reference/Forces/squats1_forces_compareToMonocular.mot',
             'video_2cam':'Reference/Video_2cam/Dynamics/squats1_1/forces_resultant_compareToMonocular.mot',
             'wham_opt':'Dynamics/squats1_5_rep1/GRF_resultant_squats1_5_0_compareToMonocular.mot'
            }   

grf_paths_walking = {'mocap':'Reference/ForceData/walking4_forces_compareToMonocular.mot',
                     'video_2cam':'Reference/OpenSimData/Video/HRNet/2-cameras/Dynamics/walking4/forces_resultant_compareToMonocular.mot', 
                     'wham_opt':'Dynamics/walking4/GRF_resultant_walking4_0_compareToMonocular.mot'} 

mot_paths_walking = {'mocap':'Reference/OpenSimData/Mocap/IK/walking4_compareToMonocular.mot',
                     'video_2cam':'Reference/OpenSimData/Video/HRNet/2-cameras/IK/walking4_compareToMonocular.mot',
                     'wham_opt':'IK/walking4_trimmed_5/walking4_trimmed_5_compareToMonocular.mot',
                    }

grf_names = {'mocap':'R_ground_force_',
             'video_2cam':'ground_force_r_',
             'wham_opt':'ground_force_right_'
                }


# %% Functions
# Function to resample a dataframe to match the mocap time vector
def resample_dataframe(df, target_time):
    # Interpolating each column except the time column
    resampled_data = {}
    for col in df.columns:
        if col != 'time':
            interpolator = interp1d(df['time'], df[col], bounds_error=False, fill_value='extrapolate')
            resampled_data[col] = interpolator(target_time)
    resampled_data['time'] = target_time
    return pd.DataFrame(resampled_data)

# function to compute the euclidean distance between pelvis_tx, pelvis_ty, and pelvis_tz columns of two dataframes
def compute_translation_error(df1, df2):
    # get the columns of interest
    cols = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    # compute the squared differences
    squared_diff = (df1[cols] - df2[cols])**2
    # compute the sum of the squared differences
    sum_squared_diff = squared_diff.sum(axis=1)
    # compute the square root of the sum of the squared differences
    euclidean_distance = sum_squared_diff.apply(lambda x: x**0.5).to_numpy()
    return euclidean_distance

# %%

# load the .mot files into a dict of pandas dataframes indexec by the keys of mot_paths
mot_data = {key: load_storage(os.path.join(root_path, mot_paths[key]),outputFormat='dataframe') for key in mot_paths.keys()}
mot_data_walking = {key: load_storage(os.path.join(root_path_walking, mot_paths_walking[key]),outputFormat='dataframe') for key in mot_paths_walking.keys()}

# Extract the 'mocap' time vector
mocap_time = mot_data['mocap']['time']
mocap_time_walking = mot_data_walking['mocap']['time']

# Resample all dataframes to align with the 'mocap' time vector
mot_data = {key: resample_dataframe(df, mocap_time) for key, df in mot_data.items()}  
mot_data_walking = {key: resample_dataframe(df, mocap_time_walking) for key, df in mot_data_walking.items()}

# compute the euclidean distance between video, wham_opt, and wham_no_opt compared to mocap
translation_error = {key: compute_translation_error(mot_data['mocap'], mot_data[key]) for key in ['video_2cam', 'wham_opt', 'wham_no_opt']}

# load the forces files
grf_data = {key: load_storage(os.path.join(root_path, grf_paths[key]),outputFormat='dataframe') for key in grf_paths.keys()}
grf_data_walking = {key: load_storage(os.path.join(root_path_walking, grf_paths_walking[key]),outputFormat='dataframe') for key in grf_paths_walking.keys()}

# filter all the ground force data with a 4th order butterworth
from scipy.signal import butter, filtfilt
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# filter the data
for key in grf_data.keys():
    if key == 'mocap':
        continue
    # get sampling rate from time vector
    fs = 1/(grf_data[key]['time'][1] - grf_data[key]['time'][0])
    for col in grf_data[key].columns:
        if 'time' not in col:
            grf_data[key][col] = butter_lowpass_filter(grf_data[key][col], 6, fs)
for key in grf_data_walking.keys():
    if key == 'mocap':
        continue
    # get sampling rate from time vector
    fs = 1/(grf_data_walking[key]['time'][1] - grf_data_walking[key]['time'][0])
    for col in grf_data_walking[key].columns:
        if 'time' not in col:
            grf_data_walking[key][col] = butter_lowpass_filter(grf_data_walking[key][col], 6, fs)


# %% Plot translational drift

plt.close('all')

# generate a plot with black background with the translation errror vs time (found in the 'time' column of the dataframes)
plt.style.use('dark_background')
# use arial font
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
# make the figure 10" wide and 3" tall
plt.figure(figsize=(10,3))

for key in translation_error.keys():
    # double the line width
    plt.plot(mot_data[key]['time'], translation_error[key]*100, label=key, color=colors[key], linewidth=2)
    # print the mean error at the end of the line
    plt.text(mot_data[key]['time'].iloc[-1], translation_error[key].mean()*100, f'{translation_error[key][-1]*100:.2f} cm', color=colors[key], fontsize=8, verticalalignment='center')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Translation Error [cm]')
# remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# only have a first and last x-tick. The last one should be the round-up integer of the x values
plt.xticks([mot_data['mocap']['time'].iloc[0], np.ceil(mot_data['mocap']['time'].iloc[-1])])
plt.show()

# save it as an svg
plt.savefig(os.path.join(figure_directory, 'translation_error.pdf'), format='pdf', dpi=1200)

# %% Plot GRFs
# Plot the GRFs with their respective time vectors

# find the maximum time of wham_opt, and trim all vectors to that length
max_time = max([grf_data_walking['wham_opt']['time'].iloc[-1] for key in grf_data.keys()])
for key in grf_data.keys():
    grf_data[key] = grf_data[key][grf_data[key]['time'] <= max_time]
for key in grf_data_walking.keys():
    grf_data_walking[key] = grf_data_walking[key][grf_data_walking[key]['time'] <= max_time]

# resample the time column in the grf_walking dataframe to go from 0 to 101. call this column 'time_normalized'
grf_data_walking['mocap']['time_normalized'] = np.linspace(0, 100, len(grf_data_walking['mocap']['time']))
grf_data_walking['video_2cam']['time_normalized'] = np.linspace(0, 100, len(grf_data_walking['video_2cam']['time']))
grf_data_walking['wham_opt']['time_normalized'] = np.linspace(0, 100, len(grf_data_walking['wham_opt']['time']))

plt.style.use('dark_background')
# use arial font
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
# fontsize
plt.rcParams.update({'font.size': 16})
# make the figure 10" wide and 3" tall
plt.figure(figsize=(12,4))
# 1x3 subplot

mass = 62.6*9.8


for key in grf_data_walking.keys():
    # double the line width
    # first subplot
    plt.subplot(1,3,1)
    plt.plot(grf_data_walking[key]['time_normalized'], grf_data_walking[key][grf_names[key] + 'vx']/mass, label=key, color=colors[key], linewidth=2)
    # plot vy
    plt.subplot(1,3,2)
    plt.plot(grf_data_walking[key]['time_normalized'], grf_data_walking[key][grf_names[key] + 'vy']/mass, label=key, color=colors[key], linewidth=2)
    # plot vz
    plt.subplot(1,3,3)
    plt.plot(grf_data_walking[key]['time_normalized'], grf_data_walking[key][grf_names[key] + 'vz']/mass, label=key, color=colors[key], linewidth=2)
plt.legend()
for i in range(3):
    plt.subplot(1,3,i+1)
    # remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

plt.subplot(1,3,1)
plt.ylabel('Ground force (bodyweight)')

# make the vertical scale the same on all subplots. Find the max range, and make that the range of all subplots, but keep the data centered
ranges = []
for i in range(3): 
    plt.subplot(1,3,i+1)
    ranges.append(plt.ylim())
max_range = max([abs(r[0] - r[1]) for r in ranges])
# find the middle of the range for each, and keep that in the center
range_factor = (.5,1,.5)
for i in range(3):
    plt.subplot(1,3,i+1)
    middle = sum(ranges[i])/2
    plt.ylim(middle - max_range*range_factor[i]/2, middle + max_range*range_factor[i]/2)
    plt.ylabel('Ground force (bodyweight)')
    

# only 3 ticks on the y and 5 ticks on the x
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.xticks([0, 50, 100])
    
    # Use MaxNLocator to reduce tick density
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

# make the subplots fit well
plt.tight_layout()


# %% Plot walking kinematics
dofs = ['hip_flexion_r','hip_adduction_r','knee_angle_r','ankle_angle_r']

# resample to 101 points
for key in mot_data_walking.keys():
    mot_data_walking[key]['time_normalized'] = np.linspace(0, 100, len(mot_data_walking[key]['time']))

# plot each degree of freedom from mot_data_walking
plt.style.use('dark_background')
# use arial font
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
# fontsize
plt.rcParams.update({'font.size': 16})
# make the figure 10" wide and 3" tall
plt.figure(figsize=(16,4))
# 1xnDofs subplot
for i, dof in enumerate(dofs):
    plt.subplot(1,len(dofs),i+1)
    for key in mot_data_walking.keys():
        # double the line width
        plt.plot(mot_data_walking[key]['time_normalized'], mot_data_walking[key][dof], label=key, color=colors[key], linewidth=2)
        # add mae between mocap and wham_opt and video_2cam on the plot
        if key != 'mocap':
            mae = (mot_data_walking['mocap'][dof] - mot_data_walking[key][dof]).abs().mean()
            plt.text(mot_data_walking[key]['time_normalized'].iloc[-1], mot_data_walking[key][dof].iloc[-1], f'{mae:.2f}', color=colors[key], fontsize=8, verticalalignment='center')


    plt.title(dof)
    # remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # only have a first and last x-tick. The last one should be the round-up integer of the x values
    plt.xticks([0,50,100])
    # Use MaxNLocator to reduce tick density
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    

plt.tight_layout()











