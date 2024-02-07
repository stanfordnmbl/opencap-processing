import os
import sys
sys.path.append("..")
import utilsKinematics
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
from scipy.signal import find_peaks
import pandas as pd
from scipy.fft import fft, ifft
from scipy.signal import butter, filtfilt

def dist(x,y):
    # return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
    return abs(x[1]-y[1])**10

def find_optimal_x_position(sliding_plot, fixed_plot, window=400, cutoff=30):
    distances = []

    for win in range(window, 1, -1):
        # Slide sliding plot across the X axis starting from left
        total_dist = 0
        for i in range(0, len(sliding_plot)-win-cutoff):
            total_dist += dist(sliding_plot[i+win], fixed_plot[i])
        
        distances.append(np.mean(total_dist))
    
    for win in range(window):
        # Slide sliding plot across the X axis starting from original start
        total_dist = 0
        for i in range(cutoff, len(sliding_plot)-win):
            total_dist += dist(sliding_plot[i], fixed_plot[i+win])
        
        distances.append(np.mean(total_dist))
    
    optimal_x = np.argmin(np.array(distances)) - window
    neg = False
    if optimal_x < 0:
        neg = True
    return optimal_x, neg

def smooth(s, win):
    return pd.Series(s).rolling(window=win, center=True).mean().ffill().bfill()

def peak_interval(x, y1, y2, Y1_WINDOW=75, y1_prominence=2, y2_prominence=2):
    smoothed_y = np.clip(smooth(y1, 2*Y1_WINDOW)-smooth(y1, 10*Y1_WINDOW), 0, np.inf)
    peaks_y1 = find_peaks(smoothed_y, prominence=y1_prominence)
    smoothed_y = np.clip(smooth(y2, 2*Y1_WINDOW)-smooth(y2, 10*Y1_WINDOW), 0, np.inf)
    peaks_y2 = find_peaks(y2, prominence=y2_prominence)

    peaks_y1 = np.array(peaks_y1[0])
    peaks_y2 = np.array(peaks_y2[0])
    peaks = np.concatenate((peaks_y1, peaks_y2))
    peaks = np.sort(peaks)

    largest_interval = 0
    last_p = 0
    for p in peaks:
        if p - last_p > largest_interval:
            largest_interval = p-last_p
        last_p = p
    #     plt.axvline(p)
    # plt.show()
    if len(x) - last_p > largest_interval:
        largest_interval = len(x) - last_p

    return int(largest_interval*0.8)

def fourier_smoothing(data, cutoff: int) -> np.ndarray:
        transformed = fft(data)
        transformed[cutoff:-cutoff] = 0
        return ifft(transformed).real

def exp_moving_average(self, alpha: float) -> np.ndarray:
    smoothed_data = np.zeros_like(self.data)
    smoothed_data[0] = self.data[0]
    for i in range(1, len(self.data)):
        smoothed_data[i] = alpha * self.data[i] + (1 - alpha) * smoothed_data[i-1]
    return smoothed_data

def butterworth(self, cutoff=0.125, fs=30.0, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, self.data, axis=0)
    return y

def angle_from_center(center, v1, v2, acute=True, deg=True):
    norm_v1 = v1 - center
    norm_v2 = v2 - center
    angle = np.arccos(np.dot(norm_v1, norm_v2) / (np.linalg.norm(norm_v1) * np.linalg.norm(norm_v2)))
    if acute:
        if deg:
            return angle * 180 / np.pi
        return angle
    else:
        angle = np.pi - angle
        if deg:
            return angle * 180 / np.pi
        return angle

def get_interval_indices(times, start_time, end_time):
    start_time_index = -1
    end_time_index = -1
    for i in range(len(times)):
        if start_time <= times[i] and start_time_index == -1:
            start_time_index = i
        if end_time <= times[i] and end_time_index == -1:
            end_time_index = i
    
    if start_time_index == -1:
        start_time_index = len(times) - 1
    if end_time_index == -1:
        end_time_index = len(times) - 1

    return start_time_index, end_time_index

def get_highest_peaks(x, y1, y2, Y1_WINDOW=75, y1_prominence=2, y2_prominence=2):
    smoothed_y = np.clip(smooth(y1, 2*Y1_WINDOW)-smooth(y1, 10*Y1_WINDOW), 0, np.inf)
    peaks_y1 = find_peaks(smoothed_y, prominence=y1_prominence)
    smoothed_y = np.clip(smooth(y2, 2*Y1_WINDOW)-smooth(y2, 10*Y1_WINDOW), 0, np.inf)
    peaks_y2 = find_peaks(y2, prominence=y2_prominence)

    highest_y1 = 0
    for p in peaks_y1[0]:
        if y1[p] > y1[highest_y1]:
            highest_y1 = p

    highest_y2 = 0
    for p in peaks_y2[0]:
        if y2[p] > y2[highest_y2]:
            highest_y2 = p

    return highest_y1, highest_y2

def process_cam_angle(subject: str, session: str, cam: str) -> list:
    print(subject, session, cam)
    # Preset Path Params
    trials_of_interest = ['DJ1', 'DJ2', 'DJ3', 'DJ4', 'DJ5', 'DJAsym1', 'DJAsym2', 'DJAsym3', 'DJAsym4', 'DJAsym5', 'squats1', 'squatsAsym1', 'STS1', 'STSweakLegs1']
    trials_of_interest += ['walking1', 'walking2', 'walking3', 'walking4', 'walkingTS1', 'walkingTS2', 'walkingTS3', 'walkingTS4']

    data_folder_name = str('opencap_LabValidation_withVideos_'+subject+'_VideoData_'+session)
    data_folder = os.path.join(os.getcwd(), "../Data", data_folder_name)
    json_data_folder_name = str('movenet_opencap_LabValidation_withVideos_'+subject+'_VideoData_'+session+'_'+cam)
    json_study_prefix = str(data_folder_name+'_'+cam+'_')
    json_data_folder = os.path.join(os.getcwd(), "../Data", json_data_folder_name)
    avi_data_folder = str('/Users/davidspector/Home/OpenMotion/LabValidation_withVideos/'+subject+'/VideoData/'+session+'/'+cam)

    model_path = os.path.join(data_folder, 'OpenSimData', 'Model')
    for file in os.listdir(model_path):
        if file.endswith('.osim'):
            model_name = file[:-5]
            break

    # Process data.
    trial_names = []
    kinematics, coordinates, muscle_tendon_lengths, moment_arms, center_of_mass, marker_coordinates = {}, {}, {}, {}, {}, {}
    coordinates['values'], coordinates['speeds'], coordinates['accelerations'] = {}, {}, {}
    center_of_mass['values'], center_of_mass['speeds'], center_of_mass['accelerations'] = {}, {}, {}

    for file in os.listdir(os.path.join(data_folder, 'OpenSimData', 'Kinematics')):
        if file.endswith('.mot') and file[:-4] in trials_of_interest:
            trial_name = file[:-4]

            # Create object from class kinematics.
            kinematics[trial_name] = utilsKinematics.kinematics(data_folder, trial_name, modelName=model_name, lowpass_cutoff_frequency_for_coordinate_values=10)
            
            # Get coordinate values, speeds, and accelerations.
            coordinates['values'][trial_name] = kinematics[trial_name].get_coordinate_values(in_degrees=True) # already filtered
            coordinates['speeds'][trial_name] = kinematics[trial_name].get_coordinate_speeds(in_degrees=True, lowpass_cutoff_frequency=10)
            coordinates['accelerations'][trial_name] = kinematics[trial_name].get_coordinate_accelerations(in_degrees=True, lowpass_cutoff_frequency=10)
            
            # Get muscle-tendon lengths.
            muscle_tendon_lengths[trial_name] = kinematics[trial_name].get_muscle_tendon_lengths()
            
            # Get center of mass values, speeds, and accelerations.
            center_of_mass['values'][trial_name] = kinematics[trial_name].get_center_of_mass_values(lowpass_cutoff_frequency=10)
            center_of_mass['speeds'][trial_name] = kinematics[trial_name].get_center_of_mass_speeds(lowpass_cutoff_frequency=10)
            center_of_mass['accelerations'][trial_name] = kinematics[trial_name].get_center_of_mass_accelerations(lowpass_cutoff_frequency=10)

            # Get model coordinates
            marker_coordinates[trial_name] = kinematics[trial_name].get_marker_dict(data_folder, trial_name)
            marker_coordinates[trial_name]['markers']['time'] = marker_coordinates[trial_name]['time']

            trial_names.append(trial_name)

    # Load JSON files
    keypoints_order = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 
                    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                    'right_knee', 'left_ankle', 'right_ankle']

    movenet_outputs = {}
    initial_trial_names = trial_names.copy()
    for trial_name in initial_trial_names:
        trial_path = os.path.join(json_data_folder, str(json_study_prefix + trial_name + '.json'))
        if os.path.exists(trial_path):
            file = open(trial_path)
            json_dict = json.load(file)
            movenet_outputs[trial_name] = {}
            for j in range(17): # ignore confidence score for now
                movenet_outputs[trial_name][keypoints_order[j]] = np.ones((len(json_dict),2))
            for i in range(len(json_dict)):
                for j in range(17):
                    movenet_outputs[trial_name][keypoints_order[j]][i] = json_dict[i]['keypoints'][j*3:(j*3)+2]
            avi_path = os.path.join(avi_data_folder, trial_name, str(trial_name + '_syncdWithMocap.avi'))
            avi_vid = cv2.VideoCapture(avi_path)
            fps = avi_vid.get(cv2.CAP_PROP_FPS)
            movenet_outputs[trial_name]['time'] = np.arange(0, (1/fps)*len(json_dict), 1/fps)
            if len(movenet_outputs[trial_name]['time']) > len(movenet_outputs[trial_name]['right_hip']):
                movenet_outputs[trial_name]['time'] = movenet_outputs[trial_name]['time'][:len(movenet_outputs[trial_name]['right_hip'])]
            elif len(movenet_outputs[trial_name]['time']) < len(movenet_outputs[trial_name]['right_hip']):
                movenet_outputs[trial_name]['time'] = np.append(movenet_outputs[trial_name]['time'], movenet_outputs[trial_name]['time'][-1]+(movenet_outputs[trial_name]['time'][1]-movenet_outputs[trial_name]['time'][0]))
            file.close()
        else:
            trial_names.remove(trial_name)

    # Get start and end times for all trials from saved motion
    num_trials = len(trial_names)
    start_times = []
    end_times = []
    time_arrays = list(np.empty(num_trials))
    start_time_indices = list(np.empty(num_trials))
    end_time_indices = list(np.empty(num_trials))

    for i in range(num_trials):
        start_time = kinematics[trial_names[i]].time[0]
        end_time = kinematics[trial_names[i]].time[-1]
        start_times.append(start_time)
        end_times.append(end_time)
        time_array = marker_coordinates[trial_names[i]]['markers']['time']
        start_time_indices[i], end_time_indices[i] = get_interval_indices(time_array, start_times[i], end_times[i])
        time_arrays[i] = time_array[start_time_indices[i]:end_time_indices[i]+1] - time_array[start_time_indices[i]]

    optimal_xs = []
    movenet_x_data = []
    movenet_y_data = []
    opensim_x_data = []
    opensim_y_data = []

    for trial_index in range(len(trial_names)):
        # Smooth Movenet outputs and calculate knee joint angles
        # try:
        #     mv_r_knee_y = fourier_smoothing(movenet_outputs[trial_names[trial_index]]['right_knee'][:,1], cutoff=30)
        #     mv_r_knee_x = fourier_smoothing(movenet_outputs[trial_names[trial_index]]['right_knee'][:,0], cutoff=30)
        # except:
        #     pass
        # mv_r_hip_y = fourier_smoothing(movenet_outputs[trial_names[trial_index]]['right_hip'][:,1], cutoff=30)
        # mv_r_hip_x = fourier_smoothing(movenet_outputs[trial_names[trial_index]]['right_hip'][:,0], cutoff=30)
        # mv_r_ankle_y = fourier_smoothing(movenet_outputs[trial_names[trial_index]]['right_ankle'][:,1], cutoff=30)
        # mv_r_ankle_x = fourier_smoothing(movenet_outputs[trial_names[trial_index]]['right_ankle'][:,0], cutoff=30)
        # try:
        #     mv_r_knee_y = butterworth(movenet_outputs[trial_names[trial_index]]['right_knee'][:,1], cutoff=0.125, fs=5.0, order=5)
        #     mv_r_knee_x = butterworth(movenet_outputs[trial_names[trial_index]]['right_knee'][:,0], cutoff=0.125, fs=5.0, order=5)
        # except:
        #     pass
        # mv_r_hip_y = butterworth(movenet_outputs[trial_names[trial_index]]['right_hip'][:,1], cutoff=0.125, fs=5.0, order=5)
        # mv_r_hip_x = butterworth(movenet_outputs[trial_names[trial_index]]['right_hip'][:,0], cutoff=0.125, fs=5.0, order=5)
        # mv_r_ankle_y = butterworth(movenet_outputs[trial_names[trial_index]]['right_ankle'][:,1], cutoff=0.125, fs=5.0, order=5)
        # mv_r_ankle_x = butterworth(movenet_outputs[trial_names[trial_index]]['right_ankle'][:,0], cutoff=0.125, fs=5.0, order=2)
        try:
            mv_r_knee_y = exp_moving_average(movenet_outputs[trial_names[trial_index]]['right_knee'][:,1], alpha=0.15)
            mv_r_knee_x = exp_moving_average(movenet_outputs[trial_names[trial_index]]['right_knee'][:,0], alpha=0.15)
        except:
            pass
        mv_r_hip_y = exp_moving_average(movenet_outputs[trial_names[trial_index]]['right_hip'][:,1], alpha=0.15)
        mv_r_hip_x = exp_moving_average(movenet_outputs[trial_names[trial_index]]['right_hip'][:,0], alpha=0.15)
        mv_r_ankle_y = exp_moving_average(movenet_outputs[trial_names[trial_index]]['right_ankle'][:,1], alpha=0.15)
        mv_r_ankle_x = exp_moving_average(movenet_outputs[trial_names[trial_index]]['right_ankle'][:,0], alpha=0.15)
        mv_r_ankle = np.array([mv_r_ankle_x, mv_r_ankle_y]).T
        mv_r_hip = np.array([mv_r_hip_x, mv_r_hip_y]).T
        mv_r_knee = np.array([mv_r_knee_x, mv_r_knee_y]).T
        cur_mv_knee_angles = []
        for i in range(len(movenet_outputs[trial_names[trial_index]]['time'])):
            cur_mv_knee_angles.append(angle_from_center(mv_r_knee[i], mv_r_ankle[i], mv_r_hip[i], acute=False, deg=True))

        # Make each time series share time points
        univ_x = np.linspace(time_arrays[trial_index][0]+start_times[trial_index], time_arrays[trial_index][-1]+start_times[trial_index], 1000)
        movenet_y = np.interp(univ_x, movenet_outputs[trial_names[trial_index]]['time'], np.nan_to_num(cur_mv_knee_angles))
        opensim_y = np.interp(univ_x, time_arrays[trial_index]+start_times[trial_index], coordinates['values'][trial_names[trial_index]]['knee_angle_r'])

        # Find optimal X position
        movenet_plot = np.array([univ_x, movenet_y]).T
        opensim_plot = np.array([univ_x, opensim_y]).T
        window = peak_interval(univ_x, movenet_y, opensim_y, y1_prominence=2, y2_prominence=2)
        optimal_x, neg = find_optimal_x_position(movenet_plot, opensim_plot, window)
        if neg:
            new_x = univ_x - (univ_x[-1*optimal_x]-univ_x[0])
        else:
            new_x = univ_x + (univ_x[optimal_x]-univ_x[0])

        """Make sure highest peak is in time synchronization"""
        hy1, hy2 = get_highest_peaks(univ_x, movenet_y, opensim_y, y1_prominence=2, y2_prominence=2)
        if (new_x[hy1] < univ_x[0] and (hy1>0 or movenet_y[0]<movenet_y[1])) or (new_x[hy1] > univ_x[-1] and (hy1 != len(univ_x)-1 or movenet_y[-1]<movenet_y[-2])):
            optimal_x = hy2 - hy1
            neg = (optimal_x < 0)
            if neg:
                new_x = univ_x - (univ_x[-1*optimal_x]-univ_x[0])
            else:
                new_x = univ_x + (univ_x[optimal_x]-univ_x[0])
        """End highest peak injection"""
            
        movenet_x_data.append(new_x)
        movenet_y_data.append(movenet_y)
        opensim_x_data.append(univ_x)
        opensim_y_data.append(opensim_y)
        optimal_xs.append(optimal_x)

    return movenet_x_data, movenet_y_data, opensim_x_data, opensim_y_data, optimal_xs, trial_names

def save_plots_by_trial_and_cam(subject: str, session: str, cam: str) -> None:
    movenet_x_data, movenet_y_data, opensim_x_data, opensim_y_data, optimal_xs, trial_names = process_cam_angle(subject, session, cam)

    fig, ax = plt.subplots()
    for (trial_name, movenet_x, movenet_y, opensim_x, opensim_y, optimal_x) in zip(
        trial_names, movenet_x_data, movenet_y_data, opensim_x_data, opensim_y_data, optimal_xs):   
        # Plot curves
        ax.plot(movenet_x, movenet_y, label="Synced Movenet w OpenSim")
        ax.plot(opensim_x, opensim_y, label="OpenSim")
        ax.set_title(trial_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Knee Angle (deg)")

        # Analyze peaks
        peaks_o = find_peaks(opensim_y, prominence=5)
        i = 1
        for p in peaks_o[0]:
            if p - optimal_x < 0 or p >= len(opensim_y) or p - optimal_x >=len(opensim_y):
                continue
            ax.axvline(opensim_x[p], c='purple')
            i += 1
        fig.legend()
        file_name = os.path.join('Plots_Exp', str(subject+'_MoveNetOpenSimPlot_'+session+'_'+cam+'_'+trial_name+'.png'))
        fig.savefig(file_name, bbox_inches='tight')
        ax.clear()
    plt.close()

    
def get_errors_by_trial_and_cam(subject: str, session: str, cam: str, rel_error: bool) -> None:
    _, movenet_y_data, _, opensim_y_data, optimal_xs, trial_names = process_cam_angle(subject, session, cam)

    trial_errors = {}
    for (trial_name, opensim_y, movenet_y, optimal_x) in zip(trial_names, opensim_y_data, movenet_y_data, optimal_xs):
        # Analyze peaks
        peaks_o = find_peaks(opensim_y, prominence=5)
        i = 1
        cur_errors = []
        for p in peaks_o[0]:
            if p - optimal_x < 0 or p >= len(opensim_y) or p - optimal_x >=len(opensim_y):
                continue
            if rel_error:
                error = (100*(opensim_y[p] - movenet_y[p-optimal_x])/opensim_y[p])
            else:
                error = abs(opensim_y[p] - movenet_y[p-optimal_x])
            cur_errors.append(error)
            i += 1
        trial_errors[trial_name] = cur_errors

    if rel_error:
        folder_name = 'Error_Files_Exp_Rel'
    else:
        folder_name = 'Error_Files_Exp_Abs'

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for trial_name in trial_names:
        npy_name = os.path.join(folder_name, str(trial_name+'_'+cam+'.npy'))
        if os.path.exists(npy_name):
            cur_errors = np.load(npy_name)
            os.remove(npy_name)
            np.save(npy_name, np.concatenate((np.array(cur_errors),np.array(trial_errors[trial_name]))))
        else:
            np.save(npy_name, trial_errors[trial_name])

def get_all_errors(subject: str, session: str, cam: str, rel_error: bool) -> None:
    _, movenet_y_data, _, opensim_y_data, optimal_xs, _ = process_cam_angle(subject, session, cam)

    av_errors = []
    all_errors = []

    for (movenet_y, opensim_y, optimal_x) in zip(movenet_y_data, opensim_y_data, optimal_xs):

        # Analyze peaks
        peaks_o = find_peaks(opensim_y, prominence=5)
        i = 1
        cur_errors = []
        for p in peaks_o[0]:
            if p - optimal_x < 0 or p >= len(opensim_y) or p - optimal_x >=len(opensim_y):
                continue
            if rel_error:
                error = (100*(opensim_y[p] - movenet_y[p-optimal_x])/opensim_y[p])
            else:
                error = abs(opensim_y[p] - movenet_y[p-optimal_x])
            cur_errors.append(error)
            i += 1
        if len(cur_errors):
            av_errors.append(np.mean(cur_errors))
            all_errors += cur_errors

    if rel_error:
        dest_file_prefix = 'all_rel_errors'
    else:
        dest_file_prefix = 'all_abs_errors'

    if cam == 'cam0' and subject == 'subject2' and session == 'Session0':
        np.save(str(dest_file_prefix+'_s0.npy'), all_errors)
        np.save(str(dest_file_prefix+'_both.npy'), all_errors)
    else:
        if session == 'Session0':
            all_errors_s0 = list(np.load(str(dest_file_prefix+'_s0.npy')))
            os.remove(str(dest_file_prefix+'_s0.npy'))
            np.save(str(dest_file_prefix+'_s0.npy'), list(all_errors_s0 + all_errors))
        all_errors_both = list(np.load(str(dest_file_prefix+'_both.npy')))
        os.remove(str(dest_file_prefix+'_both.npy'))
        np.save(str(dest_file_prefix+'_both.npy'), list(all_errors_both + all_errors))

if __name__ == '__main__':
    for subject in range(2,12):
        if subject == 6:
            continue
        for session in range(2):
            for cam in range(5):
                if cam == 2:
                    continue
                get_errors_by_trial_and_cam(str('subject'+str(subject)), str('Session'+str(session)), str('cam'+str(cam)), False)