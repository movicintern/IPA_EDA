import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import librosa
import scipy
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from time import sleep
import time
from datetime import datetime, timedelta
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import librosa
import scipy
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from time import sleep
# from PIL import Image
import time


def plot_spectrogram(sensor_name, spec, setup, date_info):
    im = np.flip(np.transpose(spec), axis=0)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    #ax.imshow(im, extent=[0, setup.cut_time, 0, int(setup.sampling_rate/1000/2)], cmap='jet')  # , cmap='cividis'
    ax.imshow(im, cmap='jet')  # , cmap='cividis'
    plt.xlabel(f'Time({setup.cut_time}sec)')
    plt.ylabel('Frequency(kHz)')
    plt.title('Spectrogram')
    fig.savefig(os.path.join(setup.spectrogram_path, sensor_name, f'{date_info[0]}_{date_info[1]}.png'))
    plt.clf()
    plt.close(fig)

    
def plot_spectrum(sensor_name, fft_graph, setup, date_info):
    fig = plt.figure(figsize=(24, 12))
    ax = fig.add_subplot(111)
    #     plt.stem(fft_graph[1], fft_graph[0], bottom=np.min(fft_graph[0]), markerfmt=' ', use_line_collection=True)
    ax.plot(fft_graph[1][::100], fft_graph[0][::100])
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Decibel(dB)')
    plt.title('Spectrum')
    fig.savefig(os.path.join(setup.spectrum_path, sensor_name, f'{date_info[0]}_{date_info[1]}.png'))
    plt.clf()
    plt.close(fig)

def multiple_plot_spectrogram(sensor_name_set, spec, setup, date_info):

    ch_n = len(sensor_name_set)
    row_n, col_n = int(np.round(ch_n/2)), 2
    #row_n, col_n = 1, 3

    plt.figure(figsize=(12, 12))
    plt.suptitle("Spectrogram", fontsize=20)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=1.2)
    for idx in range(ch_n):
        plt.subplot(row_n, col_n, idx+1)
        im = np.flip(np.transpose(spec[idx]), axis=0)
        plt.imshow(im, extent=[0, setup.cut_time, 0, int(setup.sampling_rate/1000/2)], cmap='jet', aspect='auto')  # , cmap='cividis'
        plt.xlabel(f'Time({setup.cut_time}sec)')
        plt.ylabel('Frequency(kHz)')
        plt.title(sensor_name_set[idx])

    plt.savefig(os.path.join(setup.spectrogram_path, 'multi_sensor', f'{date_info[0]}_{date_info[1]}.png'))
    plt.clf()
    plt.close()

def multiple_plot_spectrum(sensor_name_set, fft_graph, setup, date_info):
    ch_n = len(sensor_name_set)
    #row_n, col_n = int(np.round(ch_n / 2)), 2
    row_n, col_n = 1, 3
    plt.figure(figsize=(24, 12))
    plt.suptitle("Spectrum", fontsize=20)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=1.2)

    for idx in range(ch_n):
        plt.subplot(row_n, col_n, idx+1)
        #     plt.stem(fft_graph[1], fft_graph[0], bottom=np.min(fft_graph[0]), markerfmt=' ', use_line_collection=True)
        plt.plot(fft_graph[idx][1][::100], fft_graph[idx][0][::100])
        plt.xlabel('frequency(Hz)')
        plt.ylabel('Decibel(dB)')
        plt.title(sensor_name_set[idx])

    plt.savefig(os.path.join(setup.spectrum_path, 'multi_sensor', f'{date_info[0]}_{date_info[1]}.png'))
    plt.clf()
    plt.close()


class TrackerVisualizer:
    def __init__(self):
        pass
    
    def _threshold(self, axis_data, scaling_factor=1.3):
        if axis_data.size == 0:
            raise ValueError("Input array is empty.")
        q25, q75 = np.quantile(axis_data, [0.25, 0.75])
        iqr = q75 - q25
        ref_value = scaling_factor * (np.median(axis_data) + np.std(axis_data)) / 2
        distance_to_q3 = q75 - ref_value
        dynamic_multiplier = distance_to_q3 / iqr if iqr != 0 else 1
        dynamic_multiplier = np.clip(dynamic_multiplier, 0.1, 1.5)
        quantile_val = ref_value + dynamic_multiplier * (iqr / 2)
        return quantile_val

    def visualize(self, sensor_name, value_set, setup, T, multiview_Flag=False, max_ticks=20):
        start_time = datetime.strptime(setup.start_date, "%Y%m%d%H%M%S")
        end_time = datetime.strptime(setup.end_date, "%Y%m%d%H%M%S")
        total_seconds = int((end_time - start_time).total_seconds())
        ticks = np.arange(0, total_seconds, setup.cut_time)
        time_labels = [(start_time + timedelta(seconds=int(t+30))).strftime('%Y-%m-%d-%H-%M-%S') for t in ticks]
        
        value_set = np.array(value_set)
        num_sensors = len(value_set) if multiview_Flag else 1
        fig, axs = plt.subplots(num_sensors, 1, figsize=(18, 12), squeeze=False)

        for idx in range(num_sensors):
            values = value_set[idx] if multiview_Flag else value_set
            if len(values) != len(ticks):
                values = np.resize(values, len(ticks))

            threshold = self._threshold(values)
            print(f'Threshold for {sensor_name[idx] if multiview_Flag else sensor_name}: {threshold}')
            thresholding_list, _ = self._check_threshold_exceedances(axs[idx, 0], ticks, values, threshold, start_time)
            axs[idx, 0].axhline(y=threshold, color='red', linestyle='--', linewidth=3, label='Threshold')
            axs[idx, 0].plot(ticks, values, linestyle='-', color='b', label='Score')

            for start_str, end_str, color in T:
                self._add_interval_to_plot(axs[idx, 0], values, start_str, end_str, color, start_time)

            self._format_ticks(axs[idx, 0], ticks, time_labels, max_ticks, sensor_name[idx] if multiview_Flag else sensor_name)

            cnt = 0
            for i in range(0, len(thresholding_list) - 1, 2):
                axs[idx, 0].text(ticks[1], np.median(values),
                                  f'Event[{cnt + 1}]: {thresholding_list[i].strftime("%Y%m%d%H%M%S")}~{thresholding_list[i + 1].strftime("%Y%m%d%H%M%S")}',
                                  horizontalalignment='left', verticalalignment='bottom',
                                  fontsize=15, color='purple', fontweight='bold')
                cnt += 1

            axs[idx, 0].set_title(f'Sensor: {sensor_name[idx]}' if multiview_Flag else f'Sensor: {sensor_name}', fontsize=22)

        plt.tight_layout()
        save_path = os.path.join(setup.operation_path, 'multi_sensor' if multiview_Flag else sensor_name[-1], f'{setup.start_date}_{setup.end_date}.png')
        print(save_path)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
    def _add_interval_to_plot(self, ax, value_set, start_str, end_str, color, start_time):
        
        date_time_start = datetime.strptime(start_str, '%Y%m%d%H%M%S')
        start_str = date_time_start.strftime('%Y-%m-%d %H:%M:%S')
        date_time_end = datetime.strptime(end_str, '%Y%m%d%H%M%S')
        end_str = date_time_end.strftime('%Y-%m-%d %H:%M:%S')

        T_start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        T_end_time = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
        T_start_tick = (T_start_time - start_time).total_seconds()
        T_end_tick = (T_end_time - start_time).total_seconds()

        ax.axvspan(T_start_tick, T_end_tick, color=color, alpha=0.5)
        #ax.axvline(x=T_start_tick, color='black', linestyle='--', linewidth=2, alpha=0.5)
        #ax.axvline(x=T_end_tick, color='black', linestyle='--', linewidth=2, alpha=0.5)
        self._add_labels(ax, value_set, T_start_tick, T_end_tick, color)

    def _add_labels(self, ax, value_set, T_start_tick, T_end_tick, color):
        mid_point = (T_start_tick + T_end_tick) / 2
        label = 'RUN' if color == 'gray' else 'DOWN'
        ax.text(mid_point, np.median(value_set), label, horizontalalignment='center', verticalalignment='center', fontsize=14,
                color='red' if label == 'RUN' else 'black', fontweight='bold')

    def _format_ticks(self, ax, ticks, time_labels, max_ticks, sensor_name):
        tick_indices = np.linspace(0, len(time_labels) - 1, max_ticks, dtype=int)
        ax.set_xticks(ticks[tick_indices])
        ax.set_xticklabels(np.array(time_labels)[tick_indices], rotation=270)
        ax.legend(loc='upper right', fontsize=13)
        ax.set_ylabel('Operational Probability', fontsize=15)
        ax.set_title(sensor_name, fontsize=22)

    def _check_threshold_exceedances(self, ax, ticks, value_set, threshold, start_time):
        exceeds_threshold = np.where(value_set > threshold)[0]
        thresholding_list = []
        if exceeds_threshold.size > 0:
            group_start = exceeds_threshold[0]
            for i in range(1, exceeds_threshold.size):
                if exceeds_threshold[i] != exceeds_threshold[i - 1] + 1:
                    thresholding_list.extend(self._highlight_exceedances(ax, start_time, int(ticks[group_start]), int(ticks[exceeds_threshold[i - 1]])))
                    group_start = exceeds_threshold[i]
            thresholding_list.extend(self._highlight_exceedances(ax, start_time, int(ticks[group_start]), int(ticks[exceeds_threshold[-1]])))
        return thresholding_list, len(thresholding_list) // 2

    def _highlight_exceedances(self, ax, start_time, first_tick, last_tick):
        first_time = start_time + timedelta(seconds=first_tick)
        last_time = start_time + timedelta(seconds=last_tick)
        ax.axvspan(first_tick, last_tick, color='green', alpha=0.5, label=f'Detected Operational Interval')
        return [first_time, last_time]
