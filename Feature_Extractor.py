import warnings
import librosa
import numpy as np
from scipy import signal
from skimage.transform import resize
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os 
from scipy.stats import zscore
from scipy.signal import find_peaks

class SignalConvertor:

    def __init__(self) -> None:
        pass

    @staticmethod
    def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):

        S = np.asarray(S)

        if np.issubdtype(S.dtype, np.complexfloating):
            warnings.warn(
                "power_to_db was called on complex input so phase "
                "information will be discarded. To suppress this warning, "
                "call power_to_db(np.abs(D)**2) instead."
            )
            magnitude = np.abs(S)
        else:
            magnitude = S

        if callable(ref):
            # User supplied a function to calculate reference power
            ref_value = ref(magnitude)
        else:
            ref_value = np.abs(ref)

        log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
        log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

        if top_db is not None:
            if top_db < 0:
                top_db = 0
                print('top_db must be a Positive, set top_db to 0')
            log_spec = np.maximum(log_spec, log_spec.max() - top_db)

        return log_spec

    def amplitude_to_db(self, S, ref=1.0, amin=1e-5, top_db=80.0):

        S = np.asarray(S)
        magnitude = np.abs(S)
        if callable(ref):
            # User supplied a function to calculate reference power
            ref_value = ref(magnitude)
        else:
            ref_value = np.abs(ref)

        power = np.square(magnitude, out=magnitude)
        return self.power_to_db(power, ref=ref_value ** 2, amin=amin ** 2, top_db=top_db)


class Spectrogram(SignalConvertor):

    def __init__(self):
        super(Spectrogram, self).__init__()
        
    def __call__(self, data, sr, fft_size, hop_size):
        return self.spectrogram(data, sr, fft_size, hop_size)

    def spectrogram(self, data, sr, fft_size, hop_size,):
        _, _, input_matrix = signal.spectrogram(x=data, fs=sr, nperseg=fft_size, nfft=fft_size, noverlap=hop_size,
                                                mode='magnitude', scaling='spectrum')
        input_matrix = self.amplitude_to_db(input_matrix, ref=np.max)
        input_matrix = np.transpose(input_matrix)
        return input_matrix

class Spectrum(SignalConvertor):

    def __init__(self):
        super(Spectrum, self).__init__()

    @staticmethod
    def pseudo_spectrum(spectrogram):
        return np.sum(spectrogram, axis=0, keepdims=True) / np.shape(spectrogram)[0]
        
    def spectrum(self, data, sr, mode='decibel'):
        T = 1 / sr
        s_fft = np.fft.fft(data)
        frequency = np.fft.fftfreq(len(s_fft), T)

        f_idx_e = int(len(s_fft) / 2)

        amplitude = (abs(s_fft) * (2 / len(s_fft)))[:f_idx_e]

        if mode == 'decibel':
            value = self.amplitude_to_db(amplitude, ref=np.max)
        elif mode == 'amplitude':
            value = amplitude
        elif mode == 'power':
            S = np.asarray(amplitude)
            magnitude = np.abs(S)
            value = np.square(magnitude, out=magnitude)

        else:
            raise KeyError


        frequency = frequency[:f_idx_e]

        fft_graph = [value, frequency]

        return fft_graph


class RunDownTracker:
    def __init__(self, tot_sensors, PLC_min=3.18, PLC_max=41.56, window_size=5, low_cut=20000, high_cut=60000, scaling_vec=2, upper_bound=4.4):
        self.window_size = window_size
        self.PLC_min = PLC_min
        self.PLC_max = PLC_max
        self.tot_sensors = tot_sensors
        self.low_cut = low_cut
        self.high_cut = high_cut

    @staticmethod
    def time_diff(start_date_str, end_date_str):
        date_format = "%Y%m%d%H%M%S"
        time_difference = datetime.strptime(end_date_str, date_format) - datetime.strptime(start_date_str, date_format)
        return int(time_difference.total_seconds())

    @staticmethod
    def time_formatter(dt):
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    def time_range_extention(self, start_date, end_date, minutes= 7,  extend=True):
        start_dt = datetime.strptime(start_date, '%Y%m%d%H%M%S')
        end_dt = datetime.strptime(end_date, '%Y%m%d%H%M%S')
        
        if extend:
            start_dt -= timedelta(minutes=minutes)
            end_dt += timedelta(minutes=minutes)
            return start_dt.strftime("%Y%m%d%H%M%S"), end_dt.strftime("%Y%m%d%H%M%S")
        
        return start_date, end_date

    def run_down_classification(self, start_date, end_date, Is_init=True, only_non=False):
        setup_start_dt, setup_end_dt = self.time_range_extention(start_date, end_date, Is_init=False)
        
        if Is_init:
            T_range = [
                (start_date, setup_start_dt, 'white'),
                (setup_start_dt, setup_end_dt, 'gray'),
                (setup_end_dt, end_date, 'white')
            ]
        elif only_non:
            T_range = [(start_date, end_date, 'white')]
        else:
            T_range = [(start_date, end_date, 'gray')]

        return T_range
    

    # def threshold(self, axis_data):
    #     if axis_data.size == 0:
    #         raise ValueError("Input array is empty.")
        
    #     q25, q75 = np.percentile(axis_data, [25, 75])
    #     iqr = q75 - q25
    #     median_value = np.median(axis_data)
        
    #     multiplier = (q75 - median_value) / iqr if iqr != 0 else 1
    #     quantile_val = median_value + np.clip(multiplier, 0.1, 1.5) * (iqr / 2)
        
    #     return quantile_val

    def _freq_cut(self, data, low_cut=20000, high_cut=60000):
        # if data.ndim != 2:
        #     raise ValueError("Input must be a 2D array.")

        low_cut = low_cut
        high_cut = high_cut
        
        # h, _ = data.shape
        # unit = h / 192000
        # low_index, high_index = max(0, int(h - unit * high_cut)), min(h, int(h - unit * low_cut))
        
        # if low_index >= high_index:
        #     raise ValueError("Cut indices are out of bounds.")
        
        # return data[low_index:high_index, :]
        _ , dim = data.shape
        start_idx = int(low_cut / (384000 / 2) * dim)
        # data = np.array(data)[:, start_idx:]
        
        end_idx = int(high_cut / (384000 / 2) * dim)
        # data = np.array(data)[:, :end_idx]
        
        data = np.array(data)[:, start_idx:end_idx]
        return data

    def _rate_of_change_analysis(self, data):
        # cut_data = self._freq_cut(np.flip(data.T, axis=0),low_cut=self.low_cut, high_cut=self.high_cut)
        # abs_change = np.abs(np.diff(cut_data, axis=1))
        abs_change = np.abs(np.diff(np.flip(data.T, axis=0), axis=1))
        summary_value = np.mean([np.mean(abs_change, axis=1), np.median(abs_change, axis=1), np.std(abs_change, axis=1), np.max(abs_change, axis=1)], axis=0)
        return np.mean(summary_value)
    def interpolation(self, data):
        data = np.asarray(data)
        is_nested = data.ndim == 2

        if is_nested:
            return [self._interpolate_and_convolve(sensor_data) for sensor_data in data]
        return self._interpolate_and_convolve(data)


    def _interpolate_and_convolve(self, data):
        return np.convolve(data, np.ones(self.window_size) / self.window_size, mode='same')

    def processor(self, arr, multiview_Flag=False):
        min_max_values = [
            (2.462394344174288, 5.992459175876193), 
            (1.5785974723781264, 4.821355045390776)    
            ] if multiview_Flag else [(3.0840127322051862, 5.754039890453067)]
            
        if multiview_Flag:
            return [self.PLC_normalize(np.array(arr[i]), min_max_values[i]) for i in range(len(arr))]
        else:
            return self.PLC_normalize(np.array(arr), min_max_values[0])

    def PLC_normalize(self, data, min_max_values):
        data_min, data_max = min_max_values
        
        if data_min == data_max:
            raise ValueError("Normalization range is zero, cannot normalize.")
        
        normalized_data = ((data - data_min) / (data_max - data_min)) * (self.PLC_max - self.PLC_min) + self.PLC_min
        return normalized_data
    

    def timestamp_grouping(self, prob, setup, temp_threshold=4.4):
        upper_bound = temp_threshold
        
        start_time = datetime.strptime(setup.start_date, "%Y%m%d%H%M%S")
        end_time = datetime.strptime(setup.end_date, "%Y%m%d%H%M%S")
        total_seconds = int((end_time - start_time).total_seconds())
        ticks = np.arange(0, total_seconds, setup.cut_time)
        time_labels = [(start_time + timedelta(seconds=int(t+30))).strftime('%Y %m %d %H:%M:%S') for t in ticks]

        prob_1 = np.array(prob[1])
        group_idx_array = np.where(prob_1 >= upper_bound)[0]
        print(f"{group_idx_array = }")
        if group_idx_array.size == 0:
            return []

        split_points = np.where(np.diff(group_idx_array) > 1)[0] + 1
        print(f"{split_points = }")
        if len(split_points) > 0:
            grouped_indices = np.split(group_idx_array, split_points)
        else:
            grouped_indices = [group_idx_array]

        group_list = [[time_labels[spp[0]], time_labels[spp[-1]]] for spp in grouped_indices]
        
        return group_list
    

class FrequencyFeatures(Spectrum):

    def __init__(self):
        super(FrequencyFeatures, self).__init__()  # for python 2.x
        self.DATA_PATH = None
        self.ORIG_DATA, self.SAMPLING_RATE = None, None
        self.METHOD, self.FEATURE_NUM = None, None
        self.RESOLUTION = None
        self.RESOLUTION_BLOCK_WEIGHT = None
        self.ORIG_FEATURES, self.FREQUENCY_INFO = None, None

    def __call__(self, pseudo_spectrum, sr, method, feature_n, resolution=None):
        self.METHOD = method
        self.FEATURE_NUM = feature_n
        self.RESOLUTION = resolution
        self.SAMPLING_RATE = sr
        self.ORIG_DATA = pseudo_spectrum
        if self.METHOD in ['resolution', 'reverse_resolution']:
            self.get_split_weigth()
        generated_features, self.ORIG_FEATURES, self.FREQUENCY_INFO = self.feature_generator()
        return generated_features

    def get_split_weigth(self):

        resolution = self.RESOLUTION
        feature_n = self.FEATURE_NUM

        if feature_n - 1 == resolution:
            weight = np.ones(resolution)
        else:
            r = (feature_n - 1) % resolution
            m = (feature_n - 1) // resolution

            base_array = np.ones(resolution) * m

            if r >= 3:
                base_array_sub = np.ones(r)
                sub = np.copy(base_array_sub)
                sum_l = np.zeros(r) * m
                i = 1
                while True:
                    sub_mag = max(int(np.round(sub[0] / len(sub))), 1)
                    sum_l[i - 1] -= sub_mag
                    sum_l[-i] += sub_mag
                    sub = sub[1:-1]
                    i += 1
                    if len(sub) < 3:
                        break
                sum_l.sort()
                weight_sub = base_array_sub + sum_l
                base_array[-r:] += weight_sub
                weight = base_array

            else:
                if r == 0:
                    sub = np.copy(base_array)
                    sum_l = np.zeros(resolution) * m
                    i = 1
                    while True:
                        sub_mag = max(int(np.round(sub[0] / len(sub))), 1)
                        sum_l[i - 1] -= sub_mag
                        sum_l[-i] += sub_mag
                        sub = sub[1:-1]
                        i += 1
                        if len(sub) == 1:
                            break
                    sum_l.sort()
                    weight = base_array + sum_l

                else:
                    for idx in range(r):
                        base_array[-(idx + 1)] += 1
                    weight = base_array
        self.RESOLUTION_BLOCK_WEIGHT = weight.astype('int')

    def feature_generator(self):
        orig_data = np.reshape(self.ORIG_DATA, -1)
        features = []
        frequency_info = []

        if self.METHOD == 'equal':
            new_feature = []
            interval = int(len(orig_data) / (self.FEATURE_NUM))

            for ite in range(0, len(orig_data), interval):

                remain_len = len(orig_data[ite + interval:])

                if remain_len >= interval:
                    data_seg = orig_data[ite: ite + interval]

                    freq_s = ite / len(orig_data) * self.SAMPLING_RATE / 2
                    freq_e = (ite + interval) / len(orig_data) * self.SAMPLING_RATE / 2
                    frequency_info.append([freq_s, freq_e])
                    new_feature.append(np.sum(data_seg) / len(data_seg))
                    features.append(data_seg)
                else:
                    data_seg = orig_data[ite:]
                    freq_s = ite / len(orig_data) * self.SAMPLING_RATE / 2
                    freq_e = self.SAMPLING_RATE / 2
                    frequency_info.append([freq_s, freq_e])

                    new_feature.append(np.sum(data_seg) / len(data_seg))
                    features.append(data_seg)
                    break

        else:
            new_feature = []
            audible_last_idx = int(np.round((20000 / (self.SAMPLING_RATE / 2)) * len(orig_data)))

            new_feature.append(np.sum(orig_data[:audible_last_idx]) / audible_last_idx)

            freq_s = 0
            freq_e = (audible_last_idx) / len(orig_data) * self.SAMPLING_RATE / 2
            frequency_info.append([freq_s, freq_e])

            features.append(orig_data[:audible_last_idx])

            ultra_data = orig_data[audible_last_idx:]

            feature_n_remain = self.FEATURE_NUM - 1

            if self.METHOD == 'u_equal':

                interval = int(len(ultra_data) / (feature_n_remain))

                for ite in range(0, len(ultra_data), interval):

                    remain_len = len(ultra_data[ite + interval:])

                    if remain_len >= interval:
                        data_seg = ultra_data[ite: ite + interval]

                        freq_s = (ite + audible_last_idx) / len(orig_data) * self.SAMPLING_RATE / 2
                        freq_e = (ite + interval + audible_last_idx) / len(orig_data) * self.SAMPLING_RATE / 2
                        frequency_info.append([freq_s, freq_e])

                        new_feature.append(np.sum(data_seg) / len(data_seg))
                        features.append(data_seg)

                    else:
                        data_seg = ultra_data[ite:]

                        freq_s = (ite + audible_last_idx) / len(orig_data) * self.SAMPLING_RATE / 2
                        freq_e = self.SAMPLING_RATE / 2
                        frequency_info.append([freq_s, freq_e])

                        new_feature.append(np.sum(data_seg) / len(data_seg))
                        features.append(data_seg)

                        break

            elif self.METHOD == 'resolution':

                try:
                    block_split_info = self.RESOLUTION_BLOCK_WEIGHT
                except:
                    print(
                        f'Resolution {self.RESOLUTION} is wrong value, Resolution is must larger than 1 and not larger than Feature Number')
                    raise ValueError
                block_len = int(len(ultra_data) / self.RESOLUTION)
                ultra_new_feature = []

                for r in range(self.RESOLUTION):

                    if r == self.RESOLUTION - 1:
                        block = ultra_data[r * block_len:]
                    else:
                        block = ultra_data[r * block_len:(1 + r) * block_len]

                    feature_freq_s = audible_last_idx + (r * block_len)

                    block = np.reshape(block[:int(len(block) / block_split_info[r]) * block_split_info[r]],
                                       (block_split_info[r], -1))

                    for b_l in range(block_split_info[r]):
                        freq_s = feature_freq_s / len(orig_data) * self.SAMPLING_RATE / 2
                        freq_e = min((feature_freq_s + len(block[b_l])) / len(orig_data) * self.SAMPLING_RATE / 2,
                                     self.SAMPLING_RATE / 2)

                        frequency_info.append([freq_s, freq_e])

                        features.append(block[b_l])
                        feature = np.sum(block[b_l]) / len(block[b_l])
                        ultra_new_feature.append(feature)

                        feature_freq_s += len(block[b_l])

                new_feature = np.append(new_feature, ultra_new_feature)

            elif self.METHOD == 'reverse_resolution':

                try:
                    block_split_info = np.flip(self.RESOLUTION_BLOCK_WEIGHT)
                except:
                    print(
                        f'Resolution {self.RESOLUTION} is wrong value, Resolution is must larger than 1 and not larger than Feature Number')
                    raise ValueError

                block_len = int(len(ultra_data) / self.RESOLUTION)
                ultra_new_feature = []

                for r in range(self.RESOLUTION):

                    if r == self.RESOLUTION - 1:
                        block = ultra_data[r * block_len:]
                    else:
                        block = ultra_data[r * block_len:(1 + r) * block_len]

                    feature_freq_s = audible_last_idx + (r * block_len)
                    block = np.reshape(block[:int(len(block) / block_split_info[r]) * block_split_info[r]],
                                       (block_split_info[r], -1))

                    for b_l in range(block_split_info[r]):
                        freq_s = feature_freq_s / len(orig_data) * self.SAMPLING_RATE / 2
                        freq_e = min((feature_freq_s + len(block[b_l])) / len(orig_data) * self.SAMPLING_RATE / 2,
                                     self.SAMPLING_RATE / 2)

                        frequency_info.append([freq_s, freq_e])

                        features.append(block[b_l])
                        feature = np.sum(block[b_l]) / len(block[b_l])
                        ultra_new_feature.append(feature)

                        feature_freq_s += len(block[b_l])

                new_feature = np.append(new_feature, ultra_new_feature)

            else:
                print('method is wrong')
                raise ValueError

        return new_feature, features, frequency_info

    def target_detail_feature(self, target_num, feature_n):
        data = self.ORIG_FEATURES[target_num]
        frequency_info = self.FREQUENCY_INFO[target_num]

        data = np.reshape(data, -1)

        new_feature = []
        interval = int(len(data) / (feature_n))

        for ite in range(0, len(data), interval):

            remain_len = len(data[ite + interval:])

            if remain_len >= interval:
                data_seg = data[ite: ite + interval]
                new_feature.append(np.sum(data_seg) / len(data_seg))
            else:
                data_seg = data[ite:]
                new_feature.append(np.sum(data_seg) / len(data_seg))
                break

        freq_array = (np.round(np.array(frequency_info) / 100) * 100).astype('int').astype(str)
        freq_array = list(map(lambda x: x + 'Hz', freq_array))

        return np.array(new_feature), freq_array
    


def extract_roi(data, method, hz):

    _, dim = np.shape(data)

    if method == 'lower':
        end_idx = int(hz / (384000 / 2) * dim)
        data = np.array(data)[:, :end_idx]

    elif method == 'upper':
        start_idx = int(hz / (384000 / 2) * dim)
        data = np.array(data)[:, start_idx:]

    elif method == 'full':
        data = np.array(data)

    elif method == 'band':
        start_idx = int(hz[0] / (384000 / 2) * dim)
        end_idx = int(hz[-1] / (384000 / 2) * dim)
        data = np.array(data)[:, start_idx:end_idx]

    else:
        print(f'Analysis method argument is wrong: {method} (lower/upper/full)')
        raise KeyError
    return data