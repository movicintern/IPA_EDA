import os
import numpy as np
import pickle
import json
import soundfile as sf
import random
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
# import time
# from functools import wraps
#
# def timeit(func):
#     @wraps(func)
#     def timeit_wrapper(*args, **kwargs):
#         start_time = time.perf_counter()
#         result = func(*args, **kwargs)
#         end_time = time.perf_counter()
#         total_time = end_time - start_time
#         print(f'Function {func.__name__}{args} {kwargs} / {total_time:.3f} seconds')
#         return result
#     return timeit_wrapper

class FileRW:

    def __init__(self) -> None:
        pass

    @staticmethod
    def file_reader(path, file_type):

        if file_type == 'wav':
            data, sr = sf.read(path)
            return data, sr

        elif file_type == 'txt':
            with open(path, "rb") as txt_fp:
                txt = pickle.load(txt_fp)
            return txt

        elif file_type == 'pkl':
            pkl = pickle.load(open(path, 'rb'))
            return pkl

        elif file_type == 'json':
            with open(path, "r") as js_fp:
                json_f = json.load(js_fp)
            return json_f

        else:
            print(f'file type is wrong: {file_type}')

    @staticmethod
    def file_writer(data, path, file_type):

        if file_type == 'txt':
            with open(path, "wb") as txt_fp:
                pickle.dump(data, txt_fp)

        elif file_type == 'pkl':
            pickle.dump(data, open(path, 'wb'))

        elif file_type == 'json':
            with open(path, 'w') as js:
                json.dump(data, js)

        else:
            print(f'file type is wrong: {file_type}')
            exit()


class FileExtractor:

    def __init__(self, abs_dir):
        self.abs_dir = abs_dir
    
    def file_inspect(self, file_name):

        file_path = os.path.join(self.abs_dir, file_name[:-8], file_name)

        if os.path.isfile(file_path):
            return True
        else:
            return False
    
    @staticmethod
    def target_file_track(date_list, bottom, top):

        number_date_list = [int(s.split('.')[0]) for s in date_list]
        zips = [number_date_list, date_list]
        zips = np.transpose(zips)
        zips = np.array(sorted(zips, key=lambda x: x[0]))

        sorted_data_list = zips[:, 1]
        sorted_number_date_list = np.array(zips[:, 0]).astype('int')
        
        sorted_number_date_list[sorted_number_date_list < bottom] = 0
        sorted_number_date_list[sorted_number_date_list >= top] = 0

        target_idx = sorted(np.nonzero(sorted_number_date_list)[0])

        if len(target_idx) == 0:
            file_list = []
        else:
            file_list = sorted_data_list[target_idx]

        return file_list
    
    def file_path_extract(self, date_list, data_num, data_sampling):
        target_list = []
        date_start, date_end = str(date_list[0]), str(date_list[1])

        folder_list = []
        target_dir_start, target_dir_end = date_start[:-4], date_end[:-4]

        target_inner_dir = np.array(sorted([path for path in os.listdir(self.abs_dir) if path.isdigit()])).astype('int')
        target_inner_dir = target_inner_dir[
            np.where(np.logical_and(target_inner_dir > int(target_dir_start), target_inner_dir < int(target_dir_end)))[
                0]]
        target_inner_dir = target_inner_dir.astype('int')
        target_inner_dir.sort()
        target_inner_dir = target_inner_dir.astype('str')

        if target_dir_start in os.listdir(self.abs_dir):
            folder_list = np.append(folder_list, target_dir_start)
        folder_list = np.append(folder_list, target_inner_dir)
        if target_dir_end in os.listdir(self.abs_dir):
            folder_list = np.append(folder_list, target_dir_end)

        folder_list = list(set(folder_list))

        if len(folder_list) == 0:
            return [], []

        with concurrent.futures.ProcessPoolExecutor(max_workers=max(os.cpu_count() - 2, 1)) as executor:
            future_to_folder = {executor.submit(self.process_folder, folder): folder for folder in folder_list}

            for future in concurrent.futures.as_completed(future_to_folder):
                target_list_set = future.result()
                target_list.extend(target_list_set)

        if len(target_list) == 0:
            return [], []
        else:
            files_list = self.target_file_track(target_list, int(date_start), int(date_end))

            files_list = list(set(files_list))
            files_list = sorted(files_list)

            if data_sampling and (len(files_list) > data_num):
                path_list = []
                dummy_files_list = []
                for f in np.linspace(0, len(files_list) - 1, data_num).astype('int'):
                    file = files_list[f]
                    file_path = os.path.join(self.abs_dir, file[:-8], file)
                    path_list.append(file_path)
                    dummy_files_list.append(file)
                files_list = dummy_files_list
            else:
                path_list = [os.path.join(self.abs_dir, file[:-8], file) for file in files_list]

            return path_list, files_list

    def process_folder(self, folder):
        return os.listdir(os.path.join(self.abs_dir, folder))
    
    # def file_path_extract(self, date_list, data_num, data_sampling):

    #     target_list = []
    #     date_start, date_end = str(date_list[0]), str(date_list[1])

    #     folder_list = []
    #     target_dir_start, target_dir_end = date_start[:-4], date_end[:-4]

    #     target_inner_dir = np.array(sorted([path for path in os.listdir(self.abs_dir) if path.isdigit()])).astype('int')
    #     target_inner_dir = target_inner_dir[
    #         np.where(np.logical_and(target_inner_dir > int(target_dir_start), target_inner_dir < int(target_dir_end)))[
    #             0]]
    #     target_inner_dir = target_inner_dir.astype('int')
    #     target_inner_dir.sort()
    #     target_inner_dir = target_inner_dir.astype('str')

    #     if target_dir_start in os.listdir(self.abs_dir):
    #         folder_list = np.append(folder_list, target_dir_start)
    #     folder_list = np.append(folder_list, target_inner_dir)
    #     if target_dir_end in os.listdir(self.abs_dir):
    #         folder_list = np.append(folder_list, target_dir_end)

    #     folder_list = list(set(folder_list))

    #     if len(folder_list) == 0:
    #         return [], []
        
    #     else:
    #         for folder in folder_list:
    #             target_list_set = os.listdir(os.path.join(self.abs_dir, folder))
    #             target_list = np.append(target_list, target_list_set)

    #         if len(target_list) == 0:
    #             return [], []
    #         else:
    #             files_list = self.target_file_track(target_list, int(date_start), int(date_end))

    #             files_list = list(set(files_list))
    #             files_list = sorted(files_list)

    #             if data_sampling and (len(files_list) > data_num):
    #                 path_list = []
    #                 dummy_files_list = []
    #                 for f in np.linspace(0, len(files_list)-1, data_num).astype('int'):
    #                     file = files_list[f]
    #                     file_path = os.path.join(self.abs_dir, file[:-8], file)
    #                     path_list.append(file_path)
    #                     dummy_files_list.append(file)
    #                 files_list = dummy_files_list
    #             else:
    #                 path_list = [os.path.join(self.abs_dir, file[:-8], file) for file in files_list]

    #             return path_list, files_list
            
class TimeSynchronizer(FileExtractor):

    def __init__(self, abs_dir):
        super(TimeSynchronizer, self).__init__(abs_dir=abs_dir)
    
    def target_time_correct(self, start_date, end_date, record_time):

        path_info, file_name_info = self.file_path_extract(date_list=[start_date, end_date], data_num=np.inf, data_sampling=False)
        
        if len(file_name_info) == 0:

            file_check_start = start_date - record_time
            path_info, file_name_info = self.file_path_extract(date_list=[file_check_start, end_date], data_num=np.inf, data_sampling=False)

            if len(file_name_info) == 0:
                time_correction = None
            else:
                ref_file_info = np.transpose([path_info, file_name_info])
                ref_file_info = np.array(sorted(ref_file_info, key=lambda x: x[1]))

                ref_start_date, file_format = ref_file_info[0, 1].split('.')
                ref_end_date = ref_file_info[-1, 1].split('.')[0]

                ref_start_date_sec = self.str2time(ref_start_date)
                ref_end_date_sec = self.str2time(ref_end_date)

                start_date_sec = self.str2time(str(start_date))
                end_date_sec = self.str2time(str(end_date))

                time_correction = [int(ref_start_date_sec-start_date_sec), int(end_date_sec-ref_end_date_sec)] 
            
        else:
            ref_file_info = np.transpose([path_info, file_name_info])
            ref_file_info = np.array(sorted(ref_file_info, key=lambda x: x[1]))

            ref_start_date, file_format = ref_file_info[0, 1].split('.')
            ref_end_date = ref_file_info[-1, 1].split('.')[0]

            ref_start_date_sec = self.str2time(ref_start_date)
            ref_end_date_sec = self.str2time(ref_end_date)

            start_date_sec = self.str2time(str(start_date))
            end_date_sec = self.str2time(str(end_date))

            if start_date_sec > (ref_start_date_sec-record_time) and start_date_sec < ref_start_date_sec:
                
                target_file_date_start = int(self.time2datetime(ref_start_date_sec-record_time))

                if self.file_inspect(file_name=f'{target_file_date_start}.{file_format}'):
                    print('!!')
                    # start_date_sec = ref_start_date_sec-record_time
                    path_info, file_name_info = self.file_path_extract(date_list=[target_file_date_start, end_date], data_num=np.inf, data_sampling=False)
                    # print(file_name_info, int(self.time2datetime(start_date_sec)))
                    ref_file_info = np.transpose([path_info, file_name_info])
                    ref_file_info = np.array(sorted(ref_file_info, key=lambda x: x[1]))

                    ref_start_date, file_format = ref_file_info[0, 1].split('.')
                    ref_end_date = ref_file_info[-1, 1].split('.')[0]

                    ref_start_date_sec = self.str2time(ref_start_date)
                    ref_end_date_sec = self.str2time(ref_end_date)
            else:
                target_file_date_start = start_date

            if end_date_sec < (ref_end_date_sec+record_time) and end_date_sec > ref_end_date_sec and ref_end_date_sec != ref_start_date_sec:
                
                target_file_date_end = int(self.time2datetime(ref_end_date_sec+record_time))

                if self.file_inspect(file_name=f'{target_file_date_end}.{file_format}'):
                    # start_date_sec = ref_start_date_sec-record_time
                    path_info, file_name_info = self.file_path_extract(date_list=[target_file_date_start, target_file_date_end], data_num=np.inf, data_sampling=False)
                    ref_file_info = np.transpose([path_info, file_name_info])
                    ref_file_info = np.array(sorted(ref_file_info, key=lambda x: x[1]))

                    ref_start_date, file_format = ref_file_info[0, 1].split('.')
                    ref_end_date = ref_file_info[-1, 1].split('.')[0]

                    ref_start_date_sec = self.str2time(ref_start_date)
                    ref_end_date_sec = self.str2time(ref_end_date)

            else:
                target_file_date_end = end_date
            
            time_correction = [int(ref_start_date_sec-start_date_sec), int(end_date_sec-ref_end_date_sec)] 
        #print(time_correction)

        return path_info, file_name_info, time_correction
    
    @staticmethod
    def str2time(date_info_str):
        date_info_datetime = datetime.strptime(date_info_str, '%Y%m%d%H%M%S')
        timestamp = datetime.timestamp(date_info_datetime)
        return timestamp
    
    @staticmethod
    def time2datetime(timestamp):
        return datetime.fromtimestamp(timestamp).strftime('%Y%m%d%H%M%S')
    

class DataExtractor(TimeSynchronizer):

    def __init__(self, abs_dir, start_date, end_date, record_time, sampling_rate, cut_time, random_load=False):
        super(DataExtractor, self).__init__(abs_dir=abs_dir)
        self.abs_dir = abs_dir
        self.file_rw = FileRW()
        self.data_length = sampling_rate * record_time
        self.start_date = start_date
        self.end_date = end_date
        self.record_time = record_time
        self.sampling_rate = sampling_rate
        self.cut_time = cut_time
        self.random_load = random_load
        self.split_num = None
        self.random_idx = None
        self.split_data_sequence_load_info()

    def __len__(self):
        return self.split_num

    def __getitem__(self, idx):

        if self.random_load:
            idx = self.random_idx[idx]

        start_date_n = self.str2time(str(self.start_date)) + (self.cut_time * idx)
        end_date_n = start_date_n + self.cut_time

        start_date_n = int(self.time2datetime(start_date_n))
        end_date_n = int(self.time2datetime(end_date_n))

        split_data = self.data_sequence_load(start_date_n, end_date_n)

        return split_data, [start_date_n, end_date_n]

    def random_shuffle_idx(self):
        random_idx = list(range(self.split_num))
        random.shuffle(random_idx)
        self.random_idx = random_idx

    def data_load(self, file_path):
        return self.file_rw.file_reader(file_path, 'wav')

    def data_sequence_load(self, start_date, end_date):
        path_info, file_name_info, time_correction = self.target_time_correct(start_date, end_date, self.record_time)

        if time_correction is None:
            target_time_length = int(self.str2time(str(end_date)) - self.str2time(str(start_date)))
            data_sequence_total = [0] * target_time_length * self.sampling_rate

        else:
            data_sequence_total = []
            data_sequence = []

            before_file_name = None

            if (len(file_name_info) ==1) and (time_correction[0] > 1) and (time_correction[1] == 0):
                _, sr = self.data_load(path_info[0])
                pass
            else:
                
                for f in range(len(file_name_info)):

                    data, sr = self.data_load(path_info[f])
                    
                    if sr != self.sampling_rate:
                        print(f'Data Sampling Rate({sr}) is not matched with setup({self.sampling_rate})')
                        raise ValueError
                    
                    if f == 0:
              
                        if time_correction[0] <= 0:

                            if len(data) < self.data_length:
                                
                                data_s = data[int(abs(time_correction[0]) * sr):]
                                data_sequence = np.append(data_sequence, data_s)
                                data_sequence = np.append(data_sequence, [0]*(self.data_length - len(data)))
                            else:
                                if time_correction[1] <= 0:
                                    data_s = data[int(abs(time_correction[0]) * sr):self.data_length]
                                    data_sequence = np.append(data_sequence, data_s)
                                else:
                                    if len(file_name_info) != 1 :
                                        data_s = data[int(abs(time_correction[0]) * sr):self.data_length]
                                        data_sequence = np.append(data_sequence, data_s)
                                    else:
                                        data_s = data[int(abs(time_correction[0]) * sr):int(abs(time_correction[1]) * sr)]
                                        data_sequence = np.append(data_sequence, data_s)

                        else:

                            if (len(file_name_info) ==1) and (time_correction[1] != 0):
                                data_sequence = np.append(data_sequence, data[:(self.record_time-time_correction[1])*sr])
                            else:
                                if len(data) < self.data_length:
                                    data_sequence = np.append(data_sequence, data)
                                    data_sequence = np.append(data_sequence, [0]*(self.data_length - len(data)))
                                else:
                                    data_sequence = np.append(data_sequence, data[:self.data_length])
                            
                        before_file_name = file_name_info[f]
                        continue

                    elif f == (len(file_name_info)-1):

                        if (self.record_time > time_correction[1]) and (time_correction[1] >= 0):
                            
                            if (time_correction[1] > 0):
                                data_s = data[:int(abs(time_correction[1]) * sr)]
                                data_sequence = np.append(data_sequence, data_s)
                            else:
                                pass

                        else:
                            if len(data) < self.data_length:
                                data_sequence = np.append(data_sequence, data)
                                data_sequence = np.append(data_sequence, [0]*(self.data_length - len(data)))
                            else:
                                data_sequence = np.append(data_sequence, data[:self.data_length])
                    else:
                        if sr != self.sampling_rate:
                            print(f'[{file_name_info[f]}] Sampling Rate is changed')
                            raise ValueError

                        time_stamp = self.str2time(file_name_info[f].split('.')[0]) - self.str2time(before_file_name.split('.')[0])
                        if time_stamp == self.record_time:

                            if len(data) < self.data_length:
                                data_sequence = np.append(data_sequence, data)
                                data_sequence = np.append(data_sequence, [0]*(self.data_length - len(data)))
                            else:
                                data_sequence = np.append(data_sequence, data[:self.data_length])

                        else:
                            if time_stamp < self.record_time:
                                print(f'[{file_name_info[f]}] File timestamp is changed')
                                print(f'Setup Time: {self.record_time} / File Time: {time_stamp}')
                                raise ValueError
                            else:
                                data_sequence = np.append(data_sequence, [0] * time_stamp)
                                data_sequence = np.append(data_sequence, data)

                        before_file_name = file_name_info[f]

            if time_correction[0] > 0:
                correction = [0] * sr * time_correction[0]
                data_sequence_total = np.append(data_sequence_total, correction)


            data_sequence_total = np.append(data_sequence_total, data_sequence)

            if time_correction[1] > self.record_time:
                correction = [0] * sr * (time_correction[1]-self.record_time)
                data_sequence_total = np.append(data_sequence_total, correction)

        return np.array(data_sequence_total)

    def split_data_sequence_load_info(self):

        target_time_length = int(self.str2time(str(self.end_date)) - self.str2time(str(self.start_date)))
        self.split_num = int(target_time_length/self.cut_time)
        remain_time = int(target_time_length - self.split_num*self.cut_time)

        if remain_time != 0:
            self.split_num += 1

        self.random_shuffle_idx()

