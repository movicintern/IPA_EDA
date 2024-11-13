import os

class SETUP:

    def __init__(self, data_abs_dir, target_sensors, start_date, end_date, cut_time):

        self.data_abs_dir = data_abs_dir 
        self.target_sensors = target_sensors

        self.resul_abs_dir = "/home/mg/.CODE/IPA/IPA_EDA/Results"

        self.data_dir_path = [os.path.join(self.data_abs_dir, target_sensor) for target_sensor in self.target_sensors]
        
        self.start_date = start_date
        self.end_date = end_date
        self.record_time = 3
        self.sampling_rate = 384000
        self.cut_time = cut_time

        self.facility = self.data_abs_dir.split('/')[-1]

        self.frequency_feature_n = 10

        self.multi_sensor_view = False if len(self.target_sensors) == 1 else True

        # self.save_result_path = os.path.join(self.resul_abs_dir, f'({self.facility})Date_{self.start_date}~{self.end_date}',
        #                                      f'data_length_{self.cut_time}sec')
        
        self.save_result_path = os.path.join(self.resul_abs_dir, '4_watergate', f'Date_{(self.data_abs_dir.split("/")[-1])}_{self.start_date}~{self.end_date}',
                                             f'data_length_{self.cut_time}sec')

        self.spectrogram_path = os.path.join(self.save_result_path, 'img', 'spectrogram')
        self.spectrum_path = os.path.join(self.save_result_path, 'img', 'spectrum')
        self.pseudo_spectrum_path = os.path.join(self.save_result_path, 'img', 'pseudo_spectrum')
        self.frequency_feature_path = os.path.join(self.save_result_path, 'trend', 'frequency_feature')
        self.operation_path = os.path.join(self.save_result_path, 'img', 'operation')

    def apply(self):

        if self.multi_sensor_view:
            os.makedirs(os.path.join(self.spectrogram_path, 'multi_sensor'), exist_ok=True)
            # os.makedirs(os.path.join(self.spectrum_path, 'multi_sensor'), exist_ok=True)
            # os.makedirs(os.path.join(self.pseudo_spectrum_path, 'multi_sensor'), exist_ok=True)
            # os.makedirs(os.path.join(self.frequency_feature_path, 'multi_sensor'), exist_ok=True)
            # os.makedirs(os.path.join(self.operation_path, 'multi_sensor'), exist_ok=True)

        else:
            for sensor_name in self.target_sensors:
                os.makedirs(os.path.join(self.spectrogram_path, sensor_name), exist_ok=True)
                # os.makedirs(os.path.join(self.spectrum_path, sensor_name), exist_ok=True)
                # os.makedirs(os.path.join(self.pseudo_spectrum_path, sensor_name), exist_ok=True)
                # os.makedirs(os.path.join(self.frequency_feature_path, sensor_name), exist_ok=True)
                # os.makedirs(os.path.join(self.operation_path, sensor_name), exist_ok=True)

   