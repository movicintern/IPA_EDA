from utils.Data_Loader import *
from Feature_Extractor import *
from utils.Data_Visualizer import *
from tqdm import tqdm
import numpy as np
import time
from EDA_Setup import SETUP
import pandas as pd
from datetime import datetime
import argparse

def main():


    parser = argparse.ArgumentParser(description="[EDA]-Arguments")
    parser.add_argument('--data_abs_dir', type=str, required=True, help='data path')
    parser.add_argument('--target_sensors', type=str,  nargs='+', required=True, help='target sensors')
    parser.add_argument('--start_date', type=str, required=True, help='start date')
    parser.add_argument('--end_date', type=str, required=True, help='end date')
    parser.add_argument('--cut_time', type=int, required=False, help='cut time')


    args = parser.parse_args()
    spectrogram = Spectrogram()
    tracker = RunDownTracker(tot_sensors = len(args.target_sensors))
    spectrum = Spectrum()
    plot_tracker = TrackerVisualizer()
    
    
    cut_time = args.cut_time if args.cut_time else 30

    setup = SETUP(
        data_abs_dir=args.data_abs_dir,
        target_sensors=args.target_sensors,
        start_date=args.start_date,
        end_date=args.end_date,
        cut_time=cut_time,
    )
    
    setup.apply()
    data_extractors = [DataExtractor(abs_dir=data_dir_path, start_date=setup.start_date, end_date=setup.end_date,
                                   record_time=setup.record_time, sampling_rate=setup.sampling_rate,
                                   cut_time=setup.cut_time) for data_dir_path in setup.data_dir_path]

    spectrogram = Spectrogram()
    spectrum = Spectrum()
    
    sensors_trend_data = {sensor: {'max_db': [], 'features': [], 'date_info': []} for sensor in setup.target_sensors}
    prob_set = []   
    for d in tqdm(range(len(data_extractors[0])), desc='[EDA]'):

        prob_set = []   
        spectro_set = []
        pseudo_spec_set =[]
        date_info = None

        for idx in range(len(setup.target_sensors)):

            sensor_name = setup.target_sensors[idx]
            data_extractor = data_extractors[idx]
     
            data, date_info = data_extractor[d]

            spectro = spectrogram(data, sr = setup.sampling_rate, fft_size = 4096,  hop_size = 2048)
            pseudo_spec = spectrum.pseudo_spectrum(spectro)
            prob = tracker._rate_of_change_analysis(spectro)


            if not setup.multi_sensor_view:
                plot_spectrogram(sensor_name, spectro, setup, date_info)
                #plot_pseudo_spectrum(sensor_name, pseudo_spec, setup, date_info)

            else:
                spectro_set.append(spectro)
                #pseudo_spec_set.append(pseudo_spec)
                prob_set.append(prob)

 
        if setup.multi_sensor_view:
            multiple_plot_spectrogram(setup.target_sensors, spectro_set, setup, date_info)
            # multiple_plot_pseudo_spectrum(setup.target_sensors, pseudo_spec_set, setup, date_info)

    # Flag = True if setup.multi_sensor_view else False
    # V = tracker.processor(prob_set, args.target_sensors, multiview_Flag = Flag)
    # plot_tracker.visualize(args.target_sensors, V, setup, T=tracker.get_T_range(args.start_date, args.end_date, Is_init=False),  multiview_Flag = Flag)


if __name__ == '__main__':
    main()

