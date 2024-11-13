import argparse
import numpy as np
from tqdm import tqdm
from EDA_Setup import SETUP
from utils.Data_Loader import *
from Feature_Extractor import *
from utils.Data_Visualizer import *
from concurrent.futures import ProcessPoolExecutor
from utils.Excel_processor import IPA_Excel_Processor


def process_sensor_data(sensor_name, data_extractor, setup, spectrogram, tracker, idx, blue="\033[34m", reset="\033[0m"):

    spectro_set = []
    date_info_list = []


    prob_set = [[] for _ in range(len(setup.target_sensors))] if setup.multi_sensor_view else []

    for d in tqdm(range(len(data_extractor)), desc=f"{blue}[Processing Data] {sensor_name}{reset}",  bar_format=f"{blue}{{l_bar}}{{bar}}{reset}{{r_bar}}"):
        data, date_info = data_extractor[d]

        spectro = spectrogram(data, sr=setup.sampling_rate, fft_size=4096, hop_size=2048)
        prob = tracker._rate_of_change_analysis(spectro)
       
        date_info_list.append(date_info[0])

        if not setup.multi_sensor_view:
            plot_spectrogram(sensor_name, spectro, setup, date_info)
            prob_set.append(prob)
        else:
            spectro_set.append(spectro)
            prob_set[idx].append(prob)
            
    return sensor_name, spectro_set, prob_set, date_info_list


def main():
    yellow = "\033[33m"
    reset = "\033[0m"
    red = "\033[31m"

    parser = argparse.ArgumentParser(description="[EDA]-Arguments")
    parser.add_argument('--data_abs_dir', type=str, required=True, help='Data path')
    parser.add_argument('--target_sensors', type=str, nargs='+', required=True, help='Target sensors')
    parser.add_argument('--start_date', type=str, required=True, help='Start date')
    parser.add_argument('--end_date', type=str, required=True, help='End date')
    parser.add_argument('--cut_time', type=int, required=False, help='cut time')
    parser.add_argument('--extend', type=str, required=False, help='time_extension')

    args = parser.parse_args()
    
    
    spectrogram = Spectrogram()
    tracker = RunDownTracker(tot_sensors=len(args.target_sensors))
    #spectrum = Spectrum()
    #plot_tracker = TrackerVisualizer()

    #####--???
    start_date, end_date = (tracker.time_range_extention(args.start_date, args.end_date, minutes=0, extend=False) 
                        if args.extend else (args.start_date, args.end_date))

        
    cut_time = args.cut_time if args.cut_time else 3

    setup = SETUP(
        data_abs_dir=args.data_abs_dir,
        target_sensors=args.target_sensors,
        start_date=start_date,
        end_date=end_date,
        cut_time=cut_time,
    )
    
    setup.apply()

    data_extractors = [
        DataExtractor(
            abs_dir=data_dir_path,
            start_date=setup.start_date,
            end_date=setup.end_date,
            record_time=setup.record_time,
            sampling_rate=setup.sampling_rate,
            cut_time=setup.cut_time
        ) for data_dir_path in setup.data_dir_path
    ]


    with ProcessPoolExecutor() as executor:
        futures = []
        for idx, sensor_name in enumerate(setup.target_sensors):
            futures.append(executor.submit(
                process_sensor_data,
                sensor_name,
                data_extractors[idx],
                setup,
                spectrogram,
                tracker,
                idx,
            ))

        results = [future.result() for future in futures]

    # if setup.multi_sensor_view:
    #     spectro_sets = [result[1] for result in results]
    #     date_info_lists = [result[3] for result in results]  
    #     num_samples = len(spectro_sets[0])  

    #     for i in tqdm(
    #         range(num_samples), 
    #         desc=f"{yellow}Extracting Spectrogram{reset}", 
    #         bar_format=f"{yellow}{{l_bar}}{{bar}}{reset}{{r_bar}}"
    #     ):
    #         current_slice = [spectro_set[i] for spectro_set in spectro_sets]
    #         current_date_info = [date_info_list[i] for date_info_list in date_info_lists] 
    #         print(current_slice)
    #         print()
    #         multiple_plot_spectrogram(setup.target_sensors, current_slice, setup, current_date_info[0])

    prob_organizer = [[] for _ in range(len(setup.target_sensors))] if setup.multi_sensor_view else [[]]
    Flag = setup.multi_sensor_view


    for idx, (sensor_name, _, sensor_prob_set, _) in tqdm(
        enumerate(results), 
        total=len(results), 
        desc=f"{red}Processing Tracker{reset}",  
        bar_format=f"{red}{{l_bar}}{{bar}}{reset}{{r_bar}}"
    ):
        if setup.multi_sensor_view:
            prob_organizer[idx] = sensor_prob_set[idx]  
        else:
            prob_organizer = sensor_prob_set  

        print(f"Complete Processing sensor: {sensor_name}\n")  
    
    speed_command = tracker.processor(prob_organizer, multiview_Flag=Flag)
 
 
    # plot_tracker.visualize(args.target_sensors, V, setup, 
    #     T=tracker.run_down_classification(args.start_date, args.end_date, Is_init=True, only_non=False),  
    #     multiview_Flag=Flag
    # )



    Excel_info = {}
    machine = os.path.basename(args.data_abs_dir)

    names = ['1_motor-50kt', '3_motor-10kt', '2_waterpump', '4_watergate']
    syncs = [[43, 30], [22, 0], [41, 0], [41, 0]]

    for name, sync in zip(names, syncs):
        Excel_info[name] = {
            'sync': sync,
            'path': f'Excel_dir/base_{name}.xlsx'
        }

    excel_path = Excel_info[f'{machine}']['path']
    excel_sync = Excel_info[f'{machine}']['sync']
    
    to_excel = IPA_Excel_Processor(
        excel_path=excel_path,
        save_path="/home/movic/Excel/PLC_data/plc_10.xlsx",
        minutes_sync=excel_sync[0],
        seconds_sync=excel_sync[1],
        speed_command=np.array(speed_command),
        setup=setup,
        organizer=prob_organizer,
        step=args.step
        )
    to_excel.update_data()
    
    print(f"Operation Intervals: {tracker.timestamp_grouping(prob_organizer, setup)}")

if __name__ == '__main__':
    main()
