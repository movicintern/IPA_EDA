import pandas as pd
from datetime import datetime, timedelta
import numpy as np


class IPA_Excel_Processor():
    def __init__(self, setup, excel_path, save_path, minutes_sync, seconds_sync, speed_command, organizer, step):
        self.excel_path = excel_path
        self.save_path = save_path
        self.minutes_sync = minutes_sync
        self.seconds_sync = seconds_sync
        self.step = step
        self.setup = setup
        
        self.organizer = organizer # prob_organizer
        self.speed_command = speed_command # V

    def load_excel(self):
        return pd.read_excel(self.excel_path, engine='openpyxl')


    def save_data(self, data):
        return data.to_excel(self.save_path, index=False)


    def data_filter(self , data):
        start_time = datetime.strptime(self.setup.start_date, "%Y%m%d%H%M%S")
        end_time = datetime.strptime(self.setup.end_date, "%Y%m%d%H%M%S")
        total_seconds = int((end_time - start_time).total_seconds())
        self.ticks = np.arange(0, total_seconds, self.setup.cut_time)

        start_date = datetime.strptime(self.setup.start_date, "%Y%m%d%H%M%S")
        standard_time = [(start_date + timedelta(seconds=int(tick) + 30) + timedelta(minutes=self.minutes_sync, seconds=self.seconds_sync)).strftime("%Y %m %d %H:%M:%S") for tick in self.ticks]

        standard_datetime = datetime.strptime(standard_time[0], "%Y %m %d %H:%M:%S")

        if data[data['시각'] == standard_datetime].empty:
            search_time = standard_time[0][:-2]
            filtered_df = data[data['시각'].astype(str).str.contains(search_time)]
            return filtered_df, standard_datetime

        else: return data[data['시각'] == standard_datetime], standard_datetime

        
    def find_time_index(self, filtered_data, data, standard_datetime):
        if not filtered_data.empty:
            diff1 = datetime.strptime(filtered_data.iloc[0]['시각'], "%Y %m %d %H:%M:%S") - standard_datetime
            diff2 = datetime.strptime(filtered_data.iloc[-1]['시각'], "%Y %m %d %H:%M:%S") - standard_datetime
            
            if diff1.total_seconds() - diff2.total_seconds() > 0:
                result_val = filtered_data.iloc[-1]['시각']
            else:
                result_val = filtered_data.iloc[0]['시각']

            result_index = filtered_data[data['시각'] == result_val].index.tolist()
            return result_index      
        else: 
            result_val = filtered_data.iloc[0]['시각']
            result_index = filtered_data[data['시각'] == result_val].index.tolist()
            return result_index


    def update_data(self):
        
        df = self.load_excel()

        filtered_df, standard_datetime = self.data_filter(df)
        print(filtered_df)
        
        result_index = self.find_time_index(filtered_df, df, standard_datetime)
        print(result_index)
        print(df.info())
        if result_index:
            if self.step:
                for i in range(len(self.speed_command[0])):
                    df.loc[result_index[0]+i*self.step, "[Pred_E#1]속도지령"], df.loc[result_index[0]+i*self.step, "[Pred_E#2]속도지령"] = self.speed_command[0][i], self.speed_command[1][i]
                    df.loc[result_index[0]+i*self.step, "prob_organizer_E#1"], df.loc[result_index[0]+i*self.step, "prob_organizer_E#2"] = self.organizer[0][i], self.organizer[1][i]
            else:
                for i in range(len(self.speed_command[0])):
                    df.loc[result_index[0]+i, "[Pred_E#1]속도지령"], df.loc[result_index[0]+i, "[Pred_E#2]속도지령"] = self.speed_command[0][i], self.speed_command[1][i]
                    df.loc[result_index[0]+i, "prob_organizer_E#1"], df.loc[result_index[0]+i, "prob_organizer_E#2"] = self.organizer[0][i], self.organizer[1][i]
        return self.save_data(df)