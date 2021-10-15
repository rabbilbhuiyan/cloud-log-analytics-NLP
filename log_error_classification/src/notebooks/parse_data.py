# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/preparation//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# importing the module and class
from get_data import DataLoader

# to ingnore the warning 
import warnings
warnings.filterwarnings('ignore')

# +
# creating the instance of class
loaderInstance = DataLoader()

# calling the instance of class
data = loaderInstance.load_data()

# print the output
#data.head(2)

# +
import pandas as pd
import datetime
import numpy as np
import os
import re



PROCESSED_DATA_PATH = r'C:\Users\rbhuiyan\Desktop\log_classification\log_error_classification\data\processed'

class DataParser():

    def __init__(self):
        #self.input_dir = input_dir
        #self.load_data()
        self.PROCESSED_DATA_PATH = PROCESSED_DATA_PATH
        self.parsed_data = None
        #self.timestamp_data = None
        self.parse_data()
        self.create_label()
        self.store_data()
        #self.timestamp()
        #self.create_label()
        #pass
    
        

    #Display and extract the features from the log data
    def parse_data(self) ->pd.DataFrame:
        '''
        data parsing.
        Args:
            df {pandas.DataFrame}: dataset
        Return:
            pandas.Dataframe: updated data with new features
        '''
        
        # copy the dataframe
        parsed_data = data.copy()
        
        # spliting the data into variables
        parsed_data[['timestamp', 'server','component','date', 'time', 'customer_ID','event', 'nova_compute', 'log_message']] = parsed_data['Column'].str.split(' ', 8, expand=True)
        
        # sub dataframe
        parsed_data = parsed_data[['server','component','date', 'time', 'customer_ID', 'nova_compute', 'log_message']]
        
        # extracting the info within the square bracket
        parsed_data['request_ID'] = parsed_data['log_message'].str.extract('\[(.*?)\]', expand=False).str.strip()
        
        # removing the square bracket and the contents within the square bracket
        parsed_data['log_message'] = parsed_data['log_message'].str.replace(r'\[.*?\]','')

        # Taking only first 100 characthers of the log message
        parsed_data['log_message'] = parsed_data['log_message'].str[:100]

        # stripping off everything after #
        parsed_data['log_message'] = parsed_data['log_message'].str.split('#').str[0]
        
        ##########################
        # deleting the non-date values: first make the non-dates to NaT value, then apply dropna() method
        parsed_data['date'] = pd.to_datetime(parsed_data['date'], errors='coerce')
        parsed_data = parsed_data.dropna(subset=['date'])

        # creating a timestamp column
        date_time=pd.to_datetime(parsed_data['date'].astype(str)+ ' '+ parsed_data['time'].astype(str))
        parsed_data.insert(0, 'timestamp', date_time)

        # droping the columns date and time
        parsed_data.drop(['date','time'], axis=1, inplace=True)

        
        return parsed_data
    
    
        
    
    def create_label(self) ->pd.DataFrame:
    
        labeled_data = DataParser.parse_data(self)
        
        conditions = [
            (labeled_data['log_message'].str.contains('service status')),
            #(df3_error['log_message_3'].str.contains('network cache')) & (df3_error['log_message_3'].str.contains('ConnectTimeout')),
            (labeled_data['log_message'].str.contains('network cache')),
            (labeled_data['log_message'].str.contains('ComputeManager.update')),
            (labeled_data['log_message'].str.contains('ComputeManager._heal')),
            (labeled_data['log_message'].str.contains('ComputeManager._sync_scheduler')),
            (labeled_data['log_message'].str.contains('ConnectTimeout')),
            (labeled_data['log_message'].str.contains('ComputeManager._run_pending')),
            (labeled_data['log_message'].str.contains('ComputeManager._cleanup_incomplete')),
            (labeled_data['log_message'].str.contains('connection blocked')),
            (labeled_data['log_message'].str.contains('ComputeManager._sync_power')),
            (labeled_data['log_message'].str.contains('AMQP server on pouta2')),
            (labeled_data['log_message'].str.contains('AMQP server on pouta1')),
            (labeled_data['log_message'].str.contains('ComputeManager._run_image')),
            (labeled_data['log_message'].str.contains('ComputeManager._cleanup_running')),
            (labeled_data['log_message'].str.contains('Error updating resources for node')),
            (labeled_data['log_message'].str.contains('Unable to access floating IP')),
            (labeled_data['log_message'].str.contains('ProcessExecutionError')),
            (labeled_data['log_message'].str.contains('InvalidSharedStorage_Remote')),
            (labeled_data['log_message'].str.contains('Failed storing info cache'))]

        # create a list of the values we want to assign for each condition
        values = ['service status', 'network cache', 'CP update resource', 'CP heal instance', 'CP sync scheduler',
                 'NC ConnectTimeout', 'CP run pending', 'CP clearup migrations', 'broker blocked connection','CP sync power',
                 'AMQP server pouta2',  'AMQP server pouta1','CP run image cache','CP cleanup running instance','node updating',
                 'floating IP access','ProcessExecutionError','InvalidSharedStorage_Remote', 'storing info cache']

        # create a new column and use np.select to assign values to it using the lists as arguments
        labeled_data['label'] = np.select(conditions, values)
        
        return labeled_data



    def store_data(self) ->pd.DataFrame:
        #self.parsed_data.to_csv(PROCESSED_DATA_PATH, index = False, header = True)
        # csv the file so we do not need to reprocess each time
        self.csv_processed_df_filename = 'processed_data.csv'
        self.csv_file_loc = os.path.join(PROCESSED_DATA_PATH, self.csv_processed_df_filename)
        data_saved = DataParser.create_label(self)
        # df to csv
        data_saved.to_csv(self.csv_file_loc, index = False, header = True)
        
        

    def sub_dataframe(self) ->pd.DataFrame:
        final_dataframe = DataParser().create_label()
        final_dataframe = final_dataframe[['log_message', 'label']]
        
        return final_dataframe
    
# -

dataParseInstance = DataParser()
data2 = dataParseInstance.sub_dataframe()
data2.head(2)


