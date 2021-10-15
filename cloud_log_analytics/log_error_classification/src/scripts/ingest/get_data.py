# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/ingest//py
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
import os
import json
import fnmatch
import re

import io
from io import StringIO, BytesIO
import pandas as pd

BASE_DIR= r'C:\Users\rbhuiyan\Desktop\log_classification\log_error_classification\data\raw\compute_nodes_5'


class DataLoader():
    
    def __init__(self):
        self.BASE_DIR = BASE_DIR
        
    
    def load_data(self) -> pd.DataFrame:
        self.new_list=[]
        for root, dirs, files in os.walk(self.BASE_DIR):
            for file in files:
                if fnmatch.fnmatch(file, 'messages*'): #reading files that start with messages
                    with open(os.path.join(root, file), 'r') as f:
                        for line in f:
                            if re.search('ERROR', line): #if 'ERROR' in line:
                                self.new_list.append(line)
        # converting list into string
        new_string= ''.join(map(str, self.new_list))
        df_log_error = pd.read_csv(StringIO(new_string), sep="\n", names=['Column'], engine='python')
        # to view the column in wide
        pd.set_option('max_colwidth', 1000)

        return df_log_error

# +
# creating a instance of the class
loaderInstance = DataLoader()

# calling the instance of the class
data = loaderInstance.load_data()

# printing the output
data.head(2)
# -
data.Column.iloc[1]



