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

from get_data import DataLoader
from parse_data import DataParser

# +
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.preprocessing import LabelEncoder


class DataPreprocess():
    
    def __init__(self):
        self.raw_text = None
        

        #print(self.labeled_data.head(2))
        
    # fucntion to clean the text
    def clean_text(raw_text):
        
    
        # keep only words
        letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

        # convert to lower case and split
        words = letters_only_text.lower().split()

        # remove the single word
        no_single_words = [word for word in words if len(word)>1]
        # in case of dataframe
        #df[''].map(lambda x: ' '.join(word for word in x.split() if len(word)>1))

        # remove stopwords
        stopwords_set = set(stopwords.words("english"))
        meaningful_words = [w for w in no_single_words if w not in stopwords_set]

        # stemmed words
        #ps = PorterStemmer()
        #stemmed_words = [ps.stem(word) for word in meaningful_words]

        # tokenize the words
        #tokenized_words = [word_tokenize(entry) for entry in meaningful_words]


        # join the cleanned words in a list
        cleaned_word_list = " ".join(meaningful_words)

        return cleaned_word_list
    
    
    def clean_log(self) ->pd.DataFrame:
        cleaned_log = DataParser().sub_dataframe()
        #aa= df1['log_message'].apply(DataPreprocess)
        #bb= df1['log_message'].apply(lambda x: DataPreprocess(x))
        cleaned_log['processed_log']= cleaned_log['log_message'].apply(lambda x: DataPreprocess.clean_text(x))
        cleaned_log.drop(['log_message'], axis=1, inplace=True)
        
        return cleaned_log
    
    
    # transform the label categories into distinct integer values representing the initial categorical values
    def create_label_encoder(self) ->pd.DataFrame:
        
        encoded_label = DataPreprocess().clean_log()
        
        label_encoder = LabelEncoder()

        encoded_label['label_ID'] = label_encoder.fit_transform(encoded_label['label'])
        
        # Also its good to have the categories as a dictionary
        label_map = encoded_label.set_index('label_ID').to_dict()['label']
        #print(label_map)

        return encoded_label
        
# -

dataPreprocessInstance = DataPreprocess()
data = dataPreprocessInstance.create_label_encoder()
data.head(2)


