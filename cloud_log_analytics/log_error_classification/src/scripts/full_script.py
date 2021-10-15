# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts//py
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
import datetime
import numpy as np
import io
from io import StringIO, BytesIO
import pandas as pd
from numpy import random
import pickle


# to ingnore the warning 
import warnings
warnings.filterwarnings('ignore')


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report




BASE_DIR= r'C:\Users\rbhuiyan\Desktop\log_classification\log_error_classification\data\raw\compute_nodes_5'
PROCESSED_DATA_PATH = r'C:\Users\rbhuiyan\Desktop\log_classification\log_error_classification\data\processed'



# loading the data

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




# Data wrangling : parsing the data and creating label 

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
        
        # call the instance of the class DataLoader
        parsed_data = DataLoader().load_data()
        
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
    



# Data preprocessing

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
        
        

# Train the data into classifier

class TrainModel():
    
    def __init__(self):
        self.data = None
        self.classifier = None
        self.vectorizer = None
        self.X_test = None
        self.y_test = None

    
    @staticmethod
    def load_data():
        
        data = DataPreprocess().create_label_encoder()
        X = data['processed_log']
        y = data['label_ID']
    
        return X, y
    
    # creating function for evaluation 

    @staticmethod # alternatively we can add 'self' as an argument for the object
    def evaluate_classifier(title, classifier, vectorizer, X_test, y_test):
        
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = classifier.predict(X_test_tfidf)

        precision = metrics.precision_score(y_test, y_pred, average= 'weighted')
        recall = metrics.recall_score(y_test, y_pred, average= 'weighted')
        f1 = metrics.f1_score(y_test, y_pred, average= 'weighted')

        print("%s\t%f\t%f\t%f\n" % (title, precision, recall, f1))

    
    # creating the training classifier

    def train_classifier(data):
        
        # load data
        X, y = TrainModel().load_data()
        
        # splitting the data into train test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

        # the object to turn data (text) into vectors
        vectorizer = TfidfVectorizer()

        # create doceument term matrix
        dtm = vectorizer.fit_transform(X_train)

        # train Naive Bayes classifier
        naive_bayes_classifier = MultinomialNB().fit(dtm, y_train)

        # evaluating the model accuracy  
        TrainModel().evaluate_classifier('Naive Bayes\tTRAIN\t', naive_bayes_classifier, vectorizer, X_train, y_train)

        TrainModel().evaluate_classifier('Naive Bayes\tTEST\t', naive_bayes_classifier, vectorizer, X_test, y_test)

        # store the classifier so we can call that
        clf_filename = 'naive_bayes_classifier.pkl'
        pickle.dump(naive_bayes_classifier, open(clf_filename, 'wb'))

        # store the vectorizer so we can transform to new data
        vec_filename = 'tfidf_vectorizer.pkl'
        pickle.dump(vectorizer, open(vec_filename, 'wb'))
        
        return naive_bayes_classifier
    
    


# Predict the model

class PredictModel():
    
    def __init__(self):
        self.data = None
        self.cm = None
        self.classes = None
    
   
    # function to print and plot the confusion matrix
    
    @staticmethod # adding this TypeError: plot_confusion_matrix() got multiple values for argument 'classes' is solved
    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        classes: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']

        normalize:    If False, plot the raw numbers
                      If True, plot the proportions
        
        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """
        import itertools
        import numpy as np
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        
        accuracy = np.trace(cm) / np.sum(cm).astype('float')
        misclass = 1 - accuracy

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45) 
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.tight_layout()
    
    
        
    # creating prediction classifier
    
    def predict(self):
        
        # load data
        X, y = TrainModel().load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42 )
        
        # the object to turn data (text) into vectors
        vectorizer = TfidfVectorizer()

        # create doceument term matrix
        dtm = vectorizer.fit_transform(X_train)
        
        model = TrainModel().train_classifier()
        
        y_predicted = model.predict(vectorizer.transform(X_test))
        
        
        print('accuracy %s' % accuracy_score(y_predicted, y_test))
        print(classification_report(y_test, y_predicted)) # target_names=my_label
        
        
        #########################
        cnf_matrix = confusion_matrix(y_test, y_predicted)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure(figsize=(12,10))
        PredictModel().plot_confusion_matrix(cnf_matrix, classes=['network cache', 'service status', 'CP heal instance',
       'CP update resource', 'CP run pending', 'CP sync scheduler',
       'CP clearup migrations', 'CP run image cache', 'CP sync power',
       'broker blocked connection', 'AMQP server pouta1', 'node updating',
       'InvalidSharedStorage_Remote', 'ProcessExecutionError',
       'CP cleanup running instance', 'AMQP server pouta2',
       'floating IP access', 'storing info cache'], title='Confusion matrix, without normalization')
        
        # saving the plot
        plt.savefig('confusion_matrix_plot')
        
    
    

    #@staticmethod- we can use the @staticmethod decorator to avoid the error, If the method doesn't require self as an argument
    def classify(self, data):

        # load classifier
        clf_filename = 'naive_bayes_classifier.pkl'
        nb_clf = pickle.load(open(clf_filename, 'rb'))

        # vectorize the new text
        vec_filename = 'tfidf_vectorizer.pkl'
        vectorizer = pickle.load(open(vec_filename, 'rb'))

        pred = nb_clf.predict(vectorizer.transform([data]))

        print(pred[0])
        #print('accuracy %s' % accuracy_score(pred, y_test))
        #return pred




# printing the output  

predictModelInstance = PredictModel()
predict_result = predictModelInstance.predict()
print(predict_result)


# deployment in production

#new_data ="AMQP server on pouta1:5672 is unreachable: timed out. Trying again in 1 seconds"
#classify_data = predictModelInstance.classify(new_data)
#print(classify_data)
# -


