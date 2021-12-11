# Production level Data Science Project for Server Log Data
## A complete data science project from prototyping of code to productionizing of code for deployment (during Data Science Trainee at CSC) 

Server logs are typically a very common data source in business enterprise and often contain a gold mine of actionable insights and information. The importance of log data analytics is indispensable in terms of monitoring the servers, improving the business and customer intelligence, building recommendation systems, fraud detection and particularly in identifying the critical points for debugging a server failure and performing root cause analysis. Manual log analysis - checking system logs or writing rules to detect anomalies based on their domain knowledge is inadequate for large-scale systems and thus machine learning methods are highly in demand for automation.


The aim is to develop machine learning model in classifying and predicting the error log messages so that the server system can automatically detect the anomalies.

For full view please visit the PDFs - [SummerProject_LogAnalytics_report.pdf](https://github.com/rabbilbhuiyan/cloud-log-analytics-NLP/files/7696994/SummerProject_LogAnalytics_report.pdf) and [SummerProject_LogAnalytics_presentation.pdf](https://github.com/rabbilbhuiyan/cloud-log-analytics-NLP/files/7697001/SummerProject_LogAnalytics_presentation.pdf)

To see Rabbil Bhuiyan's jupyter notebook work visit this link - 

## Methods
- The dataset contains 64,275,039 i.e more than 64 million lines of logs labeled as INFO, ERROR and WARNING
- The logs are parsed into useful information using regular expressions
- The text is further cleaned to feed into NLP algorithms
- TF-IDF word vectorization technique is applied to transfrom the raw text into vectors of numbers
- Support Vector Machines (SVM) and Naïve Bayes classifers are applied for training the model
- Prototyping of code is converted into production level code by creating Python Classes and Objects (Object Oriented Programming) 

## Key Findings
The model accuracy for Naive Bayes classifier was quite satisfactory (99.72). As the model is multiclass probelm, we further explore the accuracy by each class separately by using confusion matrix. We observed most of the values in the diagonal line in the confusion matrix which is true positive, indicating perfect accuracy of the model as in the figure. 

![confusion_matrix_plot](https://user-images.githubusercontent.com/36482524/145673586-a33cddcf-19e8-45f1-b3a7-6be9ca1c7dce.png)



