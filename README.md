# HAR
### Human activity recognition is the problem of classifying sequences of data recorded by specialized harnesses or smart phones into known well-defined Human activities.
### It is a challenging problem as the large number of observations are produced each second, the temporal nature of the observations, and the lack of a clear way to relate data to known movements increase the challenges.
### In this Machine Learning Project, we will create a model for recognition of human activity using the smartphone data.
# Let’s start with Importing necessary libraries
### import pandas as pd
### import numpy as np
### import seaborn as sns
### import matplotlib.pyplot as plt
### %matplotlib inline
### import warnings
### warnings.filterwarnings("ignore")
# Download the data sets link here below
### train -> https://thecleverprogrammer.com/wp-content/uploads/2020/05/train-1.csv
### test -> https://thecleverprogrammer.com/wp-content/uploads/2020/05/test.csv
# Reading the data
### train = pd.read_csv("train.csv")
### test = pd.read_csv("test.csv")
# To Combine both the  data frames
### cc_data = pd.concat([train, test], axis=0).reset_index(drop=True)
# Then check the shape of the data set
### train.shape, test.shape
# EDA
### cc_data.head()
# info of the dataset
### pd.set_option('display.max_rows', None) -> this is for max the row
### cc_data.info()
# how data spread for numerical values
### cc_data.describe()
# handling missing values
### cc_data.isnull().sum()
# check the duplicate values
### duplicate = cc_data.duplicated()
# Outcome -> label data
### outcome = cc_data['Activity']
# Feature selection
### feature = cc_data.drop(['subject','Activity'], axis =1)
# Feature scaling
### temp = feature.values
# MinMax scaler and Standard Scaler
### from sklearn.preprocessing import MinMaxScaler
### mms = MinMaxScaler()
### from sklearn.preprocessing import StandardScaler
### ss = StandardScaler()
### train = ss.fit_transform(temp)-> for Standardization
### train_1 = mms.fit_transform(temp) -> for Normalization
# Visualize the data 
### import matplotlib.pyplot as plt
### import seaborn as sns
### plt.figure(figsize=(10, 5))
### plt.subplot(1,3,1)
### sns.distplot(temp, hist=False)
### plt.subplot(1,3,2)
### sns.distplot(train, hist=False)
###plt.subplot(1,3,3)
### sns.distplot(train_1, hist=False)
# Then go with the Normalization 
### from sklearn.model_selection import train_test_split
### X_train, X_test, y_train, y_test = train_test_split(train_1, outcome, test_size = 0.2, random_state = 0)
#Popular algorithms that can be used for multi-class classification include:
## k-Nearest Neighbors.
## Decision Trees.
## svm
## Naive Bayes.
# k-Nearest Neighbors.
### from sklearn.neighbors import KNeighborsClassifier
### from sklearn.metrics import confusion_matrix, accuracy_score
then
### model = KNeighborsClassifier(algorithm= 'auto', n_neighbors= 10, p= 1, weights= 'distance')
### model.fit(X_train, y_train)
### y_predict = model.predict(X_test)
### acc_knn = accuracy_score(y_test, y_predict)

# getting random rows
### def rand_rows(n): 
###rand = feature.sample(n=n)
###return rand
### rand = rand_rows(3)
# getting randomly as we want example 5 here
### ind = rand.index
### rand = rand.values
### o =[]
### for i in ind:
###   o.append(outcome[i])
### rand_predict = model.predict(rand)
# accuracy for the prediction
### accuracy_ = accuracy_score(o, rand_predict)
### accuracy_
#  Decision Trees.
### from sklearn.tree import DecisionTreeClassifier
### dtree_model = DecisionTreeClassifier(max_depth = 2)
### dtree_model.fit(X_train, y_train)
### dtree_predictions = dtree_model.predict(X_test)
### acc_tree = accuracy_score(y_test, dtree_predictions)
# checking the accuracy
### acc_tree = accuracy_score(y_test, dtree_predictions)
### acc_tree
### rand_predict_tree = dtree_model.predict(rand)
# accuracy for the prediction
### accuracy_tree = accuracy_score(rand_predict_tree, o)
### accuracy_tree
# svm
### from sklearn.svm import SVC
### svm_model_linear = SVC(kernel = 'linear', C = 1)
### svm_model_linear.fit(X_train, y_train)
### svm_predictions = svm_model_linear.predict(X_test)
### acc_svml = accuracy_score(y_test, svm_predictions)
### acc_svml
### rand_predict_svml = svm_model_linear.predict(rand)
### accuracy_svml = accuracy_score(rand_predict_svml, o)
### accuracy_svml
# rbf kernal
### svm_model_rbf = SVC(kernel = 'rbf', C = 1)
### svm_model_rbf.fit(X_train, y_train)
### svm_prediction = svm_model_rbf.predict(X_test)
### acc_svmr = accuracy_score(y_test, svm_prediction)
### acc_svmr
### rand_predict_svmr = svm_model_rbf.predict(rand)
### accuracy_svmr = accuracy_score(rand_predict_svmr, o)
### accuracy_svmr
# Naive Bayes 
### gnb = GaussianNB()
### gnb.fit(X_train, y_train)
### gnb_predictions = gnb.predict(X_test)
### rand_predict_gnb = gnb.predict(rand)
### accuracy_gnb = accuracy_score(rand_predict_gnb, o)
### accuracy_gnb
### acc_gnb = accuracy_score(y_test, gnb_predictions)
### acc_gnb
