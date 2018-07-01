import os
import csv
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Including DFKI
data1 = pd.read_csv("DFKI/P07_1001-50.csv", sep="\t", index_col=False, encoding='latin-1', low_memory=False)
df1 = DataFrame(data1)
df1 = df1[df1.label != 'Neutral'] #removes rows with Netral label
x1 = df1.data
y1 = df1.label

data2 = pd.read_csv("DFKI/P08_1009-50.csv", sep="\t", index_col=False, encoding='latin-1', low_memory=False)
df2 = DataFrame(data2)
df2 = df2[df2.label != 'Neutral']
x2 = df2.data
y2 = df2.label

data3 = pd.read_csv("DFKI/P08_2001-30.csv", sep="\t", index_col=False, encoding='latin-1', low_memory=False)
df3 = DataFrame(data3)
df3 = df3[df3.label != 'Neutral']
x3 = df3.data
y3 = df3.label

#including IMS
data4 = pd.read_csv("corpus2.csv", sep=",", index_col=False, encoding='latin-1', low_memory=False)
df4 = DataFrame(data4)
x4 = df4.data
y4 = df4.label

frames = [df1, df2, df3, df4]
df = pd.concat(frames, ignore_index=True)

# preprocessing list of paragraphs
paragraphs = df['data'].values.tolist()
pre_para = []
for u in paragraphs:
    u = re.sub(r"[^a-zA-Z0-9]+", ' ', u)
    pre_para.append(u.lower())

label_list = df['label'].values.tolist()

#joing lists
complete_list = []
for sent1, sent2 in zip(pre_para, label_list):
    complete_list.append([sent1, sent2])


# removing duplicates
def unique(seq):
   # order preserving
   checked = []
   for e in seq:
       if e not in checked:
           checked.append(e)
   return checked

unique_list = unique(complete_list)

#getting 179 positives and negatives
list_179 = []
j = 0
k = 0
for i in unique_list:
    if i[1] == 'Negative' and j < 179:
        list_179.append(i)
        j = j + 1
    if i[1] == 'Positive' and k < 179:
        list_179.append(i)
        k = k + 1

# x and y
x = [x[0] for x in list_179]
y = [x[1] for x in list_179]

#classification
def labelEncoding(y):
    labelEncoder = LabelEncoder()
    y_encoded = labelEncoder.fit_transform(y)
    return y_encoded

def tfidfVectorizer(x):
    stopset = stopwords.words('english')
    vect = TfidfVectorizer(analyzer='word', encoding='utf-8', min_df = 0, ngram_range=(1, 2), lowercase = True, strip_accents='ascii', stop_words = stopset)
    X_vec = vect.fit_transform(x)
    return X_vec

X_vec = tfidfVectorizer(x)
y_encoded = labelEncoding(y)


def splitTestTrain(X_vec, y_encoded):
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y_encoded,
test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = splitTestTrain(X_vec, y_encoded)

def applyNaiveBayesClassifier(X_train, y_train, X_test, y_test):
    # Thanks to sklearn, let us quickly train some multinomial models
    # Model Training: Multinomial Naive Bayes
    mnb_classifier = MultinomialNB()
    mnb_classifier.fit(X_train, y_train)
    model_accuracies = cross_val_score(estimator=mnb_classifier,
                                       X=X_train, y=y_train, cv=10)
    model_accuracies.mean()
    model_accuracies.std()
    # Model Testing: Multinomial Naive Bayes
    y_pred = mnb_classifier.predict(X_test)
    metrics.confusion_matrix(y_test, y_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    precision_mnb = precision_score(y_test, y_pred, average='macro')
    recall_mnb = recall_score(y_test, y_pred, average='macro')
    f_mnb = 2*(precision_mnb*recall_mnb)/(precision_mnb+recall_mnb)
    print("Multinomial Naive Bayes Classifier Test Accuracy: ", test_accuracy*100)
    print("Multinomial Naive Bayes Classifier Test Precision: ", precision_mnb*100)
    print("Multinomial Naive Bayes Classifier Test Recall: ", recall_mnb*100)
    print("Multinomial Naive Bayes Classifier Test F measure: ", f_mnb*100)
    return test_accuracy, precision_mnb, recall_mnb, f_mnb

accuracy, precision, recall, f1_score = applyNaiveBayesClassifier(X_train, y_train, X_test, y_test)
print(accuracy)

def applySVMClassifier(X_train, y_train, X_test, y_test):
    # Model Training: SVMs
    svc_classifier = SVC(kernel='linear', random_state=0)
    svc_classifier.fit(X_train, y_train)
    model_accuracies = cross_val_score(estimator=svc_classifier,
                                   X=X_train, y=y_train, cv=10)
    print("Model Accuracies Mean", model_accuracies.mean()*100)
    print("Model Accuracies Standard Devision", model_accuracies.std()*100)
    # Model Testing: SVMs
    y_pred = svc_classifier.predict(X_test)
    metrics.confusion_matrix(y_test, y_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    precision_SVC = precision_score(y_test, y_pred, average='macro')
    recall_SVC = recall_score(y_test, y_pred, average='macro')
    f_SVC = 2*(precision_SVC * recall_SVC) / (precision_SVC + recall_SVC)
    print("SVCs Test Accuracy: ", test_accuracy*100)
    print("SVCs Test Precision: ", precision_SVC*100)
    print("SVCs Test Recall: ", recall_SVC*100)
    print("SVCs Test F measure: ", f_SVC*100)
    return test_accuracy, precision_SVC, recall_SVC, f_SVC

accuracy, precision, recall, f1_score = applySVMClassifier(X_train, y_train, X_test, y_test)
print(accuracy)


def applyDecisionTreeClassifier(X_train, y_train, X_test, y_test):
    # Model Training: Decision Tree
    dt_classifier = tree.DecisionTreeClassifier(splitter= 'random')
    dt_classifier.fit(X_train, y_train)
    model_accuracies = cross_val_score(estimator=dt_classifier,
                                       X=X_train, y=y_train, cv=10)
    model_accuracies.mean()
    model_accuracies.std()
    # Model Testing: Decision Tree
    y_pred = dt_classifier.predict(X_test)
    metrics.confusion_matrix(y_test, y_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    precision_dt = precision_score(y_test, y_pred, average='macro')
    recall_dt = recall_score(y_test, y_pred, average='macro')
    f_dt = 2*(precision_dt*recall_dt)/(precision_dt+recall_dt)
    print("Decision Tree Classifier Test Accuracy: ", test_accuracy*100)
    print("Decision Tree Classifier Classifier Test Precision: ", precision_dt*100)
    print("Decision Tree Classifier Classifier Test Recall: ", recall_dt*100)
    print("Decision Tree Classifier Classifier Test F measure: ", f_dt*100)
    return test_accuracy, precision_dt, recall_dt, f_dt

accuracy, precision, recall, f1_score = applyDecisionTreeClassifier(X_train, y_train, X_test, y_test)
print(accuracy)

def applyLogisticRegression(X_train, y_train, X_test, y_test):
    # Model Training: Logistic Regression
    lr_classifier = LogisticRegression(random_state=0)
    lr_classifier.fit(X_train, y_train)
    model_accuracies = cross_val_score(estimator=lr_classifier,
                                       X=X_train, y=y_train, cv=10)
    model_accuracies.mean()
    model_accuracies.std()
    # Model Testing: Logistic Regression
    y_pred = lr_classifier.predict(X_test)
    metrics.confusion_matrix(y_test, y_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    precision_lr = precision_score(y_test, y_pred, average='macro')
    recall_lr = recall_score(y_test, y_pred, average='macro')
    f_lr = 2*(precision_lr*recall_lr)/(precision_lr+recall_lr)
    print("Logistic Regression Classifier Test Accuracy: ", test_accuracy*100)
    print("Logistic Regression Classifier Test Precision: ", precision_lr*100)
    print("Logistic Regression Classifier Test Recall: ", recall_lr*100)
    print("Logistic Regression Classifier Test F measure: ", f_lr*100)
    return test_accuracy, precision_lr, recall_lr, f_lr

accuracy, precision, recall, f1_score = applyLogisticRegression(X_train, y_train, X_test, y_test)
print(accuracy)
