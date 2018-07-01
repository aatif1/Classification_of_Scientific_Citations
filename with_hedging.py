import os
import csv
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud


#including DFKI
data1 = pd.read_csv("DFKI/P07_1001-50.csv", sep="\t", index_col=False, encoding='latin-1', low_memory=False)
df1 = DataFrame(data1)
df1 = df1[df1.label != 'Neutral']  #removes rows with Netral label
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

# including IMS
data4 = pd.read_csv("corpus2.csv", sep=",", index_col=False, encoding='latin-1', low_memory=False)
df4 = DataFrame(data4)
x4 = df4.data
y4 = df4.label

frames = [df1, df2, df3, df4]
df = pd.concat(frames, ignore_index=True)

# hedging words to list
with open('hedges.csv', 'r') as f:
    reader = csv.reader(f)
    hedges = list(reader)
# hedging (list of list) to list
hedges_list = [item for sublist in hedges for item in sublist]
paragraphs = df['data']

# drawing wordcloud
def wordcloud_draw(data, color='white'):
    words = ' '.join(data)
    # cleaned_word = " ".join([word for word in words.split()
    #                          if 'http' not in word
    #                          and not word.startswith('@')
    #                          and not word.startswith('#')
    #                          and word != 'RT'
    #                          ])
    wordcloud = WordCloud(background_color=color,
                          width=2500,
                          height=2000
                          ).generate(words)
    plt.figure(1, figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    #plt.savefig('word_cloud') #uncomment to save picture

# uncomment next 2 lines to generate pic
# print("Hedging Cues")
# wordcloud_draw(hedges_list)

# preprocessing list of paragraphs
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word=word)

# for sent in paragraphs:
paragraphs = paragraphs.str.replace('[^a-zA-Z0-9-_*.]', ' ')
pre_para = [" ".join([lemmatize(word) for word in sentence.split(" ")]) for sentence in paragraphs]

#replacing list of list to list

label_list = df['label'].values.tolist()

# joing lists
complete_list = []
for sent1, sent2 in zip(pre_para, label_list):
    complete_list.append([sent1, sent2])

# removing repeated things
def unique(seq):
   # order preserving
   checked = []
   for e in seq:
       if e not in checked:
           checked.append(e)
   return checked

unique_list = unique(complete_list)

# getting 179 positives and negatives
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
# getting only sentences from list_179
list_358 = [item[0] for item in list_179]


# implementing hedge cue detection
hedges_bool = []
for word in hedges_list:
    for sent in list_358:
        found = False
        found_hedge_false = False
        # if word in sent:
        for i in hedges_bool:
            if i[0] == sent and i[1] == 'hedge_true':  # checks if true is repeated
                found = True
                break
            if i[0] == sent and i[1] == 'hedge_false':  # checks if false is  repeated
                found_hedge_false = True
        if found is True: # if hedge_true
            break
        if sent.find(word) != -1 and found_hedge_false is True:  # checks two idexes if same
            hedges_bool.remove([sent, 'hedge_false'])
            hedges_bool.append([sent, 'hedge_true'])
        if found_hedge_false is True:  # it can become true in further steps
            continue
        else:
            hedges_bool.append([sent, 'hedge_false'])

# x and y
x_data = []
y_data = []
for sent1, sent2 in zip(label_list, hedges_bool):
    x_data.append(sent2[0])
    y_data.append([sent1, sent2[1]])

# classification

def labelEncoding(y_data):
    mlb = MultiLabelBinarizer()
    y_binarized = mlb.fit_transform(y_data)
    return y_binarized


def tfidfVectorizer(x_data):
    stopset = stopwords.words('english')
    vect = TfidfVectorizer(analyzer='word', encoding='utf-8', min_df = 0, ngram_range=(1, 2), lowercase = True, strip_accents='ascii', stop_words = stopset)
    X_vec = vect.fit_transform(x_data)
    return X_vec


X_vec = tfidfVectorizer(x_data)
y_encoded = labelEncoding(y_data)


def splitTestTrain(X_vec, y_encoded):
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y_encoded,
test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = splitTestTrain(X_vec, y_encoded)

def applyDecisionTreeClassifier(X_train, y_train, X_test, y_test):
    # Model Training: DecisionTreeClassifier
    DT_classifier = DecisionTreeClassifier(random_state=0)
    DT_classifier.fit(X_train, y_train)
    print(DT_classifier)
    model_accuracies = cross_val_score(estimator=DT_classifier,
                                   X=X_train, y=y_train, cv=10)
    print("Model Accuracies Mean", model_accuracies.mean()*100)
    print("Model Accuracies Standard Devision", model_accuracies.std()*100)
    # Model Testing: DTs
    y_pred = DT_classifier.predict(X_test)
    metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    precision_DT = precision_score(y_test, y_pred, average='weighted')
    recall_DT = recall_score(y_test, y_pred, average='weighted')
    f_DT = 2*(precision_DT * recall_DT) / (precision_DT + recall_DT)
    print("Decision Tree Classifier Test Accuracy: ", test_accuracy*100)
    print("Decision Tree Classifier Test Precision: ", precision_DT*100)
    print("Decision Tree Classifier Test Recall: ", recall_DT*100)
    print("Decision Tree Classifier Test F measure: ", f_DT*100)
    return test_accuracy, precision_DT, recall_DT, f_DT

accuracy, precision, recall, f1_score = applyDecisionTreeClassifier(X_train, y_train, X_test, y_test)
print(accuracy)

def applyKNeighborsClassifier(X_train, y_train, X_test, y_test):
    # Model Training: KNN
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)
    print(knn_classifier)
    model_accuracies = cross_val_score(estimator=knn_classifier,
                                   X=X_train, y=y_train, cv=10)
    print("Model Accuracies Mean", model_accuracies.mean()*100)
    print("Model Accuracies Standard Devision", model_accuracies.std()*100)
    # Model Testing: Knn
    y_pred = knn_classifier.predict(X_test)
    metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    precision_KNN = precision_score(y_test, y_pred, average='weighted')
    recall_KNN = recall_score(y_test, y_pred, average='weighted')
    f_KNN = 2*(precision_KNN * recall_KNN) / (precision_KNN + recall_KNN)
    print("KNNs Test Accuracy: ", test_accuracy*100)
    print("KNNs Test Precision: ", precision_KNN*100)
    print("KNNs Test Recall: ", recall_KNN*100)
    print("KNNs Test F measure: ", f_KNN*100)
    return test_accuracy, precision_KNN, recall_KNN, f_KNN

accuracy, precision, recall, f1_score = applyKNeighborsClassifier(X_train, y_train, X_test, y_test)
print(accuracy)