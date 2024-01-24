# -*- coding: utf-8 -*-
"""Diabetes_disease_prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Uy_NhIWYImo-q0aOSCafaDSu5xBxaCjo

## I. Importing essential libraries
"""

# Commented out IPython magic to ensure Python compatibility.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# %matplotlib inline

import os

print(os.listdir())

import warnings

warnings.filterwarnings('ignore')

"""## II. Importing and understanding our dataset

For Diabetes
"""
'''
from google.colab import files
uploaded = files.upload()

dataset = pd.read_csv("diabetes_data_upload (1).csv")

from google.colab import drive
drive.mount('/content/drive')
'''

disease_type = 2

if disease_type == 1:
    dataset = pd.read_csv("diabetes_data_upload.csv")

"""For Heart Disease"""
if disease_type == 2:
    dataset = pd.read_csv("HeartDiseaseDataset.csv")

"""For Thyroid Disease"""
if disease_type == 3:
    dataset = pd.read_csv("ThyroidDatasetwithTraining.csv")

"""#### Verifying it as a 'dataframe' object in pandas"""

type(dataset)

"""#### Shape of dataset"""

dataset.shape

"""#### Printing out a few columns"""

dataset.head(5)

dataset.sample(5)

"""#### Description"""

dataset.describe()

dataset.info()

"""## IV. Train Test split

For Diabetes
"""

from sklearn.model_selection import train_test_split

if disease_type == 1:
    predictors = dataset.drop("class", axis=1)
    target = dataset["class"]

    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=10)

"""For Heart Disease"""
if disease_type == 2:
    predictors = dataset.drop("num", axis=1)
    target = dataset["num"]

    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=10)

"""For Thyroid Disease"""
if disease_type == 3:
    predictors = dataset.drop(["referralsource", "classes"], axis=1)
    # predictors = dataset.drop("referralsource",axis=1)
    target = dataset["classes"]

    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=25)

print(X_test)

X_train.shape

X_test.shape

Y_train.shape

Y_test.shape

"""Confusion Matrix

"""

from matplotlib import pyplot as plt
from sklearn import metrics
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


"""## V. Model Fitting"""

from sklearn.metrics import accuracy_score

"""### Logistic Regression"""

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train, Y_train)

Y_pred_lr = lr.predict(X_test)

Y_pred_lr.shape

score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)

print("The accuracy score achieved using Logistic Regression is: " + str(score_lr) + " %")

cm = metrics.confusion_matrix(Y_test, Y_pred_lr, labels=[0, 1])
plot_confusion_matrix(cm, classes=['NO', 'YES'])

"""### Naive Bayes"""

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, Y_train)

Y_pred_nb = nb.predict(X_test)

Y_pred_nb.shape

score_nb = round(accuracy_score(Y_pred_nb, Y_test) * 100, 2)

print("The accuracy score achieved using Naive Bayes is: " + str(score_nb) + " %")

cm = metrics.confusion_matrix(Y_test, Y_pred_nb, labels=[0, 1])
plot_confusion_matrix(cm, classes=['NO', 'YES'])

"""### SVM"""

from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(X_train, Y_train)

Y_pred_svm = sv.predict(X_test)

Y_pred_svm.shape

score_svm = round(accuracy_score(Y_pred_svm, Y_test) * 100, 2)

print("The accuracy score achieved using Linear SVM is: " + str(score_svm) + " %")

cm = metrics.confusion_matrix(Y_test, Y_pred_svm, labels=[0, 1])
plot_confusion_matrix(cm, classes=['NO', 'YES'])

"""### K Nearest Neighbors"""

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)

Y_pred_knn.shape

score_knn = round(accuracy_score(Y_pred_knn, Y_test) * 100, 2)

print("The accuracy score achieved using KNN is: " + str(score_knn) + " %")

cm = metrics.confusion_matrix(Y_test, Y_pred_knn, labels=[0, 1])
plot_confusion_matrix(cm, classes=['NO', 'YES'])

"""### Decision Tree"""

from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0

'''
for x in range(2000):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)

'''
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test)

print(Y_pred_dt.shape)

score_dt = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)

print("The accuracy score achieved using Decision Tree is: " + str(score_dt) + " %")

cm = metrics.confusion_matrix(Y_test, Y_pred_dt, labels=[0, 1])
plot_confusion_matrix(cm, classes=['NO', 'YES'])

"""### Random Forest"""

from sklearn.ensemble import RandomForestClassifier

'''
max_accuracy = 0


for x in range(500):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
   '''
# print(max_accuracy)
# print(best_x)

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)

Y_pred_rf.shape

score_rf = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)

print("The accuracy score achieved using Decision Tree is: " + str(score_rf) + " %")

cm = metrics.confusion_matrix(Y_test, Y_pred_rf, labels=[0, 1])
plot_confusion_matrix(cm, classes=['NO', 'YES'])

"""### XGBoost"""

import xgboost as xgb

max_accuracy = 0

'''
for x in range(500):
    xgb_model = xgb.XGBClassifier( random_state=x)
    xgb_model.fit(X_train,Y_train)
    Y_pred_xgb = xgb_model.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
'''
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, Y_train)

Y_pred_xgb = xgb_model.predict(X_test)

Y_pred_xgb.shape

score_xgb = round(accuracy_score(Y_pred_xgb, Y_test) * 100, 2)

print("The accuracy score achieved using XGBoost is: " + str(score_xgb) + " %")

cm = metrics.confusion_matrix(Y_test, Y_pred_xgb, labels=[0, 1])
plot_confusion_matrix(cm, classes=['NO', 'YES'])

"""### Neural Network"""

from keras.models import Sequential
from keras.layers import Dense

if disease_type == 2:
    """For Heart Disease"""
    model = Sequential()
    model.add(Dense(11, activation='relu', input_dim=13))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

if disease_type == 1:
    """For Diabetes """

    model = Sequential()
    model.add(Dense(11, activation='relu', input_dim=16))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary();

"""For Thyroid"""

if disease_type == 3:
    model = Sequential()
    model.add(Dense(11, activation='relu', input_dim=28))
    model.add(Dense(28, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=500)

Y_pred_nn = model.predict(X_test)

Y_pred_nn.shape

rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded

score_nn = round(accuracy_score(Y_pred_nn, Y_test) * 100, 2)

print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")

cm = metrics.confusion_matrix(Y_test, Y_pred_nn, labels=[0, 1])
plot_confusion_matrix(cm, classes=['NO', 'YES'])

"""## VI. Output final score"""

scores = [score_lr, score_nb, score_svm, score_knn, score_dt, score_rf, score_xgb, score_nn]
algorithms = ["Logistic Regression", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbors", "Decision Tree",
              "Random Forest", "XGBoost", "Neural Network"]

for i in range(len(algorithms)):
    print("The accuracy score achieved using " + algorithms[i] + " is: " + str(scores[i]) + " %")
Average_accuracy = (score_lr + score_nb + score_svm + score_knn + score_dt + score_rf + score_xgb + score_nn) / 8
print("Average Accuracy: " + str(Average_accuracy))

"""Model Saving

For Diabetes
"""
if disease_type == 1:
    model_file = 'final_diabetes_model.sav'
    pickle.dump(xgb_model, open(model_file, 'wb'))

    Age = 40
    Gender = 1
    Polyuria = 0
    Polydipsia = 1
    sudden_weight_loss = 0
    weakness = 1
    Polyphagia = 0
    Genital_thrush = 0
    visual_blurring = 0
    Itching = 1
    Irritability = 0
    delayed_healing = 1
    partial_paresis = 0
    muscle_stiffness = 1
    Alopecia = 1
    Obesity = 1
    classes = 1
    test_input = {'Age': [Age], 'Gender': [Gender], 'Polyuria': [Polyuria], 'Polydipsia' : [Polydipsia],'sudden_weight_loss': [sudden_weight_loss],
                  'weakness': [weakness], 'Polyphagia': [Polyphagia], 'Genital_thrush': [Genital_thrush],
                  'visual_blurring': [visual_blurring], 'Itching': [Itching], 'Irritability': [Irritability],
                  'delayed_healing': [delayed_healing], 'partial_paresis': [partial_paresis],
                  'muscle_stiffness': [muscle_stiffness],
                  'Alopecia': [Alopecia], 'Obesity': [Obesity]}
    df_input = pd.DataFrame(test_input)
    print(df_input)

    load_model = pickle.load(open('final_diabetes_model.sav', 'rb'))
    prediction = load_model.predict(df_input)
    # score = round(accuracy_score(prediction,Y_test)*100,2)
    # print(score)
    print(prediction[0])

"""For Heart"""
if disease_type == 2:
    model_file = 'final_heart_model.sav'
    pickle.dump(xgb_model, open(model_file, 'wb'))

    age = 63
    sex = 1
    cp = 1
    trestbps = 145
    chol = 233
    fbs = 1
    restecg = 2
    thalach = 150
    exang = 0
    oldpeak = 2.3
    slope = 3
    ca = 0
    thal = 6
    num = 0
    test_input = {'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol], 'fbs': [fbs],
                  'restecg': [restecg], 'thalach': [thalach], 'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope],
                  'ca': [ca], 'thal': [thal]}
    df_input = pd.DataFrame(test_input)

    # print(X_test)

    load_model = pickle.load(open('final_heart_model.sav', 'rb'))
    prediction = load_model.predict(df_input)
    # score = round(accuracy_score(prediction,Y_test)*100,2)
    # print(score)
    print(prediction[0])

"""For Thyroid"""
if disease_type == 3:
    model_file = 'final_thyroid_model.sav'
    pickle.dump(dt, open(model_file, 'wb'))

    age = 41
    sex = 0
    onthyroxine = 0
    queryonthyroxine = 0
    antithyroid = 0
    sick = 0
    pregnant = 0
    thyroidsurgery = 0
    i131treatment = 0
    queryhypothyroid = 0
    queryhyperthyroid = 0
    lithium = 0
    goitre = 0
    tumour = 0
    hypopituitary = 0
    psych = 0
    tshmeasured = 1
    th = 1.3
    t3measured = 1
    t3 = 2.5
    tt4measured = 1
    tt4 = 125
    t4umeasured = 1
    t4u = 1.14
    ftimeasured = 1
    fti = 109
    tbgmeasured = 0
    tbg = -1
    classes = 0
    test_input = {'age': [age], 'sex': [sex], 'onthyroxine': [onthyroxine], 'queryonthyroxine': [queryonthyroxine],
                  'antithyroid': [antithyroid], 'sick': [sick], 'pregnant': [pregnant],
                  'thyroidsurgery': [thyroidsurgery],
                  'i131treatment': [i131treatment], 'queryhypothyroid': [queryhypothyroid],
                  'queryhyperthyroid': [queryhyperthyroid],
                  'lithium': [lithium], 'goitre': [goitre], 'tumour': [tumour], 'hypopituitary': [hypopituitary],
                  'psych': [psych],
                  'tshmeasured': [tshmeasured], 'th': [th], 't3measured': [t3measured], 't3': [t3],
                  'tt4measured': [tt4measured],
                  'tt4': [tt4], 't4umeasured': [t4umeasured], 't4u': [t4u], 'ftimeasured': [ftimeasured], 'fti': [fti],
                  'tbgmeasured': [tbgmeasured], 'tbg': [tbg]}
    df_input = pd.DataFrame(test_input)
    print(df_input)

    load_model = pickle.load(open('final_thyroid_model.sav', 'rb'))
    prediction = load_model.predict(df_input)
    print(prediction[0])

sns.set(rc={'figure.figsize': (15, 8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms, scores)
print("End of program")