import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense

from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

dir_list = os.listdir()

global check_flag


def data_split(disease_type):
    from sklearn.model_selection import train_test_split

    if disease_type == 1:
        dataset = pd.read_csv("diabetes_data_upload.csv")
        predictors = dataset.drop("class", axis=1)
        target = dataset["class"]

        X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=10)

    """For Heart Disease"""
    if disease_type == 2:
        dataset = pd.read_csv("HeartDiseaseDataset.csv")
        predictors = dataset.drop("num", axis=1)
        target = dataset["num"]

        X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=10)

    """For Thyroid Disease"""
    if disease_type == 3:
        dataset = pd.read_csv("ThyroidDatasetwithTraining.csv")
        predictors = dataset.drop(["referralsource", "classes"], axis=1)
        target = dataset["classes"]

        X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=25)

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    Y_pred_lr = lr.predict(X_test)
    Y_pred_lr.shape
    score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    Y_pred_nb = nb.predict(X_test)
    Y_pred_nb.shape
    score_nb = round(accuracy_score(Y_pred_nb, Y_test) * 100, 2)

    # SVM
    sv = svm.SVC(kernel='linear')
    sv.fit(X_train, Y_train)
    Y_pred_svm = sv.predict(X_test)
    Y_pred_svm.shape
    score_svm = round(accuracy_score(Y_pred_svm, Y_test) * 100, 2)

    # K-Nearest Neighbour
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, Y_train)
    Y_pred_knn = knn.predict(X_test)
    Y_pred_knn.shape
    score_knn = round(accuracy_score(Y_pred_knn, Y_test) * 100, 2)

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, Y_train)
    Y_pred_dt = dt.predict(X_test)
    print(Y_pred_dt.shape)
    score_dt = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    Y_pred_rf.shape
    score_rf = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)

    # XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, Y_train)
    Y_pred_xgb = xgb_model.predict(X_test)
    Y_pred_xgb.shape
    score_xgb = round(accuracy_score(Y_pred_xgb, Y_test) * 100, 2)

    # Neural Network
    if disease_type == 2:
        """For Heart Disease"""
        model_nn = Sequential()
        model_nn.add(Dense(11, activation='relu', input_dim=13))
        model_nn.add(Dense(13, activation='relu'))
        model_nn.add(Dense(1, activation='sigmoid'))

        model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if disease_type == 1:
        """For Diabetes """

        model_nn = Sequential()
        model_nn.add(Dense(11, activation='relu', input_dim=16))
        model_nn.add(Dense(16, activation='relu'))
        model_nn.add(Dense(1, activation='sigmoid'))

        model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    """For Thyroid"""

    if disease_type == 3:
        model_nn = Sequential()
        model_nn.add(Dense(11, activation='relu', input_dim=28))
        model_nn.add(Dense(28, activation='relu'))
        model_nn.add(Dense(1, activation='sigmoid'))

        model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_nn.fit(X_train, Y_train, epochs=500)
    Y_pred_nn = model_nn.predict(X_test)
    Y_pred_nn.shape
    rounded = [round(x[0]) for x in Y_pred_nn]
    Y_pred_nn = rounded
    score_nn = round(accuracy_score(Y_pred_nn, Y_test) * 100, 2)

    # Best Model
    maximum = 0
    check_score = [score_lr, score_nb, score_svm, score_knn, score_dt, score_rf, score_xgb, score_nn]
    i = 0
    j = 0
    while i < 8:
        if maximum < check_score[i]:
            maximum = check_score[i]
            j = i
        i += 1

    print(check_score)

    if j == 0:
        if disease_type == 1:
            model_file = 'final_diabetes_model.sav'
            pickle.dump(lr, open(model_file, 'wb'))

        """For Heart"""
        if disease_type == 2:
            model_file = 'final_heart_model.sav'
            pickle.dump(lr, open(model_file, 'wb'))

        """For Thyroid"""
        if disease_type == 3:
            model_file = 'final_thyroid_model.sav'
            pickle.dump(lr, open(model_file, 'wb'))

    if j == 1:
        if disease_type == 1:
            model_file = 'final_diabetes_model.sav'
            pickle.dump(nb, open(model_file, 'wb'))

        """For Heart"""
        if disease_type == 2:
            model_file = 'final_heart_model.sav'
            pickle.dump(nb, open(model_file, 'wb'))

        """For Thyroid"""
        if disease_type == 3:
            model_file = 'final_thyroid_model.sav'
            pickle.dump(nb, open(model_file, 'wb'))

    if j == 2:
        if disease_type == 1:
            model_file = 'final_diabetes_model.sav'
            pickle.dump(sv, open(model_file, 'wb'))

        """For Heart"""
        if disease_type == 2:
            model_file = 'final_heart_model.sav'
            pickle.dump(sv, open(model_file, 'wb'))

        """For Thyroid"""
        if disease_type == 3:
            model_file = 'final_thyroid_model.sav'
            pickle.dump(sv, open(model_file, 'wb'))

    if j == 3:
        if disease_type == 1:
            model_file = 'final_diabetes_model.sav'
            pickle.dump(knn, open(model_file, 'wb'))

        """For Heart"""
        if disease_type == 2:
            model_file = 'final_heart_model.sav'
            pickle.dump(knn, open(model_file, 'wb'))

        """For Thyroid"""
        if disease_type == 3:
            model_file = 'final_thyroid_model.sav'
            pickle.dump(knn, open(model_file, 'wb'))

    if j == 4:
        if disease_type == 1:
            model_file = 'final_diabetes_model.sav'
            pickle.dump(dt, open(model_file, 'wb'))

        """For Heart"""
        if disease_type == 2:
            model_file = 'final_heart_model.sav'
            pickle.dump(dt, open(model_file, 'wb'))

        """For Thyroid"""
        if disease_type == 3:
            model_file = 'final_thyroid_model.sav'
            pickle.dump(dt, open(model_file, 'wb'))

    if j == 5:
        if disease_type == 1:
            model_file = 'final_diabetes_model.sav'
            pickle.dump(rf, open(model_file, 'wb'))

        """For Heart"""
        if disease_type == 2:
            model_file = 'final_heart_model.sav'
            pickle.dump(rf, open(model_file, 'wb'))

        """For Thyroid"""
        if disease_type == 3:
            model_file = 'final_thyroid_model.sav'
            pickle.dump(rf, open(model_file, 'wb'))

    if j == 6:
        if disease_type == 1:
            model_file = 'final_diabetes_model.sav'
            pickle.dump(xgb_model, open(model_file, 'wb'))

        """For Heart"""
        if disease_type == 2:
            model_file = 'final_heart_model.sav'
            pickle.dump(xgb_model, open(model_file, 'wb'))

        """For Thyroid"""
        if disease_type == 3:
            model_file = 'final_thyroid_model.sav'
            pickle.dump(xgb_model, open(model_file, 'wb'))

    if j == 7:
        if disease_type == 1:
            model_file = 'final_diabetes_model.sav'
            pickle.dump(model_nn, open(model_file, 'wb'))

        """For Heart"""
        if disease_type == 2:
            model_file = 'final_heart_model.sav'
            pickle.dump(model_nn, open(model_file, 'wb'))

        """For Thyroid"""
        if disease_type == 3:
            model_file = 'final_thyroid_model.sav'
            pickle.dump(model_nn, open(model_file, 'wb'))
    global check_flag
    check_flag = 1


def load_model(disease_type, df_input):
    if disease_type == 1:
        load_model = pickle.load(open('final_diabetes_model.sav', 'rb'))
        prediction = load_model.predict(df_input)

    if disease_type == 2:
        load_model = pickle.load(open('final_heart_model.sav', 'rb'))
        prediction = load_model.predict(df_input)

    if disease_type == 3:
        load_model = pickle.load(open('final_thyroid_model.sav', 'rb'))
        prediction = load_model.predict(df_input)
    predict = prediction[0]

    return predict


def predict_heart_disease(df_input, check_flag):
    if check_flag == 0:
        data_split(2)
    prediction = load_model(2, df_input)
    return prediction


def predict_diabetes_disease(df_input, check_flag):
    if check_flag == 0:
        data_split(1)
    prediction = load_model(1, df_input)
    return prediction


def predict_thyroid_disease(df_input, check_flag):
    if check_flag == 0:
        data_split(3)
    prediction = load_model(3, df_input)
    return prediction


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/heart')
def heart():
    return render_template('heart.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/thyroid')
def thyroid():
    return render_template('thyroid.html')


@app.route('/joinheart', methods=['GET', 'POST'])
def join_heart():
    # name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']

    test_input = {'age': [age], 'sex': [gender], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol], 'fbs': [fbs],
                  'restecg': [restecg], 'thalach': [thalach], 'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope],
                  'ca': [ca], 'thal': [thal]}
    df_input = pd.DataFrame(test_input)

    global check_flag

    for i in dir_list:
        if i == "final_heart_model.sav":
            check_flag = 1
            break
        else:
            check_flag = 0

    predict = predict_heart_disease(df_input, check_flag)

    if predict == 0:
        prediction = "Negative"
    else:
        prediction = "Positive"

    result = {
        "output": prediction
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route('/joindiabetes', methods=['GET', 'POST'])
def join_diabetes():
    # name = request.form['name']
    age = request.form['age']
    Gender = request.form['gender']
    Polyuria = request.form['polyuria']
    Polydipsia = request.form['polydipsia']
    sudden_weight_loss = request.form['sudden_weight_loss']
    weakness = request.form['weakness']
    Polyphagia = request.form['polyphagia']
    Genital_thrush = request.form['genital_thrush']
    visual_blurring = request.form['visual_blurring']
    Itching = request.form['itching']
    Irritability = request.form['irritability']
    delayed_healing = request.form['delayed_healing']
    partial_paresis = request.form['partial_paresis']
    muscle_stiffness = request.form['muscle_stiffness']
    Alopecia = request.form['Alopecia']
    Obesity = request.form['Obesity']

    test_input = {'Age': [age], 'Gender': [Gender], 'Polyuria': [Polyuria], 'Polydipsia': [Polydipsia],
                  'sudden_weight_loss': [sudden_weight_loss],
                  'weakness': [weakness], 'Polyphagia': [Polyphagia], 'Genital_thrush': [Genital_thrush],
                  'visual_blurring': [visual_blurring], 'Itching': [Itching], 'Irritability': [Irritability],
                  'delayed_healing': [delayed_healing], 'partial_paresis': [partial_paresis],
                  'muscle_stiffness': [muscle_stiffness],
                  'Alopecia': [Alopecia], 'Obesity': [Obesity]}
    df_input = pd.DataFrame(test_input)
    # print(df_input)

    global check_flag

    for i in dir_list:
        if i == "final_diabetes_model.sav":
            check_flag = 1
            break
        else:
            check_flag = 0

    predict = predict_diabetes_disease(df_input, check_flag)

    if predict == 0:
        prediction = "Negative"
    else:
        prediction = "Positive"

    result = {
        "output": prediction
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route('/jointhyroid', methods=['GET', 'POST'])
def join_thyroid():
    # name = request.form['name']
    age = request.form['age']
    sex = request.form['gender']
    onthyroxine = request.form['onthyroxine']
    queryonthyroxine = request.form['queryonthyroxine']
    antithyroid = request.form['antithyroid']
    sick = request.form['sick']
    pregnant = request.form['pregnant']
    thyroidsurgery = request.form['thyroidsurgery']
    i131treatment = request.form['i131treatment']
    queryhypothyroid = request.form['queryhypothyroid']
    queryhyperthyroid = request.form['queryhyperthyroid']
    lithium = request.form['lithium']
    goitre = request.form['goitre']
    tumour = request.form['tumour']
    hypopituitary = request.form['hypopituitary']
    psych = request.form['psych']
    tshmeasured = request.form['tshmeasured']
    tsh = request.form['tsh']
    t3measured = request.form['t3measured']
    t3 = request.form['t3']
    tt4measured = request.form['tt4measured']
    tt4 = request.form['tt4']
    t4umeasured = request.form['t4umeasured']
    t4u = request.form['t4u']
    ftimeasured = request.form['ftimeasured']
    fti = request.form['fti']
    tbgmeasured = request.form['tbgmeasured']
    tbg = request.form['tbg']

    test_input = {'age': [age], 'sex': [sex], 'onthyroxine': [onthyroxine], 'queryonthyroxine': [queryonthyroxine],
                  'antithyroid': [antithyroid], 'sick': [sick], 'pregnant': [pregnant],
                  'thyroidsurgery': [thyroidsurgery],
                  'i131treatment': [i131treatment], 'queryhypothyroid': [queryhypothyroid],
                  'queryhyperthyroid': [queryhyperthyroid],
                  'lithium': [lithium], 'goitre': [goitre], 'tumour': [tumour], 'hypopituitary': [hypopituitary],
                  'psych': [psych],
                  'tshmeasured': [tshmeasured], 'tsh': [tsh], 't3measured': [t3measured], 't3': [t3],
                  'tt4measured': [tt4measured],
                  'tt4': [tt4], 't4umeasured': [t4umeasured], 't4u': [t4u], 'ftimeasured': [ftimeasured],
                  'fti': [fti],
                  'tbgmeasured': [tbgmeasured], 'tbg': [tbg]}
    df_input = pd.DataFrame(test_input)

    global check_flag

    for i in dir_list:
        if i == "final_thyroid_model.sav":
            check_flag = 1
            break
        else:
            check_flag = 0

    predict = predict_thyroid_disease(df_input, check_flag)

    if predict == 0:
        prediction = "Negative"
    else:
        prediction = "Positive"

    result = {
        "output": prediction
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


if __name__ == '__main__':
    app.run(debug=True)
