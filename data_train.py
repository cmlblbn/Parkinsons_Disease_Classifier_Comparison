from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from xgboost import XGBClassifier

import pandas as pd
import numpy as np


classification = {1 : "infected", 0 : "Healthy"}

def trainer(x_train, y_train):
    models = {"XGBoost": 0, "Logistic_Regression": 0, "SVM_Classifier": 0, "Desicion_Tree": 0}
    model_xgb = XGBClassifier()
    models["XGBoost"] = model_xgb
    model_logistic = LogisticRegression()
    models["Logistic_Regression"] = model_logistic
    model_svm = SVC()
    models["SVM_Classifier"] = model_svm
    model_desicion_tree = DecisionTreeClassifier()
    models["Desicion_Tree"] = model_desicion_tree

    for model in models:
        trainer = models[model].fit(x_train, y_train)
        models[model] = trainer

    return models


def calculate_accuraty(models, x_test, y_test):
    for model in models:
        print(model, "accuraty is: %", models[model].score(x_test, y_test) * 100, "\n")



def make_prediction(models, x_test, y_test):
    for model in models:
        mistake = 0
        predict = models[model].predict(x_test)
        y_test = list(y_test)
        print("\n\n\n")
        print(model, "Model Classification predict -- Real Data is\n")
        for i in range(len(predict)):
            if predict[i] != y_test[i]:
                print("     ", classification[predict[i]], ": ------- :", classification[y_test[i]], "   Mistake!!")
                mistake += 1
            else:
                print("     ", classification[predict[i]], ": ------- :", classification[y_test[i]])

        print("\n Total Mistake is :", mistake)

