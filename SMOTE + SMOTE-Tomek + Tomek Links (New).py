import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from numpy import mean
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import auc, precision_recall_curve
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.metrics import roc_auc_score, f1_score
os.chdir("/Users/Sarah/Desktop")
csv_read = pd.read_csv("blood_markers.csv")

csv_read.dropna(inplace=True)
csv_read = csv_read[~csv_read['Group'].isin(['analysis error/unclassifiable', 'insuff follow-up'])]
y = csv_read['Group']
y = y.replace({'single seizure': 0, 'epilepsy': 1, 'PSE': 1})
X = csv_read.drop(['Group', 'patient_number', 'age_at_test'], axis=1) #I removed age_at_test
X['Gender'] = X['Gender'].replace({'Male': 0, 'Female': 1})


skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)


def roc_auc(clf, X, y):
    auc = roc_auc_score(y, clf.predict(X))
    return auc


def imbalanced_cross_validation_score(clf, X, y, cv, scoring, sampler):
    cv_score = 0.
    train_score = 0.
    test_score = 0.
    for train_idx, test_idx in skf.split(X, y):
        Xfold_train_sampled, yfold_train_sampled = sampler.fit_resample(X.iloc[train_idx], y.iloc[train_idx])
        clf.fit(Xfold_train_sampled, yfold_train_sampled)

        train_score = scoring(clf, Xfold_train_sampled, yfold_train_sampled)
        test_score = scoring(clf, X.iloc[test_idx], y.iloc[test_idx])

        print("Train ROC AUC: %.2f Test ROC AUC: %.2f" % (train_score, test_score))
        cv_score += test_score

    return cv_score / cv

cv = 5
rf = RandomForestClassifier()
svm = SVC(probability=True)
xgb = XGBClassifier(use_label_encoder =False)
mlp = MLPClassifier()

#print("SMOTE")
#score = imbalanced_cross_validation_score(svm, X, y, cv, roc_auc, SMOTE())
#print("Cross-validated ROC AUC score: %.2f"%score)

print("SMOTE-Tomek")
score = imbalanced_cross_validation_score(mlp, X, y, cv, roc_auc, SMOTETomek())
print("Cross-validated AUPRC score: %.2f"%score)

#print("Tomek-Links")
#score = imbalanced_cross_validation_score(mlp, X, y, cv, roc_auc, TomekLinks())
#print("Cross-validated AUPRC score: %.2f"%score)




