import os
import pandas as pd
os.chdir("/Users/Sarah/Desktop")
df = pd.read_csv("blood_markers.csv")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report

#Prepare data
df.dropna(inplace=True)
csv_read = df[~df['Group'].isin(['analysis error/unclassifiable', 'insuff follow-up'])]
y = csv_read['Group']
y = y.replace({'single seizure': 0, 'epilepsy': 1, 'PSE': 1})
X = csv_read.drop(['Group', 'patient_number'], axis=1)
X['Gender'] = X['Gender'].replace({'Male': 0, 'Female': 1})

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

#Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

XGBmodel = XGBClassifier()
RFmodel = RandomForestClassifier()
NNmodel = MLPClassifier()
SVMmodel = SVC()

models = [XGBmodel, RFmodel, SVMmodel, NNmodel]
for model in models:
    model.fit(X_train_scaled, y_train)

accuracy = [metrics.accuracy_score(y_test, model.predict(X_test_scaled)) for model in models]
auc_scores = [roc_auc_score(y_test, model.predict(X_test_scaled)) for model in models]
f1_scores = [f1_score(y_test, model.predict(X_test_scaled)) for model in models]

max_index = auc_scores.index(max(auc_scores))
model = models[max_index]
print(accuracy)
print(auc_scores)
print(f1_scores)
print('Top performing model', model)