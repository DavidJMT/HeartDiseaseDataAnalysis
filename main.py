import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from model_utils import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

df = pd.read_csv("heart.csv")   

print(df['target'].value_counts())

print(df.head())
print(df.describe())
print(df.isnull().sum())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest
parameters_rf = {
    'n_estimators': [100, 200, 300],        # number of trees in the forest
    'max_depth': [None, 5, 10, 20],         # how deep each tree can grow
    'min_samples_split': [0.02, 0.05, 0.1], # min % of samples needed to split a node, first decision
    'min_samples_leaf': [0.01, 0.02, 0.05]  # min % of samples required at each leaf (final node)
}

evaluate_model(RandomForestClassifier(random_state=42), 'RandomForest', X.columns, parameters_rf, True, X_train, X_test, y_train, y_test)

#Logistic Regression
param_grid_lr = {
    'model__C': [0.01, 0.1, 1, 10],        # regularization strength
    'model__l1_ratio': [0, 1],              # 0 = L2, 1 = L1 - type of regularization
    'model__solver': ['liblinear'],         # optimization algorithm
    'model__max_iter': [1000]
}

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
])

evaluate_model(lr_pipeline,'LogisticRegression', X.columns, param_grid_lr, True, X_train, X_test, y_train, y_test);

param_grid_svm = {
    'model__C' : [0.01, 0.1, 1, 10],
    'model__kernel': ['linear','rbf'],
    'model__gamma': ['auto','scale']
}

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(random_state=42))
])

evaluate_model(svm_pipeline,'SVM', X.columns, param_grid_svm, True, X_train, X_test, y_train, y_test);