import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from model_utils import evaluate_model

df = pd.read_csv("heart.csv")   

print(df['target'].value_counts())

print(df.head())
print(df.describe())
print(df.isnull().sum())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

parameters = {
    'n_estimators': [100, 200, 300],        # number of trees in the forest
    'max_depth': [None, 5, 10, 20],         # how deep each tree can grow
    'min_samples_split': [0.02, 0.05, 0.1], # min % of samples needed to split a node, first decision
    'min_samples_leaf': [0.01, 0.02, 0.05]  # min % of samples required at each leaf (final node)
}

#evaluate_model(RandomForestClassifier(random_state=42), 'RandomForest', X.columns, parameters, True, X_train, X_test, y_train, y_test)
evaluate_model(RandomForestClassifier(random_state=42), 'RandomForest', X.columns, None, False, X_train, X_test, y_train, y_test)
