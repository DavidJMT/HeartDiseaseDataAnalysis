import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

def evaluate_model(model, model_name, feature_columns, param_grid, use_grid, X_train, X_test, y_train, y_test):

    feature_names = {
        'age':      'Age',
        'sex':      'Sex',
        'cp':       'Chest Pain Type',
        'trestbps': 'Resting Blood Pressure',
        'chol':     'Cholesterol',
        'fbs':      'Fasting Blood Sugar',
        'restecg':  'Resting ECG',
        'thalach':  'Max Heart Rate',
        'exang':    'Exercise Angina',
        'oldpeak':  'ST Depression',
        'slope':    'ST Slope',
        'ca':       'Blocked Vessels',
        'thal':     'Thalassemia Test'
    }
    if use_grid == True: 
        grid_search = GridSearchCV(  #Exhaustive search over specified parameter values for an estimator.
            estimator=model,
            param_grid=param_grid,
            cv=5,              # cross-validation with 5 folds
            scoring='accuracy', # percentage of correct predictions
            n_jobs=-1,         # uses all cpu cores
            verbose=1          # mostra o progresso
        )
        grid_search.fit(X_train, y_train) # executes with parameters
        trained_model = grid_search.best_estimator_ 
        print('Best parameters:' , grid_search.best_params_);
        predictions = trained_model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        trained_model = model
        predictions = trained_model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))


    if (model_name == 'RandomForest'):
            
        importances = pd.DataFrame({
        'feature': feature_columns,
        'importance': trained_model.feature_importances_
        })

        plt.figure(figsize=(20, 10))
        plot_tree(
            trained_model.estimators_[0],  # first tree from the forest
            feature_names=list(feature_names.values()),  # already defined name
            class_names=['Healthy', 'Disease'],
            filled=True,       # colors filled by class
            rounded=True,      # rounder courners
            max_depth=3        # only shows 3 levels
        )
        plt.title("Random Forest Tree (Depth 3)")
        plt.show()

    if(model_name == 'LogisticRegression'):
        lr_model = trained_model.named_steps['model']

        importances = pd.DataFrame({
        'feature': feature_columns,
        'importance': lr_model.coef_[0] 
        })

    importances['feature'] = importances['feature'].map(feature_names)
    importances = importances.sort_values('importance', ascending=False)

    print(f"\nFeature Importance — {model_name}")
    print(importances.to_string(index=False))

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Disease'],
                yticklabels=['Healthy', 'Disease'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.show()