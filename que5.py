import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Helper function to evaluate the model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted')
    }

def cross_validation(model, X, y, folds):
    accuracies = cross_val_score(model, X, y, cv=folds, scoring='accuracy')
    precisions = cross_val_score(model, X, y, cv=folds, scoring='precision_weighted')
    recalls = cross_val_score(model, X, y, cv=folds, scoring='recall_weighted')
    f1s = cross_val_score(model, X, y, cv=folds, scoring='f1_weighted')
    return {
        'Accuracy': np.mean(accuracies),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'F1-Score': np.mean(f1s)
    }

# Load datasets
iris = datasets.load_iris()
cancer = datasets.load_breast_cancer()

datasets_list = [
    ('Iris', iris.data, iris.target),
    ('Breast Cancer', cancer.data, cancer.target)
]

# Classifiers
models = {
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Process each dataset
results = {}

for dataset_name, X, y in datasets_list:
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    results[dataset_name] = {}
    
    for model_name, model in models.items():
        results[dataset_name][model_name] = {}
        
        # Holdout 80/20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        results[dataset_name][model_name]['Holdout 80/20'] = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Holdout 66.6/33.3
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42, stratify=y)
        results[dataset_name][model_name]['Holdout 66.6/33.3'] = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # 10-fold Cross-Validation
        results[dataset_name][model_name]['10-Fold CV'] = cross_validation(model, X, y, folds=10)
        
        # 5-fold Cross-Validation
        results[dataset_name][model_name]['5-Fold CV'] = cross_validation(model, X, y, folds=5)

# Display the results
for dataset_name, model_results in results.items():
    print(f"\n=== Results for {dataset_name} Dataset ===\n")
    for model_name, evaluations in model_results.items():
        print(f"\n-- {model_name} --")
        df = pd.DataFrame(evaluations).T
        print(df)