
from __future__ import annotations
import numpy as np
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

def train_and_eval(features, labels, test_size=0.3, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models: Dict[str, object] = {
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "MLP": MLPClassifier(max_iter=500, random_state=seed),
        "KNN": KNeighborsClassifier(),
    }
    reports = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        reports[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "report": classification_report(y_test, y_pred, digits=4),
        }
    return reports
