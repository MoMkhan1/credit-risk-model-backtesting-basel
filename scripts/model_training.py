import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

def train_pd_models(X_train, y_train, X_test, y_test, smote=True, save_path="models/best_pd_model.pkl"):
    if smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200),
        "GradientBoosting": GradientBoostingClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, prob)
        results[name] = {"model": model, "roc_auc": auc}

    best_name = max(results, key=lambda x: results[x]["roc_auc"])
    best_model = results[best_name]["model"]
    joblib.dump(best_model, save_path)
    return best_model, best_name, results

def train_lgd_model(X_train, y_train_lgd, save_path="models/lgd_model.pkl"):
    lgd_model = RandomForestRegressor(n_estimators=200)
    lgd_model.fit(X_train, y_train_lgd)
    joblib.dump(lgd_model, save_path)
    return lgd_model