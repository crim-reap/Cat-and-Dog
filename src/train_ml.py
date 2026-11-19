"""Metric + timing helpers for scikit-learn models."""
import time

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def scores_of(model, X):
    """Continuous score for the positive class (Dog) used for ROC-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score):
    return dict(
        Accuracy=accuracy_score(y_true, y_pred),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        F1=f1_score(y_true, y_pred, zero_division=0),
        ROC_AUC=roc_auc_score(y_true, y_score),
    )


def fit_timed(model, X, y):
    t0 = time.time()
    model.fit(X, y)
    return model, time.time() - t0


def evaluate(model, X, y):
    pred = model.predict(X)
    score = scores_of(model, X)
    return compute_metrics(y, pred, score), pred, score
