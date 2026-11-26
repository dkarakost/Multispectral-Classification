import numpy as np
from sklearn.metrics import jaccard_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.base import clone


def macro_iou(y_true, y_pred):
    return jaccard_score(y_true, y_pred, average="macro")


def per_class_iou(y_true, y_pred, class_ids):
    """IoU for each class. Returns a dict {class_id: iou}."""
    scores = jaccard_score(y_true, y_pred, labels=class_ids, average=None, zero_division=0)
    return dict(zip(class_ids, scores))


def normalized_confusion(y_true, y_pred, class_ids):
    """Confusion matrix where each row sums to 1 (share of each true class)."""
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    return cm / cm.sum(axis=1, keepdims=True)


def spatial_cv(model, X, y, groups, n_splits=5):
    """Cross-validation where whole spatial blocks are held out.

    Returns the list of macro IoU scores, one per fold.
    """
    scores = []
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(X, y, groups):
        fold_model = clone(model)            # fresh model every fold
        fold_model.fit(X[train_idx], y[train_idx])
        pred = fold_model.predict(X[val_idx])
        scores.append(macro_iou(y[val_idx], pred))
    return scores