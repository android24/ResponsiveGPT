def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_confusion_and_scores(y_true, y_pred):
    """
    y_true: list[bool]
    y_pred: list[bool]
    """
    tp = fp = fn = tn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt and yp:
            tp += 1
        elif (not yt) and yp:
            fp += 1
        elif yt and (not yp):
            fn += 1
        else:
            tn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "total": tp + fp + fn + tn,
    }
