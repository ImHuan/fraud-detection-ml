from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(model, X_test, y_test, threshold):

    probs = model.predict_proba(X_test)[:, 1]

    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("Threshold:", threshold)
    print("Accuracy :", acc)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)