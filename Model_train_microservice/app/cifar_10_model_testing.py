import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from app.db import add_metric,get_metrics

def evaluate_cifar10_model():
    (_, _), (X_test, y_test) = cifar10.load_data()

    # Normalize pixels
    X_test = X_test.astype('float32') / 255.0

    y_test_one_hot = to_categorical(y_test, num_classes=10)
    model = load_model("cifar10_model.h5")

    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_classes = np.argmax(y_test_one_hot, axis=1)

    
    accuracy = accuracy_score(y_test_classes, y_pred)
    f1 = f1_score(y_test_classes, y_pred, average='weighted')
    precision = precision_score(y_test_classes, y_pred, average='weighted')
    recall = recall_score(y_test_classes, y_pred, average='weighted')

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

    add_metric(accuracy,f1,precision,recall)

    print("Model Evaluation Metrics:", metrics)
    return metrics
