import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

def evaluate_model(model_path, test_data, test_labels):
    """Evaluate the saved model and return accuracy."""
    model = load_model(model_path)
    predictions = (model.predict(test_data) > 0.5).astype("int32")
    accuracy = accuracy_score(test_labels, predictions)
    return {"accuracy": accuracy}
