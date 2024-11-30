from sklearn.metrics import accuracy_score
import tensorflow as tf


def evaluate_model(model_path, test_data, test_labels):
    """Evaluate the saved model and return accuracy."""
    model = tf.keras.models.load_model(model_path)
    predictions = (model.predict(test_data) > 0.5).astype("int32")
    accuracy = accuracy_score(test_labels, predictions)
    return {"accuracy": accuracy}
