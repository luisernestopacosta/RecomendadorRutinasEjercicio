import numpy as np
from tensorflow.keras.models import load_model

def predict(model_path, input_data):
    """Load the trained model and make predictions."""
    model = load_model(model_path)
    input_data = np.array(input_data)
    predictions = model.predict(input_data)
    return predictions.tolist()
