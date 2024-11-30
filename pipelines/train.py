import os
from models.model_definition import build_model
from pipelines.processing_data import normalize_data

def train_model(data, labels, save_dir="models/trained"):
    os.makedirs(save_dir, exist_ok=True)
    input_shape = (data.shape[1],)
    model = build_model(input_shape)
    data = normalize_data(data)
    model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.2)
    model.save(os.path.join(save_dir, "model.h5"))
    
    return model
