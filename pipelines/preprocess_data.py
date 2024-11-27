import numpy as np

def normalize_data(data):
    """Normalize data to zero mean and unit variance."""

    # ToDo: Do a better normalization of the data
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
