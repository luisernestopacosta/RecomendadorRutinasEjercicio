from process_data import load_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
from model_training import training
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


prediction = model.predict(X_test)
print(X_test)
mse = mean_squared_error(Y_test, prediction)

print(f"Prediction: {prediction}")
print(f"Mean Squared Error: {mse}")