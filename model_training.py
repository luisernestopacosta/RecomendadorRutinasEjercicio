from process_data import load_data
from sklearn.model_selection import train_test_split
import tensorflow as tf


def training():
    X,y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    
    model = tf.keras.Sequential()
