from process_data import load_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.linear_model import LinearRegression


def training():
    X,y = load_data()

    # if len(X.shape) == 2:
    #     X = X.reshape((X.shape[0],1,X.shape[1]))

    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2, random_state=42)
        
    model = LinearRegression()
    trained_model = model.fit(X_train, Y_train)

    return trained_model
