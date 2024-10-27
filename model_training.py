from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def training(X, y):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    trained_model = model.fit(X_train, Y_train)

    return trained_model
