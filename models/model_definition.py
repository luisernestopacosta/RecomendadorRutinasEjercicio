import tensorflow as tf


def build_model(input_shape):
    """Defines and compiles the neural network."""

    # ToDo: Hacer una definición correcta del modelo en base a las predicciones que se van a hacer
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=input_shape),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
