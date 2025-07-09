import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten


def create_cnn_model():
    model = Sequential(
        [
            Input(shape=(28, 28, 1), name="input_28_28_1_f"),
            Conv2D(28, (1, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(28 * 2, (1, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(28 * 2, (1, 3), activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train(
    model,
    X,
    y,
    X_val=None,
    y_val=None,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
):
    hist = model.fit(
        X,
        y,
        validation_data=(X_val, y_val)
        if X_val is not None and y_val is not None
        else None,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )
    return model, hist


def predict(model, X):
    y_pred = model.predict(X).flatten()
    return y_pred
