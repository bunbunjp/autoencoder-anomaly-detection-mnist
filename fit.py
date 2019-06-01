import copy
from typing import Tuple, List

import keras
from keras import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv1D, MaxPooling1D, UpSampling1D, Dense, Flatten, Reshape


def create_model(target: np.ndarray) -> Sequential:
    ae: Sequential = Sequential()
    ae.add(Conv2D(8, (7, 7), activation='relu', padding='same', input_shape=(target.shape[1], target.shape[2], 1)))
    ae.add(MaxPooling2D((7, 7), padding='same'))
    ae.add(Conv2D(4, (1, 1), activation='relu', padding='same'))
    ae.add(MaxPooling2D((1, 1), padding='same'))
    ae.add(Conv2D(4, (1, 1), activation='relu', padding='same'))
    ae.add(MaxPooling2D((1, 1), padding='same'))

    ae.add(Conv2D(4, (1, 1), activation='relu', padding='same'))
    ae.add(UpSampling2D((1, 1)))
    ae.add(Conv2D(4, (1, 1), activation='relu', padding='same'))
    ae.add(UpSampling2D((1, 1)))
    ae.add(Conv2D(8, (7, 7), activation='relu', padding='same'))
    ae.add(UpSampling2D((7, 7)))

    ae.add(Conv2D(1, (7, 7), activation='relu', padding='same'))

    ae.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    ae.summary()
    return ae


def load_normaly_data() -> Tuple[np.ndarray, np.ndarray]:
    load_data: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = mnist.load_data()
    x_train: np.ndarray = load_data[0][0]
    y_train: np.ndarray = load_data[0][1]
    x_test: np.ndarray = load_data[1][0]
    y_test: np.ndarray = load_data[1][1]

    norm_x_index: List[int] = np.where(y_train < 5)[0]
    norm_x: np.ndarray = np.zeros(shape=(len(norm_x_index), x_train.shape[1], x_train.shape[2]), dtype=float)
    norm_x[:] = x_train[norm_x_index]
    norm_x = norm_x.reshape((norm_x.shape[0], norm_x.shape[1], norm_x.shape[2], 1))
    norm_x[:] = norm_x / 255

    test_x_index: List[int] = np.where(y_test < 5)[0]
    test_x: np.ndarray = np.zeros(shape=(len(test_x_index), x_test.shape[1], x_test.shape[2]), dtype=float)
    test_x[:] = x_test[np.where(y_test < 5)]
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))
    test_x[:] = test_x / 255
    return norm_x, test_x


def main():
    norm_x, test_x = load_normaly_data()
    ae: Sequential = create_model(norm_x)
    stack = ae.fit(x=norm_x, y=norm_x, verbose=1, epochs=5, validation_data=(test_x, test_x), batch_size=256)
    ae.save(filepath='fitted.h5', overwrite=True)

    plt.subplot(1, 2, 1)
    plt.plot(range(5), stack.history['acc'], label="acc")
    plt.plot(range(5), stack.history['val_acc'], label="val_acc")

    plt.subplot(1, 2, 2)
    plt.plot(range(5), stack.history['loss'], label="loss")
    plt.plot(range(5), stack.history['val_loss'], label="val_loss")
    plt.show()


if __name__ == '__main__':
    main()
