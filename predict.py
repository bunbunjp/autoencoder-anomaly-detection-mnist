from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.datasets import mnist
from keras.losses import mean_squared_error
from sklearn.metrics import mean_squared_error

from fit import create_model, load_normaly_data


def load_abnormaly_data() -> Tuple[np.ndarray, np.ndarray]:
    load_data: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = mnist.load_data()
    x_train: np.ndarray = load_data[0][0]
    y_train: np.ndarray = load_data[0][1]
    x_test: np.ndarray = load_data[1][0]
    y_test: np.ndarray = load_data[1][1]

    norm_x_index: List[int] = np.where(y_train >= 5)[0]
    norm_x: np.ndarray = np.zeros(shape=(len(norm_x_index), x_train.shape[1], x_train.shape[2]), dtype=float)
    norm_x[:] = x_train[norm_x_index]
    norm_x = norm_x.reshape((norm_x.shape[0], norm_x.shape[1], norm_x.shape[2], 1))
    norm_x[:] = norm_x / 255

    test_x_index: List[int] = np.where(y_test >= 5)[0]
    test_x: np.ndarray = np.zeros(shape=(len(test_x_index), x_test.shape[1], x_test.shape[2]), dtype=float)
    test_x[:] = x_test[np.where(y_test >= 5)]
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))
    test_x[:] = test_x / 255
    return norm_x, test_x


def convert_3dim_img(img: np.ndarray) -> np.ndarray:
    gray_img: np.ndarray = np.zeros(shape=img.shape, dtype=int)
    gray_img[:, :] = 255 * img
    gray_img[255 < gray_img] = 255
    color_img: np.ndarray = np.zeros(shape=(img.shape[0], img.shape[1], 3), dtype=int)
    color_img[:, :, 0] = gray_img
    color_img[:, :, 1] = gray_img
    color_img[:, :, 2] = gray_img
    return color_img


def main():
    x_norm, norm_test = load_normaly_data()
    x_abno, abno_test = load_abnormaly_data()

    ae: Sequential = create_model(target=x_norm)
    ae.load_weights(filepath='fitted.h5')

    order_length: int = 25
    column: Tuple[int, int] = (5, order_length / 5)
    indexes: np.ndarray = np.arange(start=0, stop=norm_test.shape[0], dtype=int)
    target: np.ndarray = norm_test[np.random.choice(indexes, order_length, replace=False)]
    result: np.ndarray = ae.predict(x=target)
    target = np.reshape(a=target, newshape=(target.shape[0], target.shape[1], target.shape[2]))
    result = np.reshape(a=result, newshape=(result.shape[0], result.shape[1], result.shape[2]))

    plt.figure(figsize=(7, 6))
    for counter, img in enumerate(target):
        plt.subplot(column[0], column[1], counter + 1)
        plt.imshow(convert_3dim_img(img=img))
        plt.title(" ")
    plt.show()

    norm_mse_result: np.ndarray = np.zeros(shape=result.shape[0], dtype=float)
    plt.figure(figsize=(7, 6))
    for counter, img in enumerate(result):
        mse_value: float = mean_squared_error(y_true=target[counter], y_pred=img)
        plt.subplot(column[0], column[1], counter + 1)
        norm_mse_result[counter] = mse_value
        color_img: np.ndarray = convert_3dim_img(img=img)
        if mse_value > 0.02:
            color_img[:, :, 0] = 255
        plt.title('{0:.4f}'.format(mse_value))
        plt.imshow(color_img)
    plt.show()

    indexes: np.ndarray = np.arange(start=0, stop=abno_test.shape[0], dtype=int)
    target: np.ndarray = abno_test[np.random.choice(indexes, order_length, replace=False)]
    result: np.ndarray = ae.predict(x=target)
    target = np.reshape(a=target, newshape=(target.shape[0], target.shape[1], target.shape[2]))
    result = np.reshape(a=result, newshape=(result.shape[0], result.shape[1], result.shape[2]))

    plt.figure(figsize=(7, 6))
    for counter, img in enumerate(target):
        plt.subplot(column[0], column[1], counter + 1)
        plt.imshow(convert_3dim_img(img=img))
        plt.title(" ")
    plt.show()

    abno_mse_result: np.ndarray = np.zeros(shape=result.shape[0], dtype=float)
    plt.figure(figsize=(7, 6))
    for counter, img in enumerate(result):
        mse_value: float = mean_squared_error(y_true=target[counter], y_pred=img)
        abno_mse_result[counter] = mse_value
        plt.subplot(column[0], column[1], counter + 1)
        color_img: np.ndarray = convert_3dim_img(img=img)
        if mse_value > 0.02:
            color_img[:, :, 0] = 255
        plt.title('{0:.4f}'.format(mse_value))
        plt.imshow(color_img)

    plt.show()

    print('norm mse {0:.5f}, {1:.5f}, {2:.5f}'.format(norm_mse_result.mean(), norm_mse_result.max(),
                                                      norm_mse_result.min()))
    print('abno mse {0:.5f}, {1:.5f}, {2:.5f}'.format(abno_mse_result.mean(), abno_mse_result.max(),
                                                      abno_mse_result.min()))


if __name__ == '__main__':
    main()
