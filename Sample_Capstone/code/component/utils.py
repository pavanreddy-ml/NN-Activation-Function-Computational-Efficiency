from config import *
import pandas as pd
import os
from datetime import datetime
from tensorflow.keras.datasets import mnist, cifar10
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer, load_iris, load_digits, \
    load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.datasets import make_regression
import numpy as np


def load_file(new_file=False):
    if not os.path.isfile(RESULTS_PATH) or new_file:
        results_df = pd.DataFrame(columns=COLS)
        results_df.to_csv(RESULTS_PATH, index=False)
    else:
        results_df = pd.read_csv(RESULTS_PATH)
    return results_df


def create_dataframe(data, dataset_name, activation):
    rows = []
    for i in range(len(data["continuous_data"])):
        new_row = {
            "dataset": dataset_name,
            "activation": activation,
            "run": data["run"][i],
            "continuous_forward_profile_time": data["continuous_data"][i],
            "continuous_cpu_time": data["continuous_train_time"][i],
            "continuous_performance_metric": data["continuous_loss"][i],
            "piecewise_forward_profile_time": data["piecewise_data"][i],
            "piecewise_cpu_time": data["piecewise_train_time"][i],
            "piecewise_performance_metric": data["piecewise_loss"][i],
        }
        rows.append(new_row)
    temp_df = pd.DataFrame(rows, columns=COLS)
    return temp_df


def concat_results(df, results_df):
    return pd.concat([results_df, df], ignore_index=True)


def write_results_to_csv(df, cache=True):
    if cache and os.path.isfile(RESULTS_PATH):
        cache_path = os.path.join(os.path.dirname(RESULTS_PATH),
                                  f'cache/results_cache_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
        os.rename(RESULTS_PATH, cache_path)
        df.to_csv(RESULTS_PATH, index=False)
    else:
        df.to_csv(RESULTS_PATH, index=False)


def preprocess_data(dataset, num_samples=None):
    if dataset == 'mnist_digits':
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        if num_samples is not None:
            num_samples = max(num_samples, len(train_images))
            train_images = train_images[:num_samples]
            train_labels = train_labels[:num_samples]

        num_pixels = train_images.shape[1] * train_images.shape[2]
        train_images = train_images.reshape((train_images.shape[0], num_pixels)).astype('float32')
        test_images = test_images.reshape((test_images.shape[0], num_pixels)).astype('float32')

        one_hot = np.zeros((train_labels.shape[0], len(np.unique(train_labels))))
        one_hot[np.arange(train_labels.shape[0]), train_labels.flatten()] = 1
        train_labels = one_hot

        train_images /= 255.0
        test_images /= 255.0

        return train_images, train_labels, (train_images.shape[1], 100, train_labels.shape[1]), "cce"

    if dataset == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

        if num_samples is not None:
            num_samples = max(num_samples, len(train_images))
            train_images = train_images[:num_samples]
            train_labels = train_labels[:num_samples]

        num_pixels = train_images.shape[1] * train_images.shape[2] * train_images.shape[3]
        train_images = train_images.reshape((train_images.shape[0], num_pixels)).astype('float32')
        test_images = test_images.reshape((test_images.shape[0], num_pixels)).astype('float32')

        train_images /= 255.0
        test_images /= 255.0

        return train_images, train_labels.reshape(-1, 1), (train_images.shape[1], 100, train_labels.shape[1]), "cce"

    if dataset == 'synthetic':
        X, y = make_regression(n_samples=50000, n_features=10, noise=0.1, random_state=42)
        y = y.reshape(-1, 1)

        standard_scaler = StandardScaler()
        X = standard_scaler.fit_transform(X)

        scaler = MinMaxScaler(feature_range=(0, 1))
        y = scaler.fit_transform(y)

        return X, y, (X.shape[1], 100, 1), "mse"

    if dataset == 'california':
        data = fetch_california_housing()
        loss = "mse"
    elif dataset == 'diabetes':
        data = load_diabetes()
        loss = "mse"
    elif dataset == 'breast_cancer':
        data = load_breast_cancer()
        loss = "bce"
    elif dataset == 'iris':
        data = load_iris()
        loss = "cce"
    elif dataset == 'digits':
        data = load_digits()
        loss = "cce"
    elif dataset == 'wine':
        data = load_wine()
        loss = "cce"
    else:
        raise ValueError("Dataset not supported")

    X = data.data
    y = data.target

    standard_scaler = StandardScaler()
    X = standard_scaler.fit_transform(X)

    if loss == "mse" or loss == "bce":
        scaler = MinMaxScaler(feature_range=(0, 1))
        y = scaler.fit_transform(y.reshape(-1, 1))
    elif loss == "cce":
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y.reshape(-1, 1))
    else:
        raise ValueError("Loss type not supported")

    return X, y, (X.shape[1], 100, y.shape[1]), loss
