def get_mnist_data(shorten=False, normalize=True, new_arguments=False):
    """
    Reads dataset from file in ./data/datasets folder and commits specified operations on it
    :param shorten: determines wheter or not to shorten the dataset
    :param normalize:  determines whwter or not to normalize it
    :param new_arguments: determines wheter to add new arguments
    :return: pandas.DataFrame
        updated dataset
    """
    import pandas as pd
    import numpy as np
    import os

    if new_arguments:
        TEST_DATA = os.path.join('data', 'datasets', 'new_mnist_test.csv')
        TRAIN_DATA = os.path.join('data', 'datasets', 'new_mnist_train.csv')
    else:
        TEST_DATA = os.path.join('data', 'datasets', 'mnist_test.csv')
        TRAIN_DATA = os.path.join('data', 'datasets', 'mnist_train.csv')

    dataset = pd.read_csv(TRAIN_DATA)
    test_dataset = pd.read_csv(TEST_DATA)

    shorten_len = 0

    if shorten:
        corr_matrix = dataset.corr()
        pointless_col = corr_matrix.query('label == "NaN"')["label"].keys()
        dataset = dataset.drop(columns=pointless_col)
        test_dataset = test_dataset.drop(columns=pointless_col)
        shorten_len = len(pointless_col)

    x_train = dataset.iloc[:, 1:].values
    y_train = dataset.iloc[:, 0].values
    x_test = test_dataset.iloc[:, 1:].values
    y_test = test_dataset.iloc[:, 0].values

    if normalize:
        from sklearn.preprocessing import normalize
        X2 = normalize(x_train[:, 0:784 - shorten_len], axis=0, norm='max')
        X2_test = normalize(x_test[:, 0:784 - shorten_len], axis=0, norm='max')
        if len(x_train[0]) > 784 - shorten_len:
            x_train = np.concatenate((X2, x_train[:, 784 - shorten_len:]), axis=1)
            x_test = np.concatenate((X2_test, x_test[:, 784 - shorten_len:]), axis=1)
        else:
            x_train = X2
            x_test = X2_test

    return x_train, x_test, y_train, y_test



