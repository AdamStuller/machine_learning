def get_number_of_non_zero_right(dataset):
    """
    Counts number of all pixels whose value is above 100 and are in right half of image
    :param dataset: Dataframe on which calculation is dne
    :return: int
        Number of those pixels
    """
    x = dataset.iloc[:, 1:785].values

    import numpy as np
    n_zeros_list = []
    for row in x:
        number = np.array(row)
        n_nonzeros = 0
        for index, pixel in enumerate(number):
            if (not ((index // 28) & 1)) and pixel > 100:
                n_nonzeros += 1
        n_zeros_list.append(n_nonzeros)

    dataset['left_pixels'] = n_zeros_list


def get_number_of_non_zero_left(dataset):
    """
    Counts number of all pixels whose value is above 100 and are in left half of image
    :param dataset: Dataframe on which calculation is dne
    :return: int
            Number of those pixels
    """
    x = dataset.iloc[:, 1:785].values

    import numpy as np
    n_zeros_list = []
    for row in x:
        number = np.array(row)
        n_nonzeros = 0
        for index, pixel in enumerate(number):
            if (index // 28) & 1 and pixel > 100:
                n_nonzeros += 1
        n_zeros_list.append(n_nonzeros)

    dataset['right_pixels'] = n_zeros_list


def get_number_of_non_zero_down(dataset):
    """
    Counts number of all pixels whose value is above 100 and are in lower half of image
    :param dataset: Dataframe on which calculation is dne
    :return: int
        Number of those pixels
    """
    x = dataset.iloc[:, 1:785].values

    import numpy as np
    n_zeros_list = []
    for row in x:
        number = np.array(row)
        n_nonzeros = 0
        for index, pixel in enumerate(number):
            if index > len(number) // 2 and pixel > 100:
                n_nonzeros += 1
        n_zeros_list.append(n_nonzeros)
    dataset['down_pixels'] = n_zeros_list
    return dataset


def get_number_of_non_zero_up(dataset):
    """
    Counts number of all pixels whose value is above 100 and are in upper half of image
    :param dataset: Dataframe on which calculation is dne
    :return: int
        Number of those pixels
    """
    x = dataset.iloc[:, 1:785].values

    import numpy as np
    n_zeros_list = []
    for row in x:
        number = np.array(row)
        n_nonzeros = 0
        for index, pixel in enumerate(number):
            if index <= len(number) // 2 and pixel > 100:
                n_nonzeros += 1
        n_zeros_list.append(n_nonzeros)
    dataset['up_pixels'] = n_zeros_list
    return dataset




