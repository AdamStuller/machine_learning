def train(model, x_train, y_train):
    """
    Trains classifier from given model
    :param model: Model to be trained on given data
    :param x_train: Train dataset
    :param y_train: Labels
    :return:
    trained classifier
    """

    model.fit(x_train, y_train)
    return model


def test(classifier, x_test, y_test):
    """
    Tests given trained classfier on given test dataset
    :param classifier: trained classifier
    :param x_test: test dataset
    :param y_test: test labels
    :return: numpy.array, int, int
    confusion matrix of classifier, accuracy rate and error rate
    """

    y_pred = classifier.predict(x_test)

    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_test, y_pred)
    score = classifier.score(x_test, y_test)
    error_rate = 1 - score

    return matrix, score, error_rate


def save(classifier, name):
    """
    Saves classifier into specified folder
    :param classifier: trained classifier to be saved
    :param name: name of the folder in ./data/classifiers into which clf is to be saved
    """
    from joblib import dump
    import os
    CLF_PATH = os.path.join('data', 'classifiers', f"{name}.joblib")
    dump(classifier, CLF_PATH)


def load_clf(name):
    """
    loads classifier
    :param name: name of the folder from where classifier is to be loaded in ./data/classifiers
    :return:
    """
    from joblib import load
    import os
    CLF_PATH = os.path.join('data', 'classifiers', f"{name}.joblib")
    try:
        return load(CLF_PATH)
    except FileNotFoundError:
        print('No such model exists')
        raise FileNotFoundError


def cross_validate(classifier, x_test, y_test):
    """
    Performs cross validation on given classifier
    :param classifier: classifier on which cross validation is performed, supports all tree of them
    :param x_test: test dataset
    :param y_test: test labels
    :return: int
    mean of all scores
    """

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(classifier, x_test, y_test, cv=5)
    mean = scores.mean()

    return mean
