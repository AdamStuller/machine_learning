def train(clf, x_train, y_train):

    clf.fit(x_train, y_train)
    return clf


def test(classifier, x_test, y_test):

    y_pred = classifier.predict(x_test)

    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_test, y_pred)
    score = classifier.score(x_test, y_test)
    error_rate = 1 - score

    return matrix, score, error_rate


def save(classifier, name):
    from joblib import dump
    import os
    CLF_PATH = os.path.join('data', 'classifiers', f"{name}.joblib")
    dump(classifier, CLF_PATH)


def load_clf(name):
    from joblib import load
    import os
    CLF_PATH = os.path.join('data', 'classifiers', f"{name}.joblib")
    try:
        return load(CLF_PATH)
    except FileNotFoundError:
        print('No such model exists')
        raise FileNotFoundError


def cross_validate(classifier, x_test, y_test):

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(classifier, x_test, y_test, cv=5)
    mean = scores.mean()

    return mean
