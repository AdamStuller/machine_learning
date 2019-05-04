def get_importance(clf, title):
    """
    returns list of importances for individual pixels
    :param clf: classifier, whose importances to return
    :param title: title of graph
    """
    importances = clf.feature_importances_.reshape(28,28)

    import matplotlib.pyplot as plt
    plt.matshow(importances, cmap=plt.cm.hot)
    plt.title(title)
    plt.show()


def main():

    from models_common import load_clf, test, train, save, cross_validate
    from decision_tree_config import __config_values as config

    NAME = config["name"]

    print(f'Model {config["name"]}')
    print('Getting data...')
    from data_preprocessing.data_preprocessing import get_mnist_data
    x_train, x_test, y_train, y_test = get_mnist_data(shorten=config["shorten"],
                                                      normalize=config["normalize"],
                                                      new_arguments=config["new_arguments"])
    print('Data loaded')

    if config["load_first"]:
        print(f"Searching for classifier {NAME}")
        try:
            clf = load_clf(NAME)
            print('Classifier found stored')
        except FileNotFoundError:
            print('Training classifier')
            from sklearn.tree import DecisionTreeClassifier
            classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
            clf = train(classifier, x_train, y_train)
    else:
        print('Training classifier')
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        clf = train(classifier, x_train, y_train)

    matrix, score, err_rate = test(clf, x_test, y_test)
    print("Confusion Matrix: ")
    print(matrix)
    print("Score: " + str(score))
    print("Error rate: " + str(err_rate))

    if not config["new_arguments"] and not config["shorten"]:
        get_importance(clf, "Decision tree pixel importances")

    if config["cross_validate"]:
        print('Crossvalidating....')
        mean = cross_validate(clf, x_train, y_train)
        print('Cross validation mean: ' + str(mean))

    if config["save"]:
        save(classifier, NAME)


if __name__== "__main__":
    main()
