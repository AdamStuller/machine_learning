def visualize_error_rate(clf, x_test, y_test):
    import pandas as pd
    df_labels = pd.DataFrame()
    df_labels["labels"] = y_test
    df_labels["assigned"] = clf.predict(x_test)
    df_labels["success"] = (df_labels["assigned"] == y_test)
    df_labels = df_labels.rename(columns={0: "label"})

    for name, group in df_labels.groupby("labels"):
        frac = sum(group["success"]) / len(group)
        print("Success rate for labeling digit %i was %f " % (name, frac))

    import matplotlib.pyplot as plt
    import numpy as np
    fig, axs = plt.subplots(5, 2, sharex=False, sharey=True, figsize=(12, 12))
    axs = axs.flatten()

    for name, group in df_labels.groupby("labels"):
        group = group[group["assigned"] != name]
        ax = axs[name - 1]
        ax.hist(group["assigned"], label=("digit %i" % name),
                bins=np.arange(1, 11, 1) + 0.5)
        ax.set_xlim([0, 11])
        ax.legend(loc="upper right")
        ax.yaxis.set_visible(False)

    fig.show()


def main():

    from models_common import load_clf, test, train, save, cross_validate
    from neural_network_config import __config_values as config

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
            from sklearn.neural_network import MLPClassifier
            classifier = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                                       solver='sgd', verbose=10, tol=1e-4, random_state=1,
                                       learning_rate_init=.1)
            clf = train(classifier, x_train, y_train)
    else:
        print('Training classifier')
        from sklearn.neural_network import MLPClassifier
        classifier = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                                   solver='sgd', verbose=10, tol=1e-4, random_state=1,
                                   learning_rate_init=.1)
        clf = train(classifier, x_train, y_train)

    matrix, score, err_rate = test(clf, x_test, y_test)
    print("Confusion Matrix: ")
    print(matrix)
    print("Score: " + str(score))
    print("Error rate: " + str(err_rate))

    visualize_error_rate(clf, x_test, y_test)

    if config["cross_validate"]:
        print('Crossvalidating....')
        mean = cross_validate(clf, x_test, y_test)
        print('Cross validation mean: ' + str(mean))

    if config["save"]:
        save(classifier, NAME)


if __name__== "__main__":
    main()


