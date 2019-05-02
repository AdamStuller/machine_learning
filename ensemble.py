def main():
    from models_common import train
    from ensemble_config import __config_values as config

    print(f'Model {config["name"]}')
    print('Getting data...')
    from data_preprocessing.data_preprocessing import get_mnist_data
    x_train, x_test, y_train, y_test = get_mnist_data(shorten=config["shorten"],
                                                      normalize=config["normalize"],
                                                      new_arguments=config["new_arguments"])

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    dt_model = DecisionTreeClassifier(criterion='entropy', random_state=0)
    rf_model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    ann_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                               solver='sgd', verbose=10, tol=1e-4, random_state=1,
                               learning_rate_init=.1)

    dt_clf = train(dt_model, x_train, y_train)
    rf_clf = train(rf_model, x_train, y_train)
    ann_clf = train(ann_model, x_train, y_train)

    print('Classifiers trained')

    from sklearn.ensemble import VotingClassifier
    ensemble_clf = VotingClassifier(estimators=[('decision_tree', dt_clf),
                                                ('random_forest', rf_clf),
                                                ('neural_network', ann_clf)],
                                    voting=config["voting"])
    ensemble_clf.fit(x_train, y_train)
    y_pred = ensemble_clf.predict(x_test)

    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_test, y_pred)

    if config["cross_validate"]:
        print('Cross Validation')
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(ensemble_clf, x_test, y_test, cv=5)
        mean = scores.mean()
        print('MEAN: ' + str(mean))
        print(matrix)


if __name__ == "__main__":
    main()


