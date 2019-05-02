
from data_preprocessing.data_preprocessing import get_mnist_data

x_train, x_test, y_train, y_test = get_mnist_data()
print('Data processed')

from keras.layers import Dense, Dropout
from keras.models import Sequential


classifier = Sequential()
classifier.add(Dense(output_dim=400, init='uniform', activation='relu', input_dim=len(x_train[0])))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim=2500, init='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim=2000, init='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim=1500, init='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim=1000, init='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim=500, init='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim=10, init='uniform', activation='softmax'))

classifier.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

classifier.fit(x_train, y_train, batch_size=10, epochs=2)

y_pred = classifier.predict_classes(x_test)


from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, y_pred)

good_predict = 0
bad_predict = 0
for i in range(0, 10):
    for j in range(0, 10):
        if i == j:
            good_predict += matrix[i][j]
        else:
            bad_predict += matrix[i][j]

error_rate = bad_predict / (bad_predict + good_predict)
print('Error rate: ' + str(error_rate))
correct_rate = good_predict / (bad_predict + good_predict)
print('Corrext rate: ' + str(correct_rate))
print(matrix)


wrong = []
for i in range(0, len(y_pred)):
    if y_pred[i] != y_test[i]:
        wrong.append((y_pred[i], y_test[i]))
        print(wrong[-1])