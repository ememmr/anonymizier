from utils import splitter


def measure(data, train_size, model, validator):
    test_size = 1 - train_size

    # Let's split data for each case
    train, test = splitter(data, train_size, test_size)

    # Then fit model and predict
    model.fit(X=train.iloc[:, :-1], y=train.iloc[:, -1])
    predictions = model.predict(X=test.iloc[:, :-1])

    # Compare results, who has worse result?
    error = validator(test.iloc[:, -1], predictions)

    print("The error is: " + str(error))

    return error
