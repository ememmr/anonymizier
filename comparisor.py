from utils import splitter


def measure(data_orig, data_anon, train_size, model, validator):
    test_size = 1 - train_size

    # Let's split data for each case
    train_anon, test_anon = splitter(data_anon, train_size, test_size)
    train_orig, test_orig = splitter(data_orig, train_size, test_size)

    # Then fit model and predict
    model.fit(X=train_anon.iloc[:, :-1], y=train_anon.iloc[:, -1])
    predictions_anon = model.predict(X=test_anon.iloc[:, :-1])

    model.fit(X=train_orig.iloc[:, :-1], y=train_orig.iloc[:, -1])
    predictions_orig = model.predict(X=test_orig.iloc[:, :-1])

    # Compare results, who has worse result?
    error_anon = validator(test_anon.iloc[:, -1], predictions_anon)
    error_orig = validator(test_orig.iloc[:, -1], predictions_orig)

    print("For data anonimizied, the error is: " + str(error_anon))
    print("For full data, the error is: " + str(error_orig))

    return error_anon, error_orig
