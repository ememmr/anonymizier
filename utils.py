from sklearn.model_selection import train_test_split


def splitter(data, train, test):
    if train + test == 1.0:
        return train_test_split(data, train_size=train, test_size=test)
    else:
        print("Use all the data! train + test must be 1.0")
