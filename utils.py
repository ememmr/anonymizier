from sklearn.model_selection import train_test_split


def splitter(data, train, test):
    if train + test == 1.0:
        return train_test_split(data, train_size=train, test_size=test)
    else:
        print("Use all the data! train + test must be 1.0")


def exampleCleaner(df):
    df['age'] = df['age'].astype(int)
    df['cp'] = df['cp'].replace('\*', '', regex=True).astype(int)
    df['gender'] = df['gender'].replace('F', '1', regex=True).replace('M', '0', regex=True).astype(int)
    df['got_virus'] = df['got_virus'].replace('y', '1', regex=True).replace('n', '0', regex=True).astype(int)
    return df
