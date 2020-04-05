from utils import splitter
from sklearn.metrics import accuracy_score, precision_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def measure(listed_data_and_name, train_size, model, validators=None):
    if validators is None:
        validators = [accuracy_score, precision_score]
    results = pd.DataFrame()
    for (name, data) in listed_data_and_name:
        test_size = 1 - train_size

        # Let's split data for each case
        train, test = splitter(data, train_size, test_size)

        # Then fit model and predict
        model.fit(X=train.iloc[:, :-1], y=train.iloc[:, -1])
        predictions = model.predict(X=test.iloc[:, :-1])

        # Validate results
        for validator in validators:
            error = validator(test.iloc[:, -1], predictions)
            results = results.append(
                pd.DataFrame({"k-value": [name], "measure": [validator.__name__], "measure_value": [error]}))

    print("Final result:")
    print(results.iloc[:, [0, 2]])
    # Graphic results
    sns.lineplot(x="k-value", y="measure_value",
                 hue="measure", style="measure",
                 data=results)
    plt.show()
