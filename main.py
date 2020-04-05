import pandas as pd
import utils as ut
from sklearn import linear_model


from sklearn.metrics import mean_squared_error
from comparisor import measure


def main():
    dir = "data/"
    original = "data_original.csv"
    anon = "data_anon.csv"
    anon2 = "data_anon_2.csv"


    print("Starting process...")

    print("Taking default configuration and data.")
    print("- file directory: " + dir)
    print("- full data file name: " + original)
    print("- anonymizied data file name: " + anon)
    print("- model: linear model")

    print("Start reading... " + dir + original)
    df_orig = pd.read_csv(dir + original).drop('name', axis=1)
    df_orig = ut.exampleCleaner(df_orig)
    print(df_orig.head(1))

    print("...and... " + dir + anon)
    df_anon = pd.read_csv(dir + anon)
    df_anon = ut.exampleCleaner(df_anon)
    print(df_anon.head(1))

    print("...and... " + dir + anon2)
    df_anon2 = pd.read_csv(dir + anon2)
    df_anon2 = ut.exampleCleaner(df_anon2)
    print(df_anon2.head(1))

    listed_data = [("0", df_orig), ("1", df_anon), ("2", df_anon2)]
    listed_measures = [mean_squared_error] #if the user wants to add manually their own measure

    print("Create model")
    model = linear_model.LogisticRegression()

    print("Compare model with both data type (anonimizied and not)")

    measure(listed_data, 0.8, model)

    print("... process ended.")


main()
