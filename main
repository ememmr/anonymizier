import pandas as pd
import utils as ut
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from comparisor import measure


def main():
    dir = "data/"
    original = "data_original.csv"
    anon = "data_anon.csv"
    anon2 = "data_anon_2.csv"
    n_neighbors = 2

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

    listed_data = [df_orig, df_anon, df_anon2]

    print("Create model")
    model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights='uniform')

    print("Compare model with both data type (anonimizied and not)")
    for d in listed_data:
        measure(d, 0.8, model, mean_squared_error)

    print("... process ended.")


main()
