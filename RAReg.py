# Nate Glod
# RAReg.py
# Simple script for running regressions and doing basic analysis on csv files

from sklearn import linear_model
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import argparse as ap
import os

"""
Basic polynomial regression data
"""
def poly_regression(x, y, degree):

  # Reshapes the models to be able to run regression on them
  x = x.reshape(-1, 1)
  y = y.reshape(-1, 1)

  # Polynomial regression with nth degree, gives back rmse and r2
  polynomial_features = PolynomialFeatures(degree=degree)
  x_poly = polynomial_features.fit_transform(x)

  model = linear_model.LinearRegression()
  model.fit(x_poly, y)
  y_poly_pred = model.predict(x_poly)

  rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
  r2 = r2_score(y, y_poly_pred)
  return rmse, r2


"""
Should be run from terminal in the following manner:

python3 RAReg.py [csv] <other flags>

Flags:

csv: The name of the CSV file to be read in. Only non-optional argument
c<csv>

start: The column at which the regression starts. Must be the column number. Col 0 by default
-start <start>

end: The column at which the regression ends. Must be the column number. column 0 by default
-end <end>

min: The minimum number a regression's P-score must get in order to output its graph. 0 by default
-min <min>

max: The maximum number a regression's P-score must get in order to output its graph. 1 by default
-max <max>

degree: The maximum degree polynomial which the regression will be run. Must be at least 1, is 1 by default, and runs all
polynomial degree regressions up to that
-deg <degree>

out: The folder to which all regression graphs are outputted. Must be a relative file path. Outputs to current folder
by default
-out <out>

"""


def run_all_polynomials(args=None):
    # Reads in the neccessary csv file
    df = pd.read_csv(args.csv[0])
    regr = linear_model.LinearRegression()

    newpath = './'
    if (not os.path.exists('./' + args.out[0])) and (args.out[0] != ""):
        newpath = newpath + args.out[0] + '/'
        os.makedirs('./' + args.out[0])

    print("xVal, yVal, degree, r2, rmse")
    for i in range(args.start[0], args.end[0]):
        for j in range(1, args.end[0] - i):
            mat = df[[df.columns[i], df.columns[i + j]]].values
            for d in range(1, args.deg[0] + 1):
                rmse, r2 = poly_regression(mat[:, 0], mat[:, 1], d)
                plt.figure(figsize=(9, 9))
                plt.xlabel(df.columns[i])
                plt.ylabel(df.columns[i + j])
                plt.title('r2: ' + str(r2) + 'degree: ' + str(d))
                plt.scatter(mat[:, 0], mat[:, 1])

                # Eliminates all of the graphs with correlations below 0.1
                if (r2 > args.min[0]) and (r2 < args.max[0]):
                    plt.savefig(newpath + df.columns[i] + '_vs_' + df.columns[i + j] + '_' + str(d) + '_degree.png')

                print(df.columns[i] + ', ' + df.columns[i + j] + ', ' + str(d) + ', ' + str(r2) + ', ' + str(rmse))
                plt.close()


def main():
    parser = ap.ArgumentParser()
    parser.add_argument("csv", nargs=1, help="The name of the CSV file to be read in", type=str)
    parser.add_argument("-start", nargs=1,
                        help="The column at which the regression starts. Must be the column number. Col 0 by default",
                        type=int, default=0)
    parser.add_argument("-end", nargs=1, help="The column at which the regression ends. Must be the column number."
                                              " Column zero by default,", type=int, default=[0])
    parser.add_argument("-min", nargs=1, help="The minimum number a regression's P-score must get in order to output "
                                              "its graph. 0 by default", type=float, default=[0])
    parser.add_argument("-max", nargs=1, help="The maximum number a regression's P-score must get in order to output "
                                              "its graph. 1 by default", type=float, default=[0])
    parser.add_argument("-deg", nargs=1, help="The maximum degree polynomial which the regression will be run. Must be "
                                              "at least 1, is 1 by default, and runs all polynomial degree regressions "
                                              "up to that", type=int, default=[1])
    parser.add_argument("-out", nargs=1, help="The folder to which all regression graphs are outputted. Must be a "
                                              "relative file path. Outputs to current folder by default", type=str, default=[""])
    run_all_polynomials(args=parser.parse_args())


if __name__ == "__main__":
    main()
