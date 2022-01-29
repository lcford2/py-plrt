import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

from timing_function import time_function


@time_function
def read_tva_data():
    data_path = "~/projects/predict-release/tva_data/tva_timeseries.csv"
    df = pd.read_csv(data_path, index_col=[0, 1])  # dtype=np.float64)
    dt_index = pd.to_datetime(df.index.get_level_values(0))
    new_index = pd.MultiIndex.from_tuples(zip(dt_index, df.index.get_level_values(1)))
    df.index = new_index
    df["release_acre-feet"] = df["release_cfs"] * 3600 * 24 / 43560
    df["net-inflow_acre-feet"] = df["net-inflow_cfs"] * 3600 * 24 / 43560
    df = df.drop(["release_cfs", "net-inflow_cfs"], axis=1)
    df = df[df.index.get_level_values(0) >= datetime.datetime(1991, 10, 1)]
    return df


@time_function
def setup_model_state(df):
    df = df.rename(
        columns={
            "release_acre-feet": "release",
            "storage_acre-feet": "storage",
            "net-inflow_acre-feet": "inflow",
        }
    )
    res_grouper = df.index.get_level_values(1)
    df["storage_pre"] = df.groupby(res_grouper)["storage"].shift(1)
    df["release_pre"] = df.groupby(res_grouper)["release"].shift(1)
    df = df.dropna()
    return df


# @time_function
def solve_regression(X, y):
    left = np.linalg.inv(X.T @ X)
    right = X.T @ y
    return left @ right


def predict_regression(X, p):
    return X @ p


def split_data_for_node(t, j, X, y):
    right = np.argwhere(X[:,j] > t)
    left = np.argwhere(X[:,j] <= t)
    X_right = X[right[:,0]]
    y_right = y[right[:,0]]
    X_left = X[left[:,0]]
    y_left = y[left[:,0]]
    return X_left, X_right, y_left, y_right


def get_node_score(t, j, X, y):
    X_left, X_right, y_left, y_right = split_data_for_node(t, j, X, y)
    N_left = y_left.shape[0]
    N_right = y_right.shape[0]
    if N_left == 0 or N_right == 0:
        return float(inf)
    p_left = solve_regression(X_left, y_left)
    p_right = solve_regression(X_right, y_right)
    yhat_left = predict_regression(X_left, p_left)
    yhat_right = predict_regression(X_right, p_right)
    mse_left = mean_squared_error(y_left, yhat_left)
    mse_right = mean_squared_error(y_right, yhat_right)
    left_score = N_left / y.shape[0] * mse_left
    right_score = N_right / y.shape[0] * mse_right
    return left_score + right_score


def optimize_node(X, y):
    n_features = X.shape[1]
    results = []
    for j in range(n_features):
        opt = minimize(get_node_score, [np.mean(X[:,j])], args=(j, X, y), 
                       method="Nelder-Mead")
        print(opt)
        results.append((opt.fun, opt.x))
    best = np.argmin([i[0] for i in results])
    best_val = results[best][1]
    return best, best_val


def main():
    df = read_tva_data()
    df = setup_model_state(df)
    X = df[["storage_pre", "release_pre", "inflow"]].values
    y = df["release"].values
    best, best_val = optimize_node(X, y) 
    from IPython import embed as II
    II()

if __name__ == "__main__":
    main()
