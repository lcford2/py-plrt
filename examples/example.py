import datetime

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from IPython import embed as II

from tree_combo_lr import TreeComboLR



def read_tva_data():
    data_path = "~/projects/predict-release/tva_data/tva_timeseries.csv"
    df = pd.read_csv(data_path, index_col=[0, 1])
    dt_index = pd.to_datetime(df.index.get_level_values(0))
    new_index = pd.MultiIndex.from_tuples(zip(dt_index, df.index.get_level_values(1)))
    df.index = new_index
    df["release_acre-feet"] = df["release_cfs"] * 3600 * 24 / 43560
    df["net-inflow_acre-feet"] = df["net-inflow_cfs"] * 3600 * 24 / 43560
    df = df.drop(["release_cfs", "net-inflow_cfs"], axis=1)
    df = df[df.index.get_level_values(0) >= datetime.datetime(1991, 10, 1)]
    return df



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



def scale_model_inputs(df, means=None, stds=None):
    res_groups = df.index.get_level_values(1)
    means = df.groupby(res_groups).mean()
    stds = df.groupby(res_groups).std()
    scaled = df.groupby(res_groups).apply(
        lambda x: (x - means.loc[x.index[0][1]]) / stds.loc[x.index[0][1]]
    )
    return scaled, means, stds



def unscale_model_inputs(df, means, stds):
    res_groups = df.index.get_level_values(1)
    unscaled = df.groupby(res_groups).apply(
        lambda x: x * stds.loc[x.index[0][1]] + means.loc[x.index[0][1]]
    )
    return unscaled



def get_res_scores(df, metrics=None):
    metrics = ["nse"] if metrics is None else metrics

    metric_funcs = {
        "nse": r2_score,
        "rmse": lambda y, yhat: mean_squared_error(y, yhat, squared=False)
    }

    out = pd.DataFrame(
        index=df.index.get_level_values(1).unique(),
        columns=metrics
    )

    for m in metrics:
        out[m] = df.groupby(df.index.get_level_values(1)).apply(
            lambda x: metric_funcs[m](x["actual"], x["model"])
        )

    return out


def split_train_test_date(df, date):
    train = df[df.index.get_level_values(0) < date]
    test = df[df.index.get_level_values(0) >= date]
    return train, test


def main():
    # load data
    df = read_tva_data()
    df = setup_model_state(df)
    train, test = split_train_test_date(df, datetime.datetime(2010, 1, 1))
    scaled_train, means, stds = scale_model_inputs(train)
    scaled_test, means, stds = scale_model_inputs(test, means, stds)

    # setup model input
    features = ["storage_pre", "release_pre", "inflow"]
    response = "release"
    X = scaled_train[features].values
    y = scaled_train[response].values

    # define and train model
    model = TreeComboLR(X, y, feature_names=features, response_name=response)
    model.grow_tree()

    # get predictions
    yhat_train = model.predict()
    train_results = pd.DataFrame(
        {"actual": train[response], "model": yhat_train},
        index=train.index
    )

    yhat_test = model.predict(scaled_test[features].values)
    test_results = pd.DataFrame(
        {"actual": test[response], "model": yhat_test},
        index=test.index
    )

    # bring results back to original space
    train_results["model"] = unscale_model_inputs(
        train_results["model"],
        means["release"],
        stds["release"]
    )

    test_results["model"] = unscale_model_inputs(
        test_results["model"],
        means["release"],
        stds["release"]
    )

    # get scores
    train_scores = get_res_scores(
        train_results,
        metrics=["nse", "rmse"]
    )
    test_scores = get_res_scores(
        test_results,
        metrics=["nse", "rmse"]
    )
    II()


if __name__ == "__main__":
    main()
