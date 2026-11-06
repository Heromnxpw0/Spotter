"""Rolling-origin backtest. Every test window is two months, like Nov-Dec."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from spotterassessment.features import add_features, bad_rate_mask, daily_market, load
from spotterassessment.model import FreightRateModel

# name: (last training date, test start, test end)
FORWARD_FOLDS = {
    "Mar-Apr": ("2025-02-28", "2025-03-01", "2025-04-30"),
    "May-Jun": ("2025-04-30", "2025-05-01", "2025-06-30"),
    "Jul-Aug": ("2025-06-30", "2025-07-01", "2025-08-31"),
    "Sep-Oct": ("2025-08-31", "2025-09-01", "2025-10-31"),
}
TREE_FEATURES = ["log_distance", "equipment_code", "weight_clean", "day_of_week",
                 "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
TREE_PARAMS = dict(objective="l1", n_estimators=600, learning_rate=0.05, num_leaves=63,
                   min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   random_state=0, verbose=-1, deterministic=True, force_row_wise=True, n_jobs=8)


def make_splits(raw, market, low=0.6, high=1.6):
    """Yield (name, train, test). August also runs leave-one-month-out, because it is the
    only labeled month where quote_signal is noise, as it appears to be in Nov-Dec."""
    def build(past, test_rows):
        fill = past["weight"].abs().median()
        train = add_features(past[~bad_rate_mask(past, low, high)], market, fill)
        test = add_features(test_rows.assign(bad_rate=bad_rate_mask(test_rows)), market, fill)
        return train, test

    for name, (cutoff, start, end) in FORWARD_FOLDS.items():
        yield (name, *build(raw[raw["date"] <= cutoff], raw[raw["date"].between(start, end)]))
    august = raw["date"].dt.month == 8
    yield ("Leave-Aug", *build(raw[~august], raw[august]))


def score(test, pred):
    actual = test["posted_rate"].to_numpy()
    clean = ~test["bad_rate"].to_numpy()
    error = actual - pred
    return {
        "mape": (np.abs(error) / actual)[clean].mean() * 100,
        "mae": np.abs(error)[clean].mean(),
        "rmse": np.sqrt((error ** 2)[clean].mean()),
        "mae_all": np.abs(error).mean(),
        "rmse_all": np.sqrt((error ** 2).mean()),
        "bias": (-error / actual)[clean].mean() * 100,
    }


def baseline_predict(train, test):
    bands = np.unique(np.r_[0, train["distance"].quantile(np.linspace(0.1, 0.9, 9)), np.inf])
    rpm = train["posted_rate"] / train["distance"]
    table = rpm.groupby([train["equipment"], pd.cut(train["distance"], bands)], observed=True).median()
    keys = pd.MultiIndex.from_arrays([test["equipment"], pd.cut(test["distance"], bands)])
    return keys.map(table.to_dict()).to_numpy(dtype=float) * test["distance"].to_numpy()


def tree_predict(train, test, features=TREE_FEATURES):
    codes = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
    train = train.assign(equipment_code=train["equipment"].map(codes))
    test = test.assign(equipment_code=test["equipment"].map(codes))
    model = lgb.LGBMRegressor(**TREE_PARAMS).fit(train[features], np.log(train["posted_rate"]))
    return np.exp(model.predict(test[features]))


def ridge_predict(train, test):
    return FreightRateModel().fit(train).predict(test)


def evaluate(models, splits):
    rows = [{"model": name, "fold": fold, **score(test, fn(train, test))}
            for fold, train, test in splits for name, fn in models.items()]
    return pd.DataFrame(rows)


def summarize(results):
    forward = results[results["fold"] != "Leave-Aug"]
    summary = forward.groupby("model", sort=False)[["mape", "mae", "rmse", "mae_all", "rmse_all", "bias"]].mean()
    summary["worst_mape"] = forward.groupby("model")["mape"].max()
    folds = results.pivot(index="model", columns="fold", values="mape")
    return summary.join(folds[[*FORWARD_FOLDS, "Leave-Aug"]]).round(2)


def main():
    raw, future = load("train_test"), load("validation")
    splits = list(make_splits(raw, daily_market(raw, future)))
    models = {"median baseline": baseline_predict, "first LightGBM": tree_predict, "structural ridge": ridge_predict}
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(summarize(evaluate(models, splits)).to_string())


if __name__ == "__main__":
    main()
