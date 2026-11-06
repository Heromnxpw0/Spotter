"""Reproduces every rejected-idea comparison quoted in the report."""
import numpy as np
import pandas as pd

from spotterassessment.backtest import (TREE_FEATURES, evaluate, make_splits, ridge_predict, score,
                                        summarize, tree_predict)
from spotterassessment.features import add_features, bad_rate_mask, daily_market, load
from spotterassessment.model import FreightRateModel


class DollarRateModel(FreightRateModel):
    """The same model, fitted on dollars instead of log(rate)."""

    def fit(self, train):
        self.last_time = train["months_elapsed"].max()
        self.pipeline.fit(train, train["posted_rate"])
        return self

    def predict(self, df):
        frozen = df.assign(months_elapsed=df["months_elapsed"].clip(upper=self.last_time))
        return self.pipeline.predict(frozen)


def main():
    raw, future = load("train_test"), load("validation")
    market = daily_market(raw, future)
    splits = list(make_splits(raw, market))
    pd.set_option("display.width", 220, "display.max_columns", 20)

    print("== Log rate vs dollars ==")
    dollars = lambda tr, te: DollarRateModel().fit(tr).predict(te)
    print(summarize(evaluate({"log rate": ridge_predict, "dollars": dollars}, splits)).to_string())
    print("non-positive dollar predictions:", sum((dollars(tr, te) <= 0).sum() for _, tr, te in splits))

    print("== Trees and the trap features ==")
    tree_with = lambda extra: lambda tr, te: tree_predict(tr, te, TREE_FEATURES + extra)
    print(summarize(evaluate({
        "tree": tree_predict,
        "tree + quote_signal": tree_with(["quote_signal"]),
        "tree + market_index": lambda tr, te: tree_predict(tr.assign(mi=tr["market"]), te.assign(mi=te["market"]), TREE_FEATURES + ["mi"]),
    }, splits)).to_string())

    print("== Regularization strength ==")
    print(summarize(evaluate({f"alpha={a}": (lambda a: lambda tr, te: FreightRateModel(alpha=a).fit(tr).predict(te))(a)
                              for a in (0.001, 0.01, 0.1, 1.0, 10.0)}, splits)).to_string())

    print("== Time and market terms ==")
    def variant(no_market=False, no_time=False, extrapolate=False):
        def fit_predict(tr, te):
            tr2, te2 = tr.copy(), te.copy()
            if no_market:
                tr2["market"], te2["market"] = 0.0, 0.0
            if no_time:
                tr2["months_elapsed"], te2["months_elapsed"] = 0.0, 0.0
            model = FreightRateModel().fit(tr2)
            if extrapolate:
                model.last_time = np.inf
            return model.predict(te2)
        return fit_predict
    print(summarize(evaluate({
        "final: market + frozen trend": variant(),
        "no market, no time": variant(no_market=True, no_time=True),
        "market only": variant(no_time=True),
        "market + extrapolated trend": variant(extrapolate=True),
    }, splits)).to_string())

    print("== Corrupted-rate threshold ==")
    rows = []
    for low, high in [(0, 1e9), (0.4, 2.5), (0.5, 2.0), (0.6, 1.6), (0.7, 1.4), (0.8, 1.25)]:
        result = evaluate({f"keep {low}-{high}": ridge_predict}, list(make_splits(raw, market, low, high)))
        rows.append(result)
    print(summarize(pd.concat(rows)).to_string())

    print("== Random split vs time split (ridge and tree + quote_signal) ==")
    fill = raw["weight"].abs().median()
    full = add_features(raw.assign(bad_rate=bad_rate_mask(raw)), market, fill)
    in_train = np.random.RandomState(0).rand(len(full)) < 0.8
    tr, te = full[in_train & ~full["bad_rate"]], full[~in_train]
    for name, fn in {"tree": tree_predict, "tree + quote_signal": tree_with(["quote_signal"]),
                     "ridge": ridge_predict}.items():
        print(name, {k: round(v, 2) for k, v in score(te, fn(tr, te)).items()})

    print("== Held-out cities: 8 random cities, 10 repeats, test Sep-Oct ==")
    cities = sorted(set(raw["pickup"]) | set(raw["delivery"]))
    rng = np.random.RandomState(1)
    rows = []
    for repeat in range(10):
        held = set(rng.choice(cities, 8, replace=False))
        touches = full["pickup"].isin(held) | full["delivery"].isin(held)
        tr = full[(full["date"] <= "2025-08-31") & ~full["bad_rate"] & ~touches]
        te = full[(full["date"] > "2025-08-31") & touches]
        rows.append({"model": "ridge", **score(te, ridge_predict(tr, te))})
        rows.append({"model": "tree", **score(te, tree_predict(tr, te))})
    print(pd.DataFrame(rows).groupby("model")[["mape", "mae", "bias"]].agg(["mean", "std"]).round(2).to_string())

    print("== Extreme cities held out: Sep-Oct ==")
    latitude = full.groupby("pickup")["pickup_lat"].first().sort_values()
    longitude = full.groupby("pickup")["pickup_lon"].first().sort_values()
    groups = {"south": latitude.index[:3], "north": latitude.index[-2:], "west": longitude.index[:2], "east": longitude.index[-2:]}
    for name, members in groups.items():
        touches = full["pickup"].isin(members) | full["delivery"].isin(members)
        tr = full[(full["date"] <= "2025-08-31") & ~full["bad_rate"] & ~touches]
        te = full[(full["date"] > "2025-08-31") & touches]
        print(name, list(members), "ridge", round(score(te, ridge_predict(tr, te))["mape"], 2),
              "tree", round(score(te, tree_predict(tr, te))["mape"], 2))


if __name__ == "__main__":
    main()
