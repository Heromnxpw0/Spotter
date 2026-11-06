"""Train on all labeled data and write both submission files."""
import pandas as pd

from spotterassessment.features import add_features, bad_rate_mask, daily_market, load
from spotterassessment.model import FreightRateModel


def main():
    train = load("train_test")
    validation = load("validation")
    december = load("december_chart_inputs")
    template = pd.read_csv("data/validation_predictions_template.csv")

    market = daily_market(train, validation)
    fill = train["weight"].abs().median()
    model = FreightRateModel().fit(add_features(train[~bad_rate_mask(train)], market, fill))
    print(f"market effect {model.market_effect:.4f}, monthly trend {model.monthly_trend:.4f}")

    predictions = model.predict(add_features(validation, market, fill)).round(2)
    template["predicted_rate"] = template["load_id"].map(dict(zip(validation["load_id"], predictions)))
    template.to_csv("validation_predictions.csv", index=False)

    # The December file has no coordinates, so look them up from training rows.
    pickup = train.loc[train["pickup"] == "Lexington", ["pickup_lat", "pickup_lon"]].iloc[0]
    delivery = train.loc[train["delivery"] == "Fort Wayne", ["delivery_lat", "delivery_lon"]].iloc[0]
    rows = add_features(december.assign(**pickup, **delivery), market, fill)
    december["predicted_rate"] = model.predict(rows).round(2)
    december.to_csv("data/december_chart_inputs.csv", index=False)


if __name__ == "__main__":
    main()
