"""Loading, cleaning, and feature engineering. Nothing here looks at the target,
except bad_rate_mask, which is only ever used to filter training rows and to
separate clean rows when scoring."""
import numpy as np
import pandas as pd

EARTH_RADIUS_MILES = 3958.8
SMOOTH_FEATURES = ["log_distance", "haversine", "weight_clean",
                  "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


def load(name):
    return pd.read_csv(f"data/{name}.csv", parse_dates=["date"])


def daily_market(*frames):
    """Average market_index per date, used to fill gaps and for the December chart."""
    return pd.concat(frames).groupby("date")["market_index"].mean().rename("market_daily")


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return EARTH_RADIUS_MILES * 2 * np.arcsin(np.sqrt(a))


def add_features(df, market_by_date, weight_fill):
    out = df.join(market_by_date, on="date")
    load_market = out["market_index"] if "market_index" in out else out["market_daily"]
    out["market"] = load_market.fillna(out["market_daily"])
    out["weight_clean"] = out["weight"].abs().fillna(weight_fill)
    out["log_distance"] = np.log(out["distance"])
    out["haversine"] = haversine_miles(out["pickup_lat"], out["pickup_lon"],
                                       out["delivery_lat"], out["delivery_lon"])
    out["months_elapsed"] = (out["date"] - pd.Timestamp("2025-01-01")).dt.days / 30
    out["day_of_week"] = out["date"].dt.dayofweek
    return out


def bad_rate_mask(df, low=0.6, high=1.6):
    """True where rate-per-mile is far from similar loads (equipment x distance band)."""
    rate_per_mile = df["posted_rate"] / df["distance"]
    band = pd.qcut(df["distance"], 12, duplicates="drop")
    typical = rate_per_mile.groupby([df["equipment"], band], observed=True).transform("median")
    ratio = rate_per_mile / typical
    return (ratio < low) | (ratio > high)
