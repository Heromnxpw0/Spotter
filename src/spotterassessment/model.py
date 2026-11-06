"""log(rate) = smooth(distance, weight, geography) + equipment + b*market + c*time

The time term is frozen at the last training date. We assume the market term and
the level reached by the end of training carry forward, and do not extrapolate the trend.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from spotterassessment.features import SMOOTH_FEATURES


class FreightRateModel:
    def __init__(self, alpha=0.01, n_knots=8):
        columns = ColumnTransformer([
            ("smooth", SplineTransformer(n_knots=n_knots, degree=3, extrapolation="constant"), SMOOTH_FEATURES),
            ("equipment", OneHotEncoder(sparse_output=False), ["equipment"]),
            ("level", "passthrough", ["market", "months_elapsed"]),
        ], sparse_threshold=0)
        self.pipeline = make_pipeline(columns, Ridge(alpha=alpha))

    def fit(self, train):
        self.last_time = train["months_elapsed"].max()
        self.pipeline.fit(train, np.log(train["posted_rate"]))
        return self

    def predict(self, df):
        frozen = df.assign(months_elapsed=df["months_elapsed"].clip(upper=self.last_time))
        return np.exp(self.pipeline.predict(frozen))

    @property
    def market_effect(self):
        return self.pipeline[-1].coef_[-2]

    @property
    def monthly_trend(self):
        return self.pipeline[-1].coef_[-1]
