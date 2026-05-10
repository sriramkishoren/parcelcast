"""
Model wrappers: Baseline (4-week MA) / Prophet (with US retail holidays) / LightGBM.

Uniform-ish interface, but Prophet and LightGBM are different enough that
they have their own helper functions rather than a strict ABC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Baseline ─────────────────────────────────────────────────────────────────

class MovingAverageBaseline:
    """4-week moving average. The 'must beat this' floor."""

    def __init__(self, window: int = 4):
        self.window = window
        self._last_avg = None

    def fit(self, df: pd.DataFrame, target_col: str = "packages"):
        self._last_avg = df[target_col].tail(self.window).mean()
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        return np.full(n_periods, self._last_avg)


# ─── Prophet ──────────────────────────────────────────────────────────────────

US_RETAIL_HOLIDAYS = pd.DataFrame({
    "holiday": [
        "BlackFriday", "BlackFriday", "BlackFriday", "BlackFriday", "BlackFriday",
        "CyberMonday", "CyberMonday", "CyberMonday", "CyberMonday", "CyberMonday",
        "Christmas", "Christmas", "Christmas", "Christmas", "Christmas",
        "MemorialDay", "MemorialDay", "MemorialDay", "MemorialDay", "MemorialDay",
        "July4th", "July4th", "July4th", "July4th", "July4th",
        "LaborDay", "LaborDay", "LaborDay", "LaborDay", "LaborDay",
        "BTS_peak", "BTS_peak", "BTS_peak", "BTS_peak", "BTS_peak",
    ],
    "ds": pd.to_datetime([
        "2011-11-25", "2012-11-23", "2013-11-29", "2014-11-28", "2015-11-27",
        "2011-11-28", "2012-11-26", "2013-12-02", "2014-12-01", "2015-11-30",
        "2011-12-25", "2012-12-25", "2013-12-25", "2014-12-25", "2015-12-25",
        "2011-05-30", "2012-05-28", "2013-05-27", "2014-05-26", "2015-05-25",
        "2011-07-04", "2012-07-04", "2013-07-04", "2014-07-04", "2015-07-04",
        "2011-09-05", "2012-09-03", "2013-09-02", "2014-09-01", "2015-09-07",
        "2011-08-15", "2012-08-15", "2013-08-15", "2014-08-15", "2015-08-15",
    ]),
    "lower_window": -3,
    "upper_window": 3,
})


class ProphetModel:
    """Prophet wrapper with US retail holidays and yearly seasonality."""

    def __init__(self, seasonality_mode: str = "multiplicative"):
        from prophet import Prophet
        self._Prophet = Prophet
        self.seasonality_mode = seasonality_mode
        self.model = None
        self._freq = "W-SAT"  # Retailer's fiscal week starts Saturday

    def fit(self, df: pd.DataFrame, date_col: str = "week_start", target_col: str = "packages"):
        train = df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})
        train["ds"] = pd.to_datetime(train["ds"])
        self.model = self._Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=self.seasonality_mode,
            holidays=US_RETAIL_HOLIDAYS,
            interval_width=0.80,
        )
        self.model.fit(train)
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        future = self.model.make_future_dataframe(periods=n_periods, freq=self._freq)
        forecast = self.model.predict(future)
        return forecast["yhat"].values[-n_periods:]

    def predict_with_intervals(self, n_periods: int) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=n_periods, freq=self._freq)
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(n_periods)


# ─── LightGBM ─────────────────────────────────────────────────────────────────

class LightGBMForecaster:
    """Global LightGBM model — trains across all (region, channel) series."""

    def __init__(self, params: dict = None):
        import lightgbm as lgb
        self._lgb = lgb
        self.params = params or {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "verbose": -1,
        }
        self.model = None
        self.feature_cols_ = None

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "packages",
        n_estimators: int = 500,
    ):
        train = df.dropna(subset=feature_cols + [target_col])
        if len(train) == 0:
            raise ValueError("No rows left after dropna — check feature_cols / target_col / NaN handling")

        train_set = self._lgb.Dataset(train[feature_cols], label=train[target_col])
        self.model = self._lgb.train(
            self.params,
            train_set,
            num_boost_round=n_estimators,
        )
        self.feature_cols_ = feature_cols
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_cols_])

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        imp = self.model.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({"feature": self.feature_cols_, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
