import logging
import warnings

from prophet import Prophet

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


class ProphetForecaster:
    def __init__(
        self,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        horizon=24,
        freq="H",
    ):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.horizon = horizon
        self.freq = freq
        self.model_ = None
        self._last_ds = None

    def fit(self, df, datetime_col, target_col):
        prophet_df = df[[datetime_col, target_col]].rename(columns={datetime_col: "ds", target_col: "y"})
        self.model_ = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
        )
        self.model_.fit(prophet_df)
        self._last_ds = prophet_df["ds"].max()
        return self

    def predict(self, steps=None):
        n = steps or self.horizon
        future = self.model_.make_future_dataframe(periods=n, freq=self.freq)
        forecast = self.model_.predict(future)
        return forecast["yhat"].values[-n:]

    def predict_on_df(self, df, datetime_col):
        future = df[[datetime_col]].rename(columns={datetime_col: "ds"})
        forecast = self.model_.predict(future)
        return forecast["yhat"].values

    def get_params(self):
        return {
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "seasonality_mode": self.seasonality_mode,
        }
