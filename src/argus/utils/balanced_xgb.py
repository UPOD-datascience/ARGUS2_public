"""XGBoost classifier with class weighting derived from the data it is fit on."""

import numpy as np
from xgboost import XGBClassifier


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier that sets scale_pos_weight to n_negative / n_positive of the labels passed to fit.

    Inside cross-validation, calibration or learning curves the weight is therefore computed per training fold,
    instead of once on the full training set. The class lives in the package so that pickled models can be loaded
    by the inference pipeline.
    """

    def fit(self, X, y, **kwargs):  # noqa: N803
        """Set scale_pos_weight from y and fit the model."""
        y = np.asarray(y)
        self.set_params(scale_pos_weight=(y == 0).sum() / (y == 1).sum())
        return super().fit(X, y, **kwargs)
