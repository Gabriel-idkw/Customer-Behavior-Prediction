"""
Custom feature transformer for the customer-segmentation pipeline.

Takes raw, per-customer RFM inputs (Recency in days, Frequency as an order
count, Monetary in dollars, AOV in dollars, Return_Rate_Pct 0-100) and
produces the log-transformed feature vector, in the exact column order the
StandardScaler was fit on in Notebook 01. This class is imported by
Notebook 05 (to build the exported pipeline) and by app/app.py (to unpickle
and reuse it), so it lives in one shared module rather than being redefined
in both places.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

FEATURE_ORDER = ['Recency', 'Frequency_log', 'Monetary_log', 'AOV', 'Return_Rate_Pct']


class RFMFeatureTransformer(BaseEstimator, TransformerMixin):
    """Log-transforms Frequency/Monetary and orders columns to match the
    fitted StandardScaler. Stateless (no fitting needed beyond sklearn's
    boilerplate), so it is safe to drop into a pre-fitted Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=['Recency', 'Frequency', 'Monetary', 'AOV', 'Return_Rate_Pct'])

        df['Frequency_log'] = np.log1p(df['Frequency'])
        df['Monetary_log'] = np.log1p(df['Monetary'])
        return df[FEATURE_ORDER]

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_ORDER)
