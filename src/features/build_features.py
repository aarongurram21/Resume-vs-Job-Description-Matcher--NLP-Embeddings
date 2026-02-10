from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC = ["feature1", "feature2"]


def build_preprocessor(numeric_features: List[str] = None) -> ColumnTransformer:
    numeric_features = numeric_features or NUMERIC
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features)], remainder="drop")
    return preprocessor


def separate_features_targets(df: pd.DataFrame, target_col: str = "target"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
