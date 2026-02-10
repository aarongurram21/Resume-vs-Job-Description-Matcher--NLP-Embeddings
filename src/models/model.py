from typing import Literal
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ModelName = Literal["logreg", "rf"]


def make_model(name: ModelName = "logreg"):
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    return LogisticRegression(max_iter=500)
