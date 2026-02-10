from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from src.data.load_data import load_raw, save_processed
from src.features.build_features import build_preprocessor, separate_features_targets
from src.models.model import make_model


def train(raw_name: str = "sample.csv", model_name: str = "logreg", model_dir: Path = Path("models")) -> dict:
    df = load_raw(raw_name)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    save_processed(train_df, val_df)

    X_train, y_train = separate_features_targets(train_df)
    X_val, y_val = separate_features_targets(val_df)

    preprocessor = build_preprocessor()
    model = make_model(model_name)
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    metrics = {"accuracy": float(accuracy_score(y_val, y_pred)), "f1": float(f1_score(y_val, y_pred))}

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = model_dir / f"model_{model_name}.joblib"
    joblib.dump(clf, artifact)
    return {"metrics": metrics, "artifact": artifact}


if __name__ == "__main__":
    result = train()
    print(result)
