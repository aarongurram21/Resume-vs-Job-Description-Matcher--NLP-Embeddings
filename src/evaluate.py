from pathlib import Path
import json
import joblib
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.data.load_data import load_raw, save_processed
from src.features.build_features import separate_features_targets


def evaluate(raw_name: str = "sample.csv", model_path: Path = Path("models/model_logreg.joblib"), report_dir: Path = Path("reports")) -> dict:
    df = load_raw(raw_name)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    save_processed(train_df, val_df)

    X_val, y_val = separate_features_targets(val_df)

    clf = joblib.load(model_path)
    y_pred = clf.predict(X_val)

    metrics = {"accuracy": float(accuracy_score(y_val, y_pred)), "f1": float(f1_score(y_val, y_pred))}
    report = classification_report(y_val, y_pred, output_dict=True)

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "classification_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return {"metrics": metrics, "report_path": report_path}


if __name__ == "__main__":
    result = evaluate()
    print(result)
