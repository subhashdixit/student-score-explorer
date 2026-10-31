"""Export small, reproducible browser inference models from existing sources."""

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = [json.loads((ROOT / "project.json").read_text())]


def field(name, lo, hi, label=None, integer=False) -> dict:
    """Declare a numeric feature contract, units and serving bounds."""
    return {
        "name": name,
        "type": "number",
        "min": lo,
        "max": hi,
        "label": label or name.replace("_", " "),
        "integer": integer,
    }


def category(name, label=None) -> dict:
    """Declare a categorical source feature."""
    return {"name": name, "type": "category", "label": label or name.replace("_", " ")}


def source(_project) -> tuple:
    """Load the attributed source and select the documented target and features."""
    p = ROOT
    notice = (
        "Original repository dataset; upstream annotation/collection "
        "provenance is not independently established."
    )
    raw = p / "notebook/data/stud.csv"
    df = pd.read_csv(raw)
    y = df.math_score.astype(float)
    groups = None
    fields = [
        field("reading_score", 0, 100),
        field("writing_score", 0, 100),
        category("lunch"),
        category("test_preparation_course"),
    ]
    return (raw, df, fields, np.asarray(y, dtype=float), groups, notice)


def cleaned(df, fields) -> tuple:
    """Normalize source fields; record and impute out-of-contract numeric values."""
    frame = df[[f["name"] for f in fields]].copy()
    invalid = {}
    for f in fields:
        c = f["name"]
        if f["type"] == "number":
            values = pd.to_numeric(frame[c].astype(str).str.strip(), errors="coerce")
            bad = ~values.between(f["min"], f["max"]) & values.notna()
            if f["integer"]:
                bad |= values.notna() & (values % 1 != 0)
            invalid[c] = int(bad.sum())
            frame[c] = values.mask(bad)
        else:
            frame[c] = frame[c].map(
                lambda v: (
                    "__MISSING__"
                    if pd.isna(v) or str(v).strip() in ["", "NaN", "nan", "None"]
                    else str(v).strip()
                )
            )
    return (frame, invalid)


def fit_preprocess(frame, fields) -> list:
    """Fit numeric imputation/scaling and category vocabulary using training only."""
    fitted = []
    for f in fields:
        spec = dict(f)
        s = frame[f["name"]]
        if f["type"] == "number":
            median = float(s.median()) if s.notna().any() else 0.0
            a = s.fillna(median).to_numpy(float)
            spec.update(
                median=median,
                mean=float(a.mean()),
                scale=float(a.std()) or 1.0,
                default=int(round(median)) if f["integer"] else round(median, 4),
            )
        else:
            cats = sorted(set(s) | {"__MISSING__"})
            spec.update(categories=cats, default=str(s.mode().iloc[0]))
        fitted.append(spec)
    return fitted


def transform(frame, specs) -> np.ndarray:
    """Apply frozen preprocessing in a stable feature order."""
    values = []
    for f in specs:
        s = frame[f["name"]]
        if f["type"] == "number":
            values.append(
                ((s.fillna(f["median"]).to_numpy(float) - f["mean"]) / f["scale"])[
                    :, None
                ]
            )
        else:
            values.append(
                np.column_stack([(s == c).to_numpy(float) for c in f["categories"]])
            )
    return np.concatenate(values, axis=1)


def export(estimator, kind) -> dict:
    """Export linear weights or bounded tree arrays without executable pickle files."""
    if hasattr(estimator, "estimators_"):
        trees = []
        for member in estimator.estimators_:
            t = member.tree_
            values = t.value[:, 0, :]
            if kind == "classification":
                values = values / np.maximum(values.sum(axis=1, keepdims=True), 1e-30)
            trees.append(
                {
                    "left": t.children_left.tolist(),
                    "right": t.children_right.tolist(),
                    "feature": t.feature.tolist(),
                    "threshold": t.threshold.tolist(),
                    "value": values.tolist(),
                }
            )
        return {
            "type": "forest",
            "trees": trees,
            "classes": (
                estimator.classes_.tolist() if kind == "classification" else None
            ),
        }
    return {
        "type": "linear",
        "coefficients": np.asarray(estimator.coef_).reshape(-1).tolist(),
        "intercept": float(np.asarray(estimator.intercept_).reshape(-1)[0]),
        "logistic": kind == "classification",
    }


def metrics(y, score, kind, threshold=0.5) -> dict:
    """Compute held-out classification or regression measures from real predictions."""
    if kind == "regression":
        return {
            "rows": len(y),
            "mae": float(mean_absolute_error(y, score)),
            "rmse": float(np.sqrt(mean_squared_error(y, score))),
            "r2": float(r2_score(y, score)),
        }
    predicted = np.asarray(score) >= threshold
    return {
        "rows": len(y),
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y, predicted, output_dict=True, zero_division=0
        ),
    }


def train(project) -> None:
    """Select on grouped validation data, evaluate once and export parity fixtures."""
    p = ROOT
    (raw, df, fields, y, group, notice) = source(project)
    (frame, invalid) = cleaned(df, fields)
    valid = np.isfinite(y)
    frame = frame.loc[valid].reset_index(drop=True)
    y = y[valid]
    if group is None:
        group = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    else:
        group = group[valid]
    (train_idx, rest) = next(
        GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42).split(
            frame, groups=group
        )
    )
    (va, te) = next(
        GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=43).split(
            frame.iloc[rest], groups=group[rest]
        )
    )
    (val_idx, test_idx) = (rest[va], rest[te])
    specs = fit_preprocess(frame.iloc[train_idx], fields)
    x = transform(frame, specs)
    classification = project["kind"] == "classification"
    estimators = {
        "linear": (
            LogisticRegression(C=1, max_iter=2000, random_state=42)
            if classification
            else Ridge(alpha=10)
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=64,
                max_depth=7,
                min_samples_leaf=8,
                n_jobs=2,
                random_state=42,
                class_weight="balanced",
            )
            if classification
            else RandomForestRegressor(
                n_estimators=64,
                max_depth=9,
                min_samples_leaf=4,
                n_jobs=2,
                random_state=42,
            )
        ),
    }
    reports = {}
    fitted = {}
    validation = {}
    for name, est in estimators.items():
        est.fit(x[train_idx], y[train_idx])
        score = (
            est.predict_proba(x[val_idx])[:, 1]
            if classification
            else est.predict(x[val_idx])
        )
        reports[name] = metrics(y[val_idx], score, project["kind"])
        fitted[name] = est
        validation[name] = score
    selected = (
        max(reports, key=lambda n: reports[n]["average_precision"])
        if classification
        else min(reports, key=lambda n: reports[n]["mae"])
    )
    est = fitted[selected]
    threshold = 0.5
    if classification:
        candidates = np.linspace(0.01, 0.99, 99)
        if project["name"] == "aps-failure-classifier":
            threshold = float(
                min(
                    candidates,
                    key=lambda t: 10
                    * np.sum((validation[selected] >= t) & (y[val_idx] == 0))
                    + 500 * np.sum((validation[selected] < t) & (y[val_idx] == 1)),
                )
            )
        else:
            threshold = float(
                max(
                    candidates,
                    key=lambda t: f1_score(
                        y[val_idx], validation[selected] >= t, zero_division=0
                    ),
                )
            )
    threshold = round(threshold, 2)
    testscore = (
        est.predict_proba(x[test_idx])[:, 1]
        if classification
        else est.predict(x[test_idx])
    )
    baseline = (
        DummyClassifier(strategy="prior")
        if classification
        else DummyRegressor(strategy="mean")
    )
    baseline.fit(x[train_idx], y[train_idx])
    basescore = (
        baseline.predict_proba(x[test_idx])[:, 1]
        if classification
        else baseline.predict(x[test_idx])
    )
    report = {
        "version": "1.0.0",
        "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "source_path": str(raw.relative_to(p)),
        "source_notice": notice,
        "original_rows": len(df),
        "valid_target_rows": len(y),
        "out_of_range_values_imputed": {k: v for (k, v) in invalid.items() if v},
        "selected_model": selected,
        "kind": project["kind"],
        "threshold": threshold,
        "split": {
            k: len(i)
            for (k, i) in [
                ("train", train_idx),
                ("validation", val_idx),
                ("test", test_idx),
            ]
        },
        "split_method": (
            "GroupShuffleSplit; distinct courier IDs"
            if project["name"] == "food-delivery-time-predictor"
            else "GroupShuffleSplit; identical selected profiles share a group"
        ),
        "validation_candidates": reports,
        "test": metrics(y[test_idx], testscore, project["kind"], threshold),
        "test_baseline": metrics(y[test_idx], basescore, project["kind"]),
        "training_only_preprocessing": True,
        "sklearn_version": __import__("sklearn").__version__,
    }
    if project["name"] == "aps-failure-classifier":
        (_tn, fp, fn, _tp) = np.array(report["test"]["confusion_matrix"]).ravel()
        report["test"]["challenge_cost"] = int(10 * fp + 500 * fn)
    model = {
        "version": 1,
        "kind": project["kind"],
        "features": specs,
        "estimator": export(est, project["kind"]),
        "threshold": threshold,
        "unit": project["unit"],
    }
    (p / "site").mkdir(exist_ok=True)
    (p / "reports").mkdir(exist_ok=True)
    payload = json.dumps(model, separators=(",", ":"), allow_nan=False) + "\n"
    (p / "site/model.json").write_text(payload)
    report["artifact_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    sample = {f["name"]: f["default"] for f in specs}
    (p / "site/sample.json").write_text(json.dumps(sample, indent=2) + "\n")
    fixtures = []
    for idx in test_idx[:30]:
        row = {
            f["name"]: (
                None
                if pd.isna(frame.iloc[idx][f["name"]])
                else (
                    frame.iloc[idx][f["name"]].item()
                    if hasattr(frame.iloc[idx][f["name"]], "item")
                    else frame.iloc[idx][f["name"]]
                )
            )
            for f in specs
        }
        fixtures.append(
            {
                "record": row,
                "expected": float(
                    est.predict_proba(x[[idx]])[0, 1]
                    if classification
                    else est.predict(x[[idx]])[0]
                ),
            }
        )
    (p / "reports/parity-fixtures.json").write_text(
        json.dumps(fixtures, allow_nan=False) + "\n"
    )
    (p / "site/metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (p / "reports/metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(project["name"], selected, report["split"], report["test"], flush=True)


if __name__ == "__main__":
    train(CONFIG[0])
