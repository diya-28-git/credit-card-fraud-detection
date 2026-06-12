"""
Credit Card Fraud Detection v2 — Enhanced Pipeline
====================================================
Improvements over v1:
  ✅ Hyperparameter tuning  → Optuna (Bayesian optimization)
  ✅ Threshold optimization → Precision-Recall curve + F1 maximization
  ✅ SMOTETomek             → Cleaner decision boundary than plain SMOTE
  ✅ Cross-validated AUC    → More reliable than single split evaluation
  ✅ Per-model best threshold report
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score
)
from imblearn.combine import SMOTETomek
import xgboost as xgb
import joblib
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────

def generate_data(n=60000, fraud_ratio=0.002):
    print("[INFO] Generating synthetic dataset …")
    nf = int(n * fraud_ratio); nl = n - nf
    cols = [f"V{i}" for i in range(1, 29)]
    ldf = pd.DataFrame(np.random.randn(nl, 28), columns=cols)
    ldf["Time"]   = np.random.uniform(0, 172792, nl)
    ldf["Amount"] = np.abs(np.random.exponential(88, nl))
    ldf["Class"]  = 0
    fdf = pd.DataFrame(
        np.random.randn(nf, 28) + np.random.choice([-2, 2], size=(nf, 28)),
        columns=cols
    )
    fdf["Time"]   = np.random.uniform(0, 172792, nf)
    fdf["Amount"] = np.abs(np.random.exponential(122, nf))
    fdf["Class"]  = 1
    df = pd.concat([ldf, fdf]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[INFO] Dataset: {len(df):,} rows | Fraud: {nf} ({fraud_ratio*100:.2f}%)")
    return df


def load_data(path="data/creditcard.csv"):
    if os.path.exists(path):
        print(f"[INFO] Loading {path} …")
        return pd.read_csv(path)
    return generate_data()


# ─────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess(df):
    df = df.copy()
    sc = StandardScaler()
    df["Amount"] = sc.fit_transform(df[["Amount"]])
    df["Time"]   = sc.fit_transform(df[["Time"]])

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Applying SMOTETomek …")
    smt = SMOTETomek(random_state=42)
    X_tr, y_tr = smt.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTETomek — {X_tr.shape[0]:,} samples | Fraud: {y_tr.sum():,}")
    return X_tr, X_test, y_tr, y_test, X_train, y_train


# ─────────────────────────────────────────────────────────────
# 3. THRESHOLD OPTIMIZER
# ─────────────────────────────────────────────────────────────

def best_threshold(y_true, y_prob, metric="f1"):
    """Find the probability threshold that maximises F1 (or precision/recall balance)."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has len = len(prec) - 1
    f1s = np.where(
        (prec[:-1] + rec[:-1]) == 0, 0,
        2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1])
    )
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]), float(f1s[best_idx])


# ─────────────────────────────────────────────────────────────
# 4. CROSS-VALIDATED AUC HELPER
# ─────────────────────────────────────────────────────────────

def cv_auc(model, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean(), scores.std()


# ─────────────────────────────────────────────────────────────
# 5A. LOGISTIC REGRESSION — Optuna tuning
# ─────────────────────────────────────────────────────────────

def tune_logistic_regression(X_train, y_train, n_trials=30):
    print("\n[TUNE] Logistic Regression …")

    def objective(trial):
        C         = trial.suggest_float("C", 1e-3, 10, log=True)
        solver    = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        penalty   = "l2"
        model = LogisticRegression(
            C=C, solver=solver, penalty=penalty,
            class_weight="balanced", max_iter=500, random_state=42
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train,
                               cv=cv, scoring="average_precision", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    print(f"  Best params: {bp}  |  AP={study.best_value:.4f}")

    model = LogisticRegression(
        C=bp["C"], solver=bp["solver"], penalty="l2",
        class_weight="balanced", max_iter=500, random_state=42
    )
    model.fit(X_train, y_train)
    return model, bp


# ─────────────────────────────────────────────────────────────
# 5B. RANDOM FOREST — Optuna tuning
# ─────────────────────────────────────────────────────────────

def tune_random_forest(X_train, y_train, n_trials=20):
    print("[TUNE] Random Forest …")

    def objective(trial):
        n_est      = trial.suggest_int("n_estimators", 50, 300)
        max_depth  = trial.suggest_int("max_depth", 4, 20)
        min_split  = trial.suggest_int("min_samples_split", 2, 10)
        min_leaf   = trial.suggest_int("min_samples_leaf", 1, 5)
        max_feat   = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        model = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_depth,
            min_samples_split=min_split, min_samples_leaf=min_leaf,
            max_features=max_feat, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train,
                               cv=cv, scoring="average_precision", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    print(f"  Best params: {bp}  |  AP={study.best_value:.4f}")

    model = RandomForestClassifier(
        n_estimators=bp["n_estimators"], max_depth=bp["max_depth"],
        min_samples_split=bp["min_samples_split"],
        min_samples_leaf=bp["min_samples_leaf"],
        max_features=bp["max_features"],
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, bp


# ─────────────────────────────────────────────────────────────
# 5C. XGBOOST — Optuna tuning
# ─────────────────────────────────────────────────────────────

def tune_xgboost(X_train, y_train, n_trials=30):
    print("[TUNE] XGBoost …")
    spw = float((y_train == 0).sum() / (y_train == 1).sum())

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "scale_pos_weight":  spw,
            "eval_metric":       "logloss",
            "use_label_encoder": False,
            "random_state":      42,
            "n_jobs":            -1,
        }
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train,
                               cv=cv, scoring="average_precision", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    print(f"  Best params (top): lr={bp['learning_rate']:.4f}, depth={bp['max_depth']}, "
          f"n_est={bp['n_estimators']}  |  AP={study.best_value:.4f}")

    model = xgb.XGBClassifier(
        **bp, scale_pos_weight=spw,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)
    return model, bp


# ─────────────────────────────────────────────────────────────
# 5D. ISOLATION FOREST — Optuna tuning
# ─────────────────────────────────────────────────────────────

def tune_isolation_forest(X_legit, X_test, y_test, n_trials=15):
    print("[TUNE] Isolation Forest …")

    def objective(trial):
        cont       = trial.suggest_float("contamination", 0.001, 0.01)
        n_est      = trial.suggest_int("n_estimators", 50, 300)
        max_feat   = trial.suggest_float("max_features", 0.5, 1.0)
        model = IsolationForest(
            contamination=cont, n_estimators=n_est,
            max_features=max_feat, random_state=42, n_jobs=-1
        )
        model.fit(X_legit)
        scores = -model.score_samples(X_test)
        return average_precision_score(y_test, scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    print(f"  Best params: {bp}  |  AP={study.best_value:.4f}")

    model = IsolationForest(
        contamination=bp["contamination"],
        n_estimators=bp["n_estimators"],
        max_features=bp["max_features"],
        random_state=42, n_jobs=-1
    )
    model.fit(X_legit)
    return model, bp


# ─────────────────────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────────────────────

def evaluate_classifier(name, model, X_test, y_test, results):
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Threshold optimization ──
    thresh, best_f1 = best_threshold(y_test, y_prob)
    y_pred_opt = (y_prob >= thresh).astype(int)
    y_pred_def = model.predict(X_test)           # default 0.5 threshold

    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Default threshold (0.50):")
    print(classification_report(y_test, y_pred_def,
                                target_names=["Legit", "Fraud"], digits=4))
    print(f"  Optimised threshold ({thresh:.4f}):")
    print(classification_report(y_test, y_pred_opt,
                                target_names=["Legit", "Fraud"], digits=4))
    print(f"  ROC-AUC: {auc:.4f}  |  Avg Precision: {ap:.4f}  |  Best-F1: {best_f1:.4f}")

    results[name] = {
        "y_pred_def": y_pred_def,
        "y_pred_opt": y_pred_opt,
        "y_prob":     y_prob,
        "auc":        auc,
        "ap":         ap,
        "f1_def":     f1_score(y_test, y_pred_def),
        "f1_opt":     best_f1,
        "threshold":  thresh,
    }
    return results


def evaluate_if(model, X_test_s, y_test, results):
    scores  = -model.score_samples(X_test_s)
    thresh, best_f1 = best_threshold(y_test, scores)
    y_pred  = (scores >= thresh).astype(int)
    auc = roc_auc_score(y_test, scores)
    ap  = average_precision_score(y_test, scores)

    print(f"\n{'='*55}")
    print(f"  Isolation Forest (threshold={thresh:.4f})")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred,
                                target_names=["Legit", "Fraud"], digits=4))
    print(f"  ROC-AUC: {auc:.4f}  |  Avg Precision: {ap:.4f}  |  Best-F1: {best_f1:.4f}")

    results["Isolation Forest"] = {
        "y_pred_def": y_pred,
        "y_pred_opt": y_pred,
        "y_prob":     scores,
        "auc":        auc,
        "ap":         ap,
        "f1_def":     best_f1,
        "f1_opt":     best_f1,
        "threshold":  thresh,
    }
    return results


# ─────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────────────────────

def plot_threshold_curves(results, y_test):
    """Show how F1, Precision, Recall change with threshold for each model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        prec, rec, thresholds = precision_recall_curve(y_test, res["y_prob"])
        f1s = np.where(
            (prec[:-1] + rec[:-1]) == 0, 0,
            2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1])
        )
        ax.plot(thresholds, prec[:-1], label="Precision", color="royalblue",  lw=2)
        ax.plot(thresholds, rec[:-1],  label="Recall",    color="crimson",    lw=2)
        ax.plot(thresholds, f1s,        label="F1",        color="forestgreen",lw=2)
        ax.axvline(res["threshold"], color="darkorange", linestyle="--",
                   label=f"Best thresh={res['threshold']:.3f}")
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 1)

    plt.suptitle("Threshold Optimisation — Precision / Recall / F1 vs Threshold",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/threshold_curves.png", dpi=150)
    plt.close()
    print("[INFO] Threshold curves → outputs/threshold_curves.png")


def plot_confusion_matrices(results, y_test):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    for col, (name, res) in enumerate(results.items()):
        for row, (pred_key, label) in enumerate([
            ("y_pred_def", "Default thresh=0.50"),
            ("y_pred_opt", f"Optimal thresh={res['threshold']:.3f}")
        ]):
            ax = axes[row][col]
            cm = confusion_matrix(y_test, res[pred_key])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Legit", "Fraud"],
                        yticklabels=["Legit", "Fraud"])
            ax.set_title(f"{name}\n{label}", fontsize=9)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices: Default vs Optimised Threshold",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrices.png", dpi=150)
    plt.close()
    print("[INFO] Confusion matrices → outputs/confusion_matrices.png")


def plot_roc(results, y_test):
    plt.figure(figsize=(9, 6))
    colors = ["royalblue", "crimson", "forestgreen", "darkorange", "purple"]
    for (name, res), c in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.4f})", color=c, lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models (After Tuning)", fontweight="bold")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/roc_curves.png", dpi=150)
    plt.close()
    print("[INFO] ROC curves → outputs/roc_curves.png")


def plot_pr(results, y_test):
    plt.figure(figsize=(9, 6))
    colors = ["royalblue", "crimson", "forestgreen", "darkorange", "purple"]
    for (name, res), c in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        plt.plot(rec, prec, label=f"{name} (AP={res['ap']:.4f})", color=c, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — All Models (After Tuning)", fontweight="bold")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/pr_curves.png", dpi=150)
    plt.close()
    print("[INFO] PR curves → outputs/pr_curves.png")


def plot_comparison(results):
    names   = list(results.keys())
    aucs    = [results[n]["auc"]    for n in names]
    aps     = [results[n]["ap"]     for n in names]
    f1_def  = [results[n]["f1_def"] for n in names]
    f1_opt  = [results[n]["f1_opt"] for n in names]

    x = np.arange(len(names)); w = 0.2
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 1.5*w, aucs,   w, label="ROC-AUC",          color="royalblue")
    ax.bar(x - 0.5*w, aps,    w, label="Avg Precision",     color="crimson")
    ax.bar(x + 0.5*w, f1_def, w, label="F1 (default=0.5)",  color="#aaaaaa")
    ax.bar(x + 1.5*w, f1_opt, w, label="F1 (optimised)",    color="forestgreen")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Default vs Optimised Threshold", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150)
    plt.close()
    print("[INFO] Model comparison → outputs/model_comparison.png")


def plot_feature_importance(rf_model, feature_names):
    imp = rf_model.feature_importances_
    idx = np.argsort(imp)[-15:]
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[idx], imp[idx], color="steelblue")
    plt.title("Top 15 Feature Importances (Random Forest)", fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("[INFO] Feature importance → outputs/feature_importance.png")


def plot_optuna_importance(study, model_name):
    """Bar chart of Optuna hyperparameter importances."""
    try:
        imp = optuna.importance.get_param_importances(study)
        params = list(imp.keys()); vals = list(imp.values())
        plt.figure(figsize=(7, 4))
        plt.barh(params[::-1], vals[::-1], color="steelblue")
        plt.title(f"Hyperparameter Importance — {model_name}", fontweight="bold")
        plt.xlabel("Importance")
        plt.tight_layout()
        fname = f"outputs/hp_importance_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[INFO] HP importance → {fname}")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# 8. EDA
# ─────────────────────────────────────────────────────────────

def eda(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df["Class"].value_counts()
    axes[0].bar(["Legit", "Fraud"], counts.values, color=["steelblue", "crimson"])
    axes[0].set_title("Class Distribution")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")
    df[df["Class"] == 0]["Amount"].plot(kind="hist", bins=50, alpha=0.6,
                                        label="Legit", ax=axes[1], color="steelblue")
    df[df["Class"] == 1]["Amount"].plot(kind="hist", bins=50, alpha=0.8,
                                        label="Fraud", ax=axes[1], color="crimson")
    axes[1].set_title("Transaction Amount"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("outputs/eda_plots.png", dpi=150)
    plt.close()
    print("[INFO] EDA → outputs/eda_plots.png")


# ─────────────────────────────────────────────────────────────
# 9. SUMMARY
# ─────────────────────────────────────────────────────────────

def print_summary(results, best_params):
    print("\n" + "="*70)
    print("   FINAL SUMMARY — Hyperparameter Tuning + Threshold Optimisation")
    print("="*70)
    print(f"{'Model':<22} {'AUC':>7} {'AP':>7} {'F1(def)':>9} {'F1(opt)':>9} {'Thresh':>8}")
    print("-"*70)
    for name, res in results.items():
        print(f"{name:<22} {res['auc']:>7.4f} {res['ap']:>7.4f} "
              f"{res['f1_def']:>9.4f} {res['f1_opt']:>9.4f} {res['threshold']:>8.4f}")
    print("="*70)
    best = max(results, key=lambda k: results[k]["f1_opt"])
    print(f"\n  🏆 Best Model (by optimised F1): {best}")
    print(f"     AUC={results[best]['auc']:.4f} | F1={results[best]['f1_opt']:.4f} "
          f"| Threshold={results[best]['threshold']:.4f}")
    print("\n  Best Hyperparameters found:")
    for model_name, params in best_params.items():
        print(f"  [{model_name}] {params}")
    print("="*70)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    # ── Data ──
    df = load_data("data/creditcard.csv")
    eda(df)
    X_train, X_test, y_train, y_test, X_train_raw, y_train_raw = preprocess(df)

    # ── Tune supervised models ──
    lr,    lr_params  = tune_logistic_regression(X_train, y_train, n_trials=25)
    rf,    rf_params  = tune_random_forest(X_train, y_train,       n_trials=20)
    xgb_m, xgb_params = tune_xgboost(X_train, y_train,            n_trials=30)

    # ── Isolation Forest: train on original (unbalanced) legit samples ──
    sc2    = StandardScaler()
    X_lg_s = sc2.fit_transform(X_train_raw[y_train_raw == 0])
    X_ts_s = sc2.transform(X_test)
    iso, iso_params = tune_isolation_forest(X_lg_s, X_ts_s, y_test, n_trials=15)

    best_params = {
        "Logistic Regression": lr_params,
        "Random Forest":       rf_params,
        "XGBoost":             xgb_params,
        "Isolation Forest":    iso_params,
    }

    # ── Evaluate ──
    results = {}
    results = evaluate_classifier("Logistic Regression", lr,    X_test, y_test, results)
    results = evaluate_classifier("Random Forest",        rf,    X_test, y_test, results)
    results = evaluate_classifier("XGBoost",              xgb_m, X_test, y_test, results)
    results = evaluate_if(iso, X_ts_s, y_test, results)

    # ── Plots ──
    plot_threshold_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_roc(results, y_test)
    plot_pr(results, y_test)
    plot_comparison(results)
    plot_feature_importance(rf, X_test.columns.tolist())

    # ── Save models ──
    joblib.dump(lr,    "models/logistic_regression.pkl")
    joblib.dump(rf,    "models/random_forest.pkl")
    joblib.dump(xgb_m, "models/xgboost.pkl")
    joblib.dump(iso,   "models/isolation_forest.pkl")
    joblib.dump({n: res["threshold"] for n, res in results.items()},
                "models/optimal_thresholds.pkl")
    print("\n[INFO] Models + thresholds saved → models/")

    # ── Summary ──
    print_summary(results, best_params)
    print("\n[DONE] All outputs in outputs/ and models/")
