import hashlib
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline, make_union

ROOT = Path(__file__).parent
NOT_FEATURES = ["dialog_id", "is_bot", "ID", "text"]


def load_config(args):
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    for arg in args:
        key, value = arg.split("=", 1)
        if key not in cfg:
            raise KeyError(f"в config.yaml нет параметра {key}")
        cfg[key] = yaml.safe_load(value)
    return cfg


def load_table(cfg, split):
    df = pd.read_csv(ROOT / cfg["processed_dir"] / f"{split}.csv")
    df["text"] = df["text"].fillna("")
    return df


def make_folds(y, groups, cfg):
    # оба участника одного диалога всегда попадают в один фолд
    cv = StratifiedGroupKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    return list(cv.split(y, y, groups))


def text_model(cfg):
    # tf-idf по символам и по словам + логистическая регрессия
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, cfg["char_ngram_max"]), min_df=cfg["min_df"],
                            max_features=cfg["max_features"], lowercase=False, sublinear_tf=True)
    words = TfidfVectorizer(ngram_range=(1, cfg["word_ngram_max"]), min_df=cfg["min_df"],
                            token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)
    return make_pipeline(make_union(chars, words), LogisticRegression(C=cfg["C"], max_iter=5000))


def add_text_scores(df, scores):
    # скор tf-idf самого участника и его собеседника
    df = df.assign(text_score=scores)
    opp = df[["dialog_id", "participant_index", "text_score"]].copy()
    opp["participant_index"] = 1 - opp["participant_index"]
    opp = opp.rename(columns={"text_score": "opp_text_score"})
    return df.merge(opp, on=["dialog_id", "participant_index"], how="left")


def features(df):
    return df[[c for c in df.columns if c not in NOT_FEATURES]]


def fit(train, cfg):
    y = train["is_bot"].to_numpy()
    tfidf = None
    if cfg["model"] == "logreg" or cfg["use_text"]:
        tfidf = text_model(cfg).fit(train["text"], y)
    if cfg["model"] == "logreg":
        return tfidf, None

    if cfg["use_text"]:
        # catboost учим на out-of-fold скорах, иначе он увидит переобученные скоры tf-idf
        oof = np.zeros(len(train))
        for tr, va in make_folds(y, train["dialog_id"], cfg):
            m = text_model(cfg).fit(train["text"].iloc[tr], y[tr])
            oof[va] = m.predict_proba(train["text"].iloc[va])[:, 1]
        train = add_text_scores(train, oof)

    booster = CatBoostClassifier(iterations=cfg["iterations"], learning_rate=cfg["learning_rate"], depth=cfg["depth"],
                                 l2_leaf_reg=cfg["l2_leaf_reg"], random_seed=cfg["seed"], thread_count=4,
                                 verbose=False, allow_writing_files=False)
    booster.fit(features(train), y)
    return tfidf, booster


def predict(models, df):
    tfidf, booster = models
    text_scores = tfidf.predict_proba(df["text"])[:, 1] if tfidf else None
    if booster is None:
        p = text_scores
    else:
        X = add_text_scores(df, text_scores) if tfidf else df
        p = booster.predict_proba(features(X))[:, 1]
    return np.clip(p, 1e-3, 1 - 1e-3)


def git_info():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain", "."], cwd=ROOT, capture_output=True, text=True)
    return {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else "no commits",
        "git_dirty": str(bool(status.stdout.strip())),
    }


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    cfg = load_config(sys.argv[1:])
    raw_dir = ROOT / cfg["raw_dir"]
    train, test = load_table(cfg, "train"), load_table(cfg, "test")
    y = train["is_bot"].to_numpy()

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        mlflow.create_experiment(cfg["experiment"], artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name=" ".join(sys.argv[1:]) or "default"):
        mlflow.log_params(cfg)
        mlflow.log_params({f"md5_{f}": md5(raw_dir / f) for f in ["train.json", "ytrain.csv", "test.json", "ytest.csv"]})
        mlflow.set_tags(git_info())
        mlflow.log_artifact(str(ROOT / "uv.lock"))

        # кросс-валидация
        oof = np.zeros(len(train))
        losses = []
        for k, (tr, va) in enumerate(make_folds(y, train["dialog_id"], cfg)):
            models = fit(train.iloc[tr], cfg)
            oof[va] = predict(models, train.iloc[va])
            losses.append(log_loss(y[va], oof[va]))
            mlflow.log_metric("fold_logloss", losses[-1], step=k)
            print(f"fold {k}: logloss {losses[-1]:.4f}")
        mlflow.log_metrics({"cv_logloss": np.mean(losses), "cv_logloss_std": np.std(losses), "oof_auc": roc_auc_score(y, oof)})
        print(f"cv logloss {np.mean(losses):.4f} ± {np.std(losses):.4f}, oof auc {roc_auc_score(y, oof):.4f}")

        # финальная модель на всем train и сабмит
        tfidf, booster = fit(train, cfg)
        submission = pd.DataFrame({"ID": test["ID"], "is_bot": predict((tfidf, booster), test)})
        sample = pd.read_csv(raw_dir / "sample_submission.csv")
        assert len(submission) == len(sample) and set(submission["ID"]) == set(sample["ID"])
        submission.to_csv(ROOT / "submission.csv", index=False)
        mlflow.log_artifact(str(ROOT / "submission.csv"))

        if tfidf is not None:
            mlflow.sklearn.log_model(tfidf, name="tfidf_model")
        if booster is not None:
            mlflow.catboost.log_model(booster, name="catboost_model")
            importance = pd.Series(booster.get_feature_importance(), booster.feature_names_).sort_values()
            fig, ax = plt.subplots(figsize=(7, 8))
            importance.tail(25).plot.barh(ax=ax)
            ax.set_title("Важность признаков (топ-25)")
            fig.tight_layout()
            mlflow.log_figure(fig, "feature_importance.png")


if __name__ == "__main__":
    main()
