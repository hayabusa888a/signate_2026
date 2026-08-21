import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from janome.tokenizer import Tokenizer

from feature_eng import (
    RANDOM_STATE,
    target_col,
    id_col,
    cat_cols,
    build_features,
    align_categories,
)

# ---- text columns to vectorize (自由記述) ----
TFIDF_TEXT_COLS = ["企業概要", "今後のDX展望", "組織図"]
N_SVD = 40  # SVD components per text column

RUN_NAME = "tfidf_lgb"

# ---- artifact directories ----
FEATURES_DIR = Path("../features")
MODELS_DIR = Path("../models") / RUN_NAME
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

y = train[target_col].values
test_ids = test[id_col].values

# ---- structured features (from feature_eng) ----
X = build_features(train)
X_test = build_features(test)
X, X_test = align_categories(X, X_test)

# 計算済みの構造化特徴量を保存（再利用可）
X.to_pickle(FEATURES_DIR / "structured_train.pkl")
X_test.to_pickle(FEATURES_DIR / "structured_test.pkl")

# ---- pre-tokenize text once with janome (word surface, whitespace-joined) ----
tokenizer = Tokenizer()


def tokenize(text):
    if not isinstance(text, str) or text == "":
        return ""
    return " ".join(tokenizer.tokenize(text, wakati=True))


tok_cache = {}


def tokenized_series(df, col):
    key = (id(df), col)
    if key not in tok_cache:
        tok_cache[key] = df[col].fillna("").map(tokenize)
    return tok_cache[key]


train_tok = {c: tokenized_series(train, c) for c in TFIDF_TEXT_COLS}
test_tok = {c: tokenized_series(test, c) for c in TFIDF_TEXT_COLS}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),
    "verbosity": -1,
    "seed": RANDOM_STATE,
}


def build_tfidf_svd(tr_texts, va_texts, test_texts, col, seed):
    """Fit TF-IDF + SVD on the training fold only, transform valid/test (no leakage)."""
    vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2), max_features=20000)
    tr_mat = vec.fit_transform(tr_texts)
    n_comp = min(N_SVD, tr_mat.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    tr_svd = svd.fit_transform(tr_mat)
    va_svd = svd.transform(vec.transform(va_texts))
    test_svd = svd.transform(vec.transform(test_texts))
    cols = [f"{col}_svd{i}" for i in range(n_comp)]
    return (
        pd.DataFrame(tr_svd, columns=cols),
        pd.DataFrame(va_svd, columns=cols),
        pd.DataFrame(test_svd, columns=cols),
        {"vectorizer": vec, "svd": svd},
    )


for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    X_va = X.iloc[va_idx].reset_index(drop=True)
    y_tr, y_va = y[tr_idx], y[va_idx]
    X_te = X_test.reset_index(drop=True).copy()

    tr_parts, va_parts, te_parts = [X_tr], [X_va], [X_te]
    fold_vectorizers = {}
    for c in TFIDF_TEXT_COLS:
        tr_texts = train_tok[c].iloc[tr_idx].tolist()
        va_texts = train_tok[c].iloc[va_idx].tolist()
        test_texts = test_tok[c].tolist()
        tr_s, va_s, te_s, vecobj = build_tfidf_svd(tr_texts, va_texts, test_texts, c, RANDOM_STATE + fold)
        tr_parts.append(tr_s)
        va_parts.append(va_s)
        te_parts.append(te_s)
        fold_vectorizers[c] = vecobj

    X_tr_f = pd.concat(tr_parts, axis=1)
    X_va_f = pd.concat(va_parts, axis=1)
    X_te_f = pd.concat(te_parts, axis=1)

    dtrain = lgb.Dataset(X_tr_f, label=y_tr, categorical_feature=cat_cols)
    dvalid = lgb.Dataset(X_va_f, label=y_va, categorical_feature=cat_cols, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_pred[va_idx] = model.predict(X_va_f, num_iteration=model.best_iteration)
    test_pred += model.predict(X_te_f, num_iteration=model.best_iteration) / skf.n_splits

    # --- fold毎のモデルとベクトライザを保存 ---
    model.save_model(str(MODELS_DIR / f"lgb_fold{fold}.txt"), num_iteration=model.best_iteration)
    with open(MODELS_DIR / f"vectorizers_fold{fold}.pkl", "wb") as f:
        pickle.dump(fold_vectorizers, f)

    print(f"fold {fold}: best_iter={model.best_iteration}, logloss={model.best_score['valid_0']['binary_logloss']:.4f}")

best_th, best_f1 = 0.5, -1
for th in np.arange(0.05, 0.95, 0.01):
    f1 = f1_score(y, (oof_pred >= th).astype(int))
    if f1 > best_f1:
        best_f1, best_th = f1, th

print(f"\nOOF best F1={best_f1:.4f} at threshold={best_th:.2f}")
print("(baseline 0.6379 / feature_eng 0.6635)")

final_pred = (test_pred >= best_th).astype(int)
submission = pd.DataFrame({0: test_ids, 1: final_pred})
submission.to_csv("../submission/tfidf_submission.csv", index=False, header=False)
print("saved submission/tfidf_submission.csv")
print(submission[1].value_counts())

# --- OOF/test予測とメタ情報を保存 ---
np.save(MODELS_DIR / "oof_pred.npy", oof_pred)
np.save(MODELS_DIR / "test_pred.npy", test_pred)
pd.DataFrame({"企業ID": train[id_col], "oof_pred": oof_pred, "y": y}).to_csv(
    MODELS_DIR / "oof_pred.csv", index=False
)

meta = {
    "run_name": RUN_NAME,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    "oof_f1": round(float(best_f1), 4),
    "best_threshold": round(float(best_th), 2),
    "n_splits": int(skf.n_splits),
    "tfidf_text_cols": TFIDF_TEXT_COLS,
    "n_svd": N_SVD,
    "lgb_params": params,
    "submission": "submission/tfidf_submission.csv",
}
with open(MODELS_DIR / "meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\nartifacts saved to: {MODELS_DIR}")
print(f"features saved to:  {FEATURES_DIR}")
