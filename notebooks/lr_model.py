"""ロジスティック回帰モデル — v3と同じ特徴量セットで LGB と比較。
テキストは2パターン: (A) SVD圧縮 / (B) 生TF-IDF（スパース, LRの本領）。
数値は中央値補完+標準化、カテゴリはone-hot、target encodingとテキストはfold内fit。
"""
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from janome.tokenizer import Tokenizer

from feat_lib import (RANDOM_STATE, target_col, id_col, cat_cols, te_cols,
                      TFIDF_TEXT_COLS, build_static, target_encode)

sys.stdout.reconfigure(encoding="utf-8")

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
y = train[target_col].values

Xs = build_static(train)
Xs_test = build_static(test)

num_cols = [c for c in Xs.columns if c not in cat_cols]

# ---- tokenize (janome wakati, once) ----
tokenizer = Tokenizer()
def tok(t):
    if not isinstance(t, str) or t == "":
        return ""
    return " ".join(tokenizer.tokenize(t, wakati=True))

print("tokenizing...", flush=True)
train_tok = {c: train[c].fillna("").map(tok) for c in TFIDF_TEXT_COLS}
test_tok = {c: test[c].fillna("").map(tok) for c in TFIDF_TEXT_COLS}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def run(text_mode, C):
    """text_mode: 'svd' or 'raw'. returns OOF pred."""
    oof = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xs, y)):
        # --- numeric: impute + scale (fold内fit) ---
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        num_tr = sc.fit_transform(imp.fit_transform(Xs.iloc[tr_idx][num_cols]))
        num_va = sc.transform(imp.transform(Xs.iloc[va_idx][num_cols]))

        # --- categorical: one-hot ---
        ohe = OneHotEncoder(handle_unknown="ignore")
        cat_tr = ohe.fit_transform(Xs.iloc[tr_idx][cat_cols].astype(str).fillna("missing"))
        cat_va = ohe.transform(Xs.iloc[va_idx][cat_cols].astype(str).fillna("missing"))

        # --- target encoding (fold内fit) ---
        gmean = y[tr_idx].mean()
        te_tr_cols, te_va_cols = [], []
        for c in te_cols:
            va_e, _, tr_e = target_encode(train[c].iloc[tr_idx].astype(str), y[tr_idx],
                                          train[c].iloc[va_idx].astype(str), test[c].astype(str), gmean)
            te_tr_cols.append(tr_e.reshape(-1, 1)); te_va_cols.append(va_e.reshape(-1, 1))
        te_tr_m = np.hstack(te_tr_cols); te_va_m = np.hstack(te_va_cols)
        te_scaler = StandardScaler().fit(te_tr_m)
        te_tr_m = te_scaler.transform(te_tr_m); te_va_m = te_scaler.transform(te_va_m)

        blocks_tr = [sparse.csr_matrix(num_tr), cat_tr, sparse.csr_matrix(te_tr_m)]
        blocks_va = [sparse.csr_matrix(num_va), cat_va, sparse.csr_matrix(te_va_m)]

        # --- text ---
        for c in TFIDF_TEXT_COLS:
            vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2), max_features=20000)
            tr_mat = vec.fit_transform(train_tok[c].iloc[tr_idx])
            va_mat = vec.transform(train_tok[c].iloc[va_idx])
            if text_mode == "svd":
                n_comp = min(40, tr_mat.shape[1] - 1)
                svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE + fold)
                tr_t = svd.fit_transform(tr_mat)
                va_t = svd.transform(va_mat)
                s = StandardScaler().fit(tr_t)
                blocks_tr.append(sparse.csr_matrix(s.transform(tr_t)))
                blocks_va.append(sparse.csr_matrix(s.transform(va_t)))
            else:  # raw sparse tfidf
                blocks_tr.append(tr_mat)
                blocks_va.append(va_mat)

        Xtr = sparse.hstack(blocks_tr).tocsr()
        Xva = sparse.hstack(blocks_va).tocsr()

        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, solver="liblinear")
        clf.fit(Xtr, y[tr_idx])
        oof[va_idx] = clf.predict_proba(Xva)[:, 1]
    return oof


def best_f1(oof):
    bth, bf1 = 0.5, -1
    for th in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y, (oof >= th).astype(int))
        if f > bf1:
            bf1, bth = f, th
    return bf1, bth


print("\n=== Logistic Regression 比較 (LGB v3 = 0.7471) ===")
for mode in ["svd", "raw"]:
    for C in [0.1, 0.5, 1.0]:
        oof = run(mode, C)
        f1, th = best_f1(oof)
        tag = "SVD圧縮テキスト" if mode == "svd" else "生TF-IDF(スパース)"
        print(f"LR [{tag}] C={C}: OOF F1={f1:.4f} (th={th:.2f})")
