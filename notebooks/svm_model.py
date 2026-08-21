"""SVM モデル — LRと同じ特徴量パイプラインで LinearSVC / SVC(RBF) を評価。
LGB v3=0.7471 / LR best=0.8127 との比較。
"""
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC, SVC
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
num_cols = [c for c in Xs.columns if c not in cat_cols]

tokenizer = Tokenizer()
def tok(t):
    if not isinstance(t, str) or t == "":
        return ""
    return " ".join(tokenizer.tokenize(t, wakati=True))

print("tokenizing...", flush=True)
train_tok = {c: train[c].fillna("").map(tok) for c in TFIDF_TEXT_COLS}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_fold(tr_idx, va_idx, text_mode, dense=False):
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    num_tr = sc.fit_transform(imp.fit_transform(Xs.iloc[tr_idx][num_cols]))
    num_va = sc.transform(imp.transform(Xs.iloc[va_idx][num_cols]))

    ohe = OneHotEncoder(handle_unknown="ignore")
    cat_tr = ohe.fit_transform(Xs.iloc[tr_idx][cat_cols].astype(str).fillna("missing"))
    cat_va = ohe.transform(Xs.iloc[va_idx][cat_cols].astype(str).fillna("missing"))

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

    for c in TFIDF_TEXT_COLS:
        vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2), max_features=20000)
        tr_mat = vec.fit_transform(train_tok[c].iloc[tr_idx])
        va_mat = vec.transform(train_tok[c].iloc[va_idx])
        if text_mode == "svd":
            n_comp = min(40, tr_mat.shape[1] - 1)
            svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
            tr_t = svd.fit_transform(tr_mat); va_t = svd.transform(va_mat)
            s = StandardScaler().fit(tr_t)
            blocks_tr.append(sparse.csr_matrix(s.transform(tr_t)))
            blocks_va.append(sparse.csr_matrix(s.transform(va_t)))
        else:
            blocks_tr.append(tr_mat); blocks_va.append(va_mat)

    Xtr = sparse.hstack(blocks_tr).tocsr(); Xva = sparse.hstack(blocks_va).tocsr()
    if dense:
        Xtr = Xtr.toarray(); Xva = Xva.toarray()
    return Xtr, Xva


def run(make_clf, text_mode, dense=False):
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(Xs, y):
        Xtr, Xva = build_fold(tr_idx, va_idx, text_mode, dense=dense)
        clf = make_clf()
        clf.fit(Xtr, y[tr_idx])
        oof[va_idx] = clf.decision_function(Xva)
    return oof


def best_f1(scores):
    """decision_function出力に対し分位点で閾値探索。"""
    qs = np.quantile(scores, np.linspace(0.05, 0.95, 60))
    bth, bf1 = 0, -1
    for th in qs:
        f = f1_score(y, (scores >= th).astype(int))
        if f > bf1:
            bf1, bth = f, th
    return bf1, bth


print("\n=== SVM 比較 (LGB v3=0.7471 / LR best=0.8127) ===")

# LinearSVC（線形, スパースOK）
for C in [0.01, 0.05, 0.1, 0.5]:
    for mode in ["svd", "raw"]:
        oof = run(lambda: LinearSVC(C=C, class_weight="balanced", max_iter=5000), mode)
        f1, th = best_f1(oof)
        tag = "SVD" if mode == "svd" else "raw"
        print(f"LinearSVC [{tag}] C={C}: OOF F1={f1:.4f}")

# SVC RBF（非線形, dense, SVDテキストのみ）
for C in [1.0, 5.0, 10.0]:
    for gamma in ["scale", 0.01]:
        oof = run(lambda: SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced"), "svd", dense=True)
        f1, th = best_f1(oof)
        print(f"SVC-RBF [SVD] C={C} gamma={gamma}: OOF F1={f1:.4f}")
