"""CV-LB乖離の検証: 順位系(現行/fold内) × 閾値(素朴/ネスト) の 2x2 でOOF F1を比較。
champion = LR_L2 + LinearSVC + LR_raw + LGB の rank-average。
LB実測=0.7603 と比べ、どの設定が実LBに近いCVを与えるかを見る。
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import sparse
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from janome.tokenizer import Tokenizer

from feat_lib import (RANDOM_STATE, target_col, cat_cols, te_cols,
                      TFIDF_TEXT_COLS, build_static, target_encode)
from rank_lib import build_base, RankTransformer

sys.stdout.reconfigure(encoding="utf-8")

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
y = train[target_col].values

tokenizer = Tokenizer()
def tok(t):
    if not isinstance(t, str) or t == "":
        return ""
    return " ".join(tokenizer.tokenize(t, wakati=True))

print("tokenizing...", flush=True)
train_tok = {c: train[c].fillna("").map(tok) for c in TFIDF_TEXT_COLS}

# variant A の構造化特徴（whole-train rank）は事前計算可
Xs_A = build_static(train)
base_all = build_base(train)   # variant B 用の基本特徴


def preprocess(Xs_tr, Xs_va, tr_idx, va_idx, num_cols):
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    num_tr = sc.fit_transform(imp.fit_transform(Xs_tr[num_cols]))
    num_va = sc.transform(imp.transform(Xs_va[num_cols]))
    ohe = OneHotEncoder(handle_unknown="ignore")
    cat_tr = ohe.fit_transform(Xs_tr[cat_cols].astype(str).fillna("missing"))
    cat_va = ohe.transform(Xs_va[cat_cols].astype(str).fillna("missing"))
    gmean = y[tr_idx].mean()
    te_tr_c, te_va_c = [], []
    for c in te_cols:
        va_e, _, tr_e = target_encode(train[c].iloc[tr_idx].astype(str), y[tr_idx],
                                      train[c].iloc[va_idx].astype(str), test[c].astype(str), gmean)
        te_tr_c.append(tr_e.reshape(-1, 1)); te_va_c.append(va_e.reshape(-1, 1))
    te_tr = np.hstack(te_tr_c); te_va = np.hstack(te_va_c)
    tesc = StandardScaler().fit(te_tr); te_tr = tesc.transform(te_tr); te_va = tesc.transform(te_va)
    svd_tr, svd_va, raw_tr, raw_va = [], [], [], []
    for c in TFIDF_TEXT_COLS:
        vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2), max_features=20000)
        tm = vec.fit_transform(train_tok[c].iloc[tr_idx]); vm = vec.transform(train_tok[c].iloc[va_idx])
        raw_tr.append(tm); raw_va.append(vm)
        n_comp = min(40, tm.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        tt = svd.fit_transform(tm); vt = svd.transform(vm)
        s = StandardScaler().fit(tt); svd_tr.append(s.transform(tt)); svd_va.append(s.transform(vt))
    base_tr = [sparse.csr_matrix(num_tr), cat_tr, sparse.csr_matrix(te_tr)]
    base_va = [sparse.csr_matrix(num_va), cat_va, sparse.csr_matrix(te_va)]
    Xsvd_tr = sparse.hstack(base_tr + [sparse.csr_matrix(np.hstack(svd_tr))]).tocsr()
    Xsvd_va = sparse.hstack(base_va + [sparse.csr_matrix(np.hstack(svd_va))]).tocsr()
    Xraw_tr = sparse.hstack(base_tr + raw_tr).tocsr()
    Xraw_va = sparse.hstack(base_va + raw_va).tocsr()
    return Xsvd_tr, Xsvd_va, Xraw_tr, Xraw_va


def fit_members(Xsvd_tr, Xsvd_va, Xraw_tr, Xraw_va, y_tr, rounds):
    out = {}
    out["LR_L2"] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, solver="liblinear").fit(Xsvd_tr, y_tr).predict_proba(Xsvd_va)[:, 1]
    out["LinearSVC"] = LinearSVC(C=0.01, class_weight="balanced", max_iter=5000).fit(Xsvd_tr, y_tr).decision_function(Xsvd_va)
    out["LR_raw"] = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, solver="liblinear").fit(Xraw_tr, y_tr).predict_proba(Xraw_va)[:, 1]
    p = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05, "num_leaves": 15,
         "min_child_samples": 10, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
         "scale_pos_weight": (y_tr == 0).sum() / (y_tr == 1).sum(), "verbosity": -1, "seed": RANDOM_STATE}
    m = lgb.train(p, lgb.Dataset(Xsvd_tr.toarray(), label=y_tr), num_boost_round=rounds)
    out["LGB"] = m.predict(Xsvd_va.toarray())
    return out


def to_rank(s):
    return rankdata(s) / len(s)


def naive_threshold_f1(ens):
    b, bf = 0, -1
    for th in np.quantile(ens, np.linspace(0.05, 0.95, 100)):
        f = f1_score(y, (ens >= th).astype(int))
        if f > bf:
            bf, b = f, th
    return bf


def nested_threshold_f1(ens, fold_id):
    """各foldの閾値を『他fold』で決めて適用（楽観バイアス除去）。"""
    pred = np.zeros(len(y), dtype=int)
    for k in np.unique(fold_id):
        other = fold_id != k
        this = fold_id == k
        b, bf = 0, -1
        for th in np.quantile(ens[other], np.linspace(0.05, 0.95, 100)):
            f = f1_score(y[other], (ens[other] >= th).astype(int))
            if f > bf:
                bf, b = f, th
        pred[this] = (ens[this] >= b).astype(int)
    return f1_score(y, pred)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
MEMBERS = ["LR_L2", "LinearSVC", "LR_raw", "LGB"]

results = {}
for variant in ["A_whole", "B_foldwise"]:
    oof = {m: np.zeros(len(y)) for m in MEMBERS}
    fold_id = np.zeros(len(y), dtype=int)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_id[va_idx] = fold
        if variant == "A_whole":
            Xs_tr = Xs_A.iloc[tr_idx].reset_index(drop=True)
            Xs_va = Xs_A.iloc[va_idx].reset_index(drop=True)
        else:
            rt = RankTransformer().fit(base_all.iloc[tr_idx])
            Xs_tr = rt.transform(base_all.iloc[tr_idx]).reset_index(drop=True)
            Xs_va = rt.transform(base_all.iloc[va_idx]).reset_index(drop=True)
        num_cols = [c for c in Xs_tr.columns if c not in cat_cols]
        Xsvd_tr, Xsvd_va, Xraw_tr, Xraw_va = preprocess(Xs_tr, Xs_va, tr_idx, va_idx, num_cols)
        # LGB rounds: early stopping で推定
        p = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05, "num_leaves": 15,
             "min_child_samples": 10, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
             "scale_pos_weight": (y[tr_idx] == 0).sum() / (y[tr_idx] == 1).sum(), "verbosity": -1, "seed": RANDOM_STATE}
        mm = lgb.train(p, lgb.Dataset(Xsvd_tr.toarray(), label=y[tr_idx]),
                       num_boost_round=1000, valid_sets=[lgb.Dataset(Xsvd_va.toarray(), label=y[va_idx])],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        sc = fit_members(Xsvd_tr, Xsvd_va, Xraw_tr, Xraw_va, y[tr_idx], mm.best_iteration)
        for m in MEMBERS:
            oof[m][va_idx] = sc[m]
        print(f"[{variant}] fold {fold} done", flush=True)

    ens = np.mean([to_rank(oof[m]) for m in MEMBERS], axis=0)
    results[variant] = {
        "naive": naive_threshold_f1(ens),
        "nested": nested_threshold_f1(ens, fold_id),
    }

print("\n================ 結果 (LB実測=0.7603) ================")
print(f"{'順位系':12s} {'素朴閾値':>10s} {'ネスト閾値':>10s}")
for v in ["A_whole", "B_foldwise"]:
    r = results[v]
    label = "現行(全train)" if v == "A_whole" else "fold内計算"
    print(f"{label:12s} {r['naive']:>10.4f} {r['nested']:>10.4f}")
