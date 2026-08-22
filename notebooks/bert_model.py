"""BERT埋め込みをPCA圧縮(fold内fit)してチャンピオンに追加し、OOF F1を検証。
比較対象: アンサンブル(TF-IDF-SVDのみ) OOF=0.8248。
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
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer

from feat_lib import (RANDOM_STATE, target_col, cat_cols, te_cols,
                      TFIDF_TEXT_COLS, build_static, target_encode)

sys.stdout.reconfigure(encoding="utf-8")

BERT_COLS = ["今後のDX展望", "企業概要", "組織図"]
BERT_PCA = {"今後のDX展望": 50, "企業概要": 30, "組織図": 20}

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
y = train[target_col].values

Xs = build_static(train); Xs_test = build_static(test)
num_cols = [c for c in Xs.columns if c not in cat_cols]

# BERT埋め込みロード
bert_tr = {c: np.load(f"../features/bert_{c}_train.npy") for c in BERT_COLS}
bert_te = {c: np.load(f"../features/bert_{c}_test.npy") for c in BERT_COLS}

tokenizer = Tokenizer()
def tok(t):
    if not isinstance(t, str) or t == "":
        return ""
    return " ".join(tokenizer.tokenize(t, wakati=True))
print("tokenizing...", flush=True)
train_tok = {c: train[c].fillna("").map(tok) for c in TFIDF_TEXT_COLS}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
MEMBERS = ["LR_L2", "LinearSVC", "LR_raw", "LGB"]


def to_rank(s): return rankdata(s) / len(s)


def run(use_bert):
    oof = {m: np.zeros(len(y)) for m in MEMBERS}
    for tr_idx, va_idx in skf.split(Xs, y):
        imp = SimpleImputer(strategy="median"); sc = StandardScaler()
        num_tr = sc.fit_transform(imp.fit_transform(Xs.iloc[tr_idx][num_cols]))
        num_va = sc.transform(imp.transform(Xs.iloc[va_idx][num_cols]))
        ohe = OneHotEncoder(handle_unknown="ignore")
        cat_tr = ohe.fit_transform(Xs.iloc[tr_idx][cat_cols].astype(str).fillna("missing"))
        cat_va = ohe.transform(Xs.iloc[va_idx][cat_cols].astype(str).fillna("missing"))
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

        # BERT: fold内でPCA圧縮
        bert_tr_parts, bert_va_parts = [], []
        if use_bert:
            for c in BERT_COLS:
                Etr = bert_tr[c][tr_idx]; Eva = bert_tr[c][va_idx]
                bsc = StandardScaler().fit(Etr)
                pca = PCA(n_components=BERT_PCA[c], random_state=RANDOM_STATE)
                pt = pca.fit_transform(bsc.transform(Etr)); pv = pca.transform(bsc.transform(Eva))
                ps = StandardScaler().fit(pt)
                bert_tr_parts.append(ps.transform(pt)); bert_va_parts.append(ps.transform(pv))

        base_tr = [sparse.csr_matrix(num_tr), cat_tr, sparse.csr_matrix(te_tr)]
        base_va = [sparse.csr_matrix(num_va), cat_va, sparse.csr_matrix(te_va)]
        svd_blk_tr = sparse.csr_matrix(np.hstack(svd_tr)); svd_blk_va = sparse.csr_matrix(np.hstack(svd_va))
        extra_tr = [sparse.csr_matrix(np.hstack(bert_tr_parts))] if use_bert else []
        extra_va = [sparse.csr_matrix(np.hstack(bert_va_parts))] if use_bert else []

        Xsvd_tr = sparse.hstack(base_tr + [svd_blk_tr] + extra_tr).tocsr()
        Xsvd_va = sparse.hstack(base_va + [svd_blk_va] + extra_va).tocsr()
        Xraw_tr = sparse.hstack(base_tr + raw_tr + extra_tr).tocsr()
        Xraw_va = sparse.hstack(base_va + raw_va + extra_va).tocsr()

        dtr = lgb.Dataset(Xsvd_tr.toarray(), label=y[tr_idx])
        dva = lgb.Dataset(Xsvd_va.toarray(), label=y[va_idx])
        p = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05, "num_leaves": 15,
             "min_child_samples": 10, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
             "scale_pos_weight": (y[tr_idx] == 0).sum() / (y[tr_idx] == 1).sum(), "verbosity": -1, "seed": RANDOM_STATE}
        mlgb = lgb.train(p, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof["LR_L2"][va_idx] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, solver="liblinear").fit(Xsvd_tr, y[tr_idx]).predict_proba(Xsvd_va)[:, 1]
        oof["LinearSVC"][va_idx] = LinearSVC(C=0.01, class_weight="balanced", max_iter=5000).fit(Xsvd_tr, y[tr_idx]).decision_function(Xsvd_va)
        oof["LR_raw"][va_idx] = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, solver="liblinear").fit(Xraw_tr, y[tr_idx]).predict_proba(Xraw_va)[:, 1]
        oof["LGB"][va_idx] = mlgb.predict(Xsvd_va.toarray())
    ens = np.mean([to_rank(oof[m]) for m in MEMBERS], axis=0)
    bf = -1
    for th in np.quantile(ens, np.linspace(0.05, 0.95, 100)):
        f = f1_score(y, (ens >= th).astype(int))
        if f > bf: bf = f
    # 個別も
    singles = {m: max(f1_score(y, (to_rank(oof[m]) >= th).astype(int))
                      for th in np.quantile(to_rank(oof[m]), np.linspace(0.05, 0.95, 60))) for m in MEMBERS}
    return bf, singles


print("=== BERTなし(現行) ===", flush=True)
f_no, s_no = run(False)
print(f"アンサンブル OOF F1={f_no:.4f}  個別={ {k: round(v,4) for k,v in s_no.items()} }")

print("\n=== BERTあり ===", flush=True)
f_yes, s_yes = run(True)
print(f"アンサンブル OOF F1={f_yes:.4f}  個別={ {k: round(v,4) for k,v in s_yes.items()} }")

print(f"\n差分: {f_yes - f_no:+.4f} (現行0.8248)")
