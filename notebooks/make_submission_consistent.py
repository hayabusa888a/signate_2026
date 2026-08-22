"""順位/zスコア特徴を「train基準」で計算しtrain/test整合を取った提出。
adversarialで判明した順位特徴のtrain/test不整合を解消し、OOFの高AUCがtestに転移するか検証。
rank_lib: build_base(順位なし基本特徴) + RankTransformer(train fitでtestも同基準変換)。
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import sparse
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from janome.tokenizer import Tokenizer

from feat_lib import (RANDOM_STATE, target_col, id_col, cat_cols, te_cols,
                      TFIDF_TEXT_COLS, target_encode)
from rank_lib import build_base, RankTransformer

sys.stdout.reconfigure(encoding="utf-8")

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
y = train[target_col].values
test_ids = test[id_col].values

base_all = build_base(train)
base_test = build_base(test)

tokenizer = Tokenizer()
def tok(t):
    if not isinstance(t, str) or t == "":
        return ""
    return " ".join(tokenizer.tokenize(t, wakati=True))
print("tokenizing...", flush=True)
train_tok = {c: train[c].fillna("").map(tok) for c in TFIDF_TEXT_COLS}
test_tok = {c: test[c].fillna("").map(tok) for c in TFIDF_TEXT_COLS}

MEMBERS = ["LR_L2", "LinearSVC", "LR_raw", "LGB"]


def preprocess(Xs_fit, Xs_out, fit_idx, tok_fit, tok_out, out_is_test):
    num_cols = [c for c in Xs_fit.columns if c not in cat_cols]
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    num_f = sc.fit_transform(imp.fit_transform(Xs_fit[num_cols]))
    num_o = sc.transform(imp.transform(Xs_out[num_cols]))
    ohe = OneHotEncoder(handle_unknown="ignore")
    cat_f = ohe.fit_transform(Xs_fit[cat_cols].astype(str).fillna("missing"))
    cat_o = ohe.transform(Xs_out[cat_cols].astype(str).fillna("missing"))
    gmean = y[fit_idx].mean()
    te_f_c, te_o_c = [], []
    for c in te_cols:
        s_fit = train[c].iloc[fit_idx].astype(str)
        s_out = (test[c] if out_is_test else train[c].iloc[Xs_out.index]).astype(str)
        o_e, _, f_e = target_encode(s_fit, y[fit_idx], s_out, test[c].astype(str), gmean)
        te_f_c.append(f_e.reshape(-1, 1)); te_o_c.append(o_e.reshape(-1, 1))
    te_f = np.hstack(te_f_c); te_o = np.hstack(te_o_c)
    tesc = StandardScaler().fit(te_f); te_f = tesc.transform(te_f); te_o = tesc.transform(te_o)
    svd_f, svd_o, raw_f, raw_o = [], [], [], []
    for c in TFIDF_TEXT_COLS:
        vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2), max_features=20000)
        fm = vec.fit_transform(tok_fit[c]); om = vec.transform(tok_out[c])
        raw_f.append(fm); raw_o.append(om)
        n_comp = min(40, fm.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        ft = svd.fit_transform(fm); ot = svd.transform(om)
        s = StandardScaler().fit(ft); svd_f.append(s.transform(ft)); svd_o.append(s.transform(ot))
    base_f = [sparse.csr_matrix(num_f), cat_f, sparse.csr_matrix(te_f)]
    base_o = [sparse.csr_matrix(num_o), cat_o, sparse.csr_matrix(te_o)]
    Xsvd_f = sparse.hstack(base_f + [sparse.csr_matrix(np.hstack(svd_f))]).tocsr()
    Xsvd_o = sparse.hstack(base_o + [sparse.csr_matrix(np.hstack(svd_o))]).tocsr()
    Xraw_f = sparse.hstack(base_f + raw_f).tocsr()
    Xraw_o = sparse.hstack(base_o + raw_o).tocsr()
    return Xsvd_f, Xsvd_o, Xraw_f, Xraw_o


def fit_members(Xsvd_f, Xsvd_o, Xraw_f, Xraw_o, y_fit, rounds):
    out = {}
    out["LR_L2"] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, solver="liblinear").fit(Xsvd_f, y_fit).predict_proba(Xsvd_o)[:, 1]
    out["LinearSVC"] = LinearSVC(C=0.01, class_weight="balanced", max_iter=5000).fit(Xsvd_f, y_fit).decision_function(Xsvd_o)
    out["LR_raw"] = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, solver="liblinear").fit(Xraw_f, y_fit).predict_proba(Xraw_o)[:, 1]
    p = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05, "num_leaves": 15,
         "min_child_samples": 10, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
         "scale_pos_weight": (y_fit == 0).sum() / (y_fit == 1).sum(), "verbosity": -1, "seed": RANDOM_STATE}
    m = lgb.train(p, lgb.Dataset(Xsvd_f.toarray(), label=y_fit), num_boost_round=rounds)
    out["LGB"] = m.predict(Xsvd_o.toarray())
    return out


def to_rank(s): return rankdata(s) / len(s)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = {m: np.zeros(len(y)) for m in MEMBERS}
lgb_iters = []
for tr_idx, va_idx in skf.split(base_all, y):
    # 順位特徴: train-fold基準でfit → val/testに適用（train/test整合）
    rt = RankTransformer().fit(base_all.iloc[tr_idx])
    Xs_tr = rt.transform(base_all.iloc[tr_idx])   # index=tr_idx を保持(TE整合のため)
    Xs_va = rt.transform(base_all.iloc[va_idx])   # index=va_idx を保持
    Xsvd_f, Xsvd_v, Xraw_f, Xraw_v = preprocess(
        Xs_tr, Xs_va, tr_idx,
        {c: train_tok[c].iloc[tr_idx] for c in TFIDF_TEXT_COLS},
        {c: train_tok[c].iloc[va_idx] for c in TFIDF_TEXT_COLS}, out_is_test=False)
    p = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05, "num_leaves": 15,
         "min_child_samples": 10, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
         "scale_pos_weight": (y[tr_idx] == 0).sum() / (y[tr_idx] == 1).sum(), "verbosity": -1, "seed": RANDOM_STATE}
    mlgb = lgb.train(p, lgb.Dataset(Xsvd_f.toarray(), label=y[tr_idx]), num_boost_round=1000,
                     valid_sets=[lgb.Dataset(Xsvd_v.toarray(), label=y[va_idx])], callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_iters.append(mlgb.best_iteration)
    sc = fit_members(Xsvd_f, Xsvd_v, Xraw_f, Xraw_v, y[tr_idx], mlgb.best_iteration)
    for m in MEMBERS:
        oof[m][va_idx] = sc[m]

oof_ens = np.mean([to_rank(oof[m]) for m in MEMBERS], axis=0)
print(f"整合版 OOF AUC={roc_auc_score(y, oof_ens):.4f}")
bth, bf = 0, -1
for th in np.quantile(oof_ens, np.linspace(0.05, 0.95, 100)):
    f = f1_score(y, (oof_ens >= th).astype(int))
    if f > bf: bf, bth = f, th
print(f"整合版 OOF F1={bf:.4f} at rank-th={bth:.4f}  (現行チャンピオン0.8248)")

# ---- 全train基準でtest予測 ----
rt_full = RankTransformer().fit(base_all)
Xs_full = rt_full.transform(base_all)
Xs_test = rt_full.transform(base_test)   # ★ train基準でtestを変換（整合）
rounds = int(np.mean(lgb_iters))
Xsvd_f, Xsvd_te, Xraw_f, Xraw_te = preprocess(Xs_full, Xs_test, np.arange(len(y)),
                                              train_tok, test_tok, out_is_test=True)
test_sc = fit_members(Xsvd_f, Xsvd_te, Xraw_f, Xraw_te, y, rounds)
test_ens = np.mean([to_rank(test_sc[m]) for m in MEMBERS], axis=0)

for rate, tag in [(bf, "best"), (0.24, "rate24"), (0.30, "rate30"), (0.35, "rate35")]:
    pass
# CV最適閾値で提出
final = (test_ens >= bth).astype(int)
pd.DataFrame({0: test_ids, 1: final}).to_csv("../submission/consistent_submission.csv", index=False, header=False)
print(f"saved consistent_submission.csv 予測率={final.mean():.3f} ({final.sum()}件)")
np.save("../features/test_ens_consistent.npy", test_ens)
