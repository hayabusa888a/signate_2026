"""Step2: pseudo-labeling
v2のtestスコア(v2_test.npy)から確信度の高いtest企業に仮ラベルを付与し、
trainに追加して再学習。test 800社>train 742社なのでデータ実質倍増の効果を狙う。
- 仮ラベル: スコア上位 P_HI% → 1 / 下位 P_LO% → 0（境界帯は使わない）
- 評価: 元のtrain 742行に対するOOF（pseudo行は各foldの学習側にのみ追加）
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import sparse
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from janome.tokenizer import Tokenizer

from feat_lib import (target_col, id_col, cat_cols, te_cols,
                      TFIDF_TEXT_COLS, build_static, target_encode)

sys.stdout.reconfigure(encoding="utf-8")

SEEDS = [42, 123, 2026]
P_HI = 0.15   # test上位15% → 仮ラベル1 (120社)
P_LO = 0.40   # test下位40% → 仮ラベル0 (320社)
N_SVD = 40

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
y = train[target_col].values
test_ids = test[id_col].values
n_tr, n_te = len(train), len(test)

v2_test = np.load("../features/v2_test.npy")
hi_th = np.quantile(v2_test, 1 - P_HI)
lo_th = np.quantile(v2_test, P_LO)
pseudo_idx = np.where((v2_test >= hi_th) | (v2_test <= lo_th))[0]
pseudo_y = (v2_test[pseudo_idx] >= hi_th).astype(int)
print(f"pseudo: {len(pseudo_idx)}社 (陽性{pseudo_y.sum()} / 陰性{(pseudo_y==0).sum()})")

Xs = build_static(train)
Xs_test = build_static(test)
num_cols = [c for c in Xs.columns if c not in cat_cols]

tokenizer = Tokenizer()
def tok(t):
    if not isinstance(t, str) or t == "":
        return ""
    return " ".join(tokenizer.tokenize(t, wakati=True))
print("tokenizing...", flush=True)
all_tok = {c: pd.concat([train[c], test[c]]).fillna("").map(tok).tolist() for c in TFIDF_TEXT_COLS}

svd_all_l, raw_all_l = [], []
for c in TFIDF_TEXT_COLS:
    vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2), max_features=20000)
    M = vec.fit_transform(all_tok[c])
    raw_all_l.append(M)
    svd = TruncatedSVD(n_components=min(N_SVD, M.shape[1] - 1), random_state=42)
    S = StandardScaler().fit_transform(svd.fit_transform(M))
    svd_all_l.append(S)
svd_all = np.hstack(svd_all_l)                 # (n_tr+n_te, d)
raw_all = sparse.hstack(raw_all_l).tocsr()
print("text ready", flush=True)

# 全行(train+test)ぶんの構造化特徴・カテゴリ・TE用素材
Xs_all = pd.concat([Xs, Xs_test], ignore_index=True)
cat_src = pd.concat([train[te_cols], test[te_cols]], ignore_index=True).astype(str)

MEMBERS = ["LR_L2", "LinearSVC", "LR_raw", "LGB"]


def to_rank(s):
    return rankdata(s) / len(s)


def run_seed(seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = {m: np.zeros(n_tr) for m in MEMBERS}
    tep = {m: np.zeros(n_te) for m in MEMBERS}
    for tr_idx, va_idx in skf.split(Xs, y):
        # 学習集合 = train(fold) + pseudo(test行)
        fit_rows = np.r_[tr_idx, n_tr + pseudo_idx]          # Xs_all上の行番号
        y_fit = np.r_[y[tr_idx], pseudo_y]

        imp = SimpleImputer(strategy="median"); sc = StandardScaler()
        num_f = sc.fit_transform(imp.fit_transform(Xs_all.iloc[fit_rows][num_cols]))
        num_va = sc.transform(imp.transform(Xs_all.iloc[va_idx][num_cols]))
        num_te = sc.transform(imp.transform(Xs_all.iloc[n_tr:][num_cols]))
        ohe = OneHotEncoder(handle_unknown="ignore")
        cat_f = ohe.fit_transform(Xs_all.iloc[fit_rows][cat_cols].astype(str).fillna("missing"))
        cat_va = ohe.transform(Xs_all.iloc[va_idx][cat_cols].astype(str).fillna("missing"))
        cat_te = ohe.transform(Xs_all.iloc[n_tr:][cat_cols].astype(str).fillna("missing"))
        gmean = y_fit.mean()
        te_f_c, te_va_c, te_te_c = [], [], []
        for c in te_cols:
            s_fit = cat_src[c].iloc[fit_rows]
            va_e, test_e, f_e = target_encode(s_fit, y_fit,
                                              cat_src[c].iloc[va_idx], cat_src[c].iloc[n_tr:], gmean)
            te_f_c.append(f_e.reshape(-1, 1)); te_va_c.append(va_e.reshape(-1, 1))
            te_te_c.append(test_e.reshape(-1, 1))
        te_f = np.hstack(te_f_c); te_va = np.hstack(te_va_c); te_te = np.hstack(te_te_c)
        tesc = StandardScaler().fit(te_f)
        te_f = tesc.transform(te_f); te_va = tesc.transform(te_va); te_te = tesc.transform(te_te)

        base_f = [sparse.csr_matrix(num_f), cat_f, sparse.csr_matrix(te_f)]
        base_va = [sparse.csr_matrix(num_va), cat_va, sparse.csr_matrix(te_va)]
        base_te = [sparse.csr_matrix(num_te), cat_te, sparse.csr_matrix(te_te)]
        Xsvd_f = sparse.hstack(base_f + [sparse.csr_matrix(svd_all[fit_rows])]).tocsr()
        Xsvd_va = sparse.hstack(base_va + [sparse.csr_matrix(svd_all[va_idx])]).tocsr()
        Xsvd_te = sparse.hstack(base_te + [sparse.csr_matrix(svd_all[n_tr:])]).tocsr()
        Xraw_f = sparse.hstack(base_f + [raw_all[fit_rows]]).tocsr()
        Xraw_va = sparse.hstack(base_va + [raw_all[va_idx]]).tocsr()
        Xraw_te = sparse.hstack(base_te + [raw_all[n_tr:]]).tocsr()

        m1 = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, solver="liblinear").fit(Xsvd_f, y_fit)
        oof["LR_L2"][va_idx] = m1.predict_proba(Xsvd_va)[:, 1]
        tep["LR_L2"] += m1.predict_proba(Xsvd_te)[:, 1] / 5
        m2 = LinearSVC(C=0.01, class_weight="balanced", max_iter=5000).fit(Xsvd_f, y_fit)
        oof["LinearSVC"][va_idx] = m2.decision_function(Xsvd_va)
        tep["LinearSVC"] += m2.decision_function(Xsvd_te) / 5
        m3 = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, solver="liblinear").fit(Xraw_f, y_fit)
        oof["LR_raw"][va_idx] = m3.predict_proba(Xraw_va)[:, 1]
        tep["LR_raw"] += m3.predict_proba(Xraw_te)[:, 1] / 5
        p = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
             "num_leaves": 15, "min_child_samples": 10, "feature_fraction": 0.8,
             "bagging_fraction": 0.8, "bagging_freq": 1,
             "scale_pos_weight": (y_fit == 0).sum() / (y_fit == 1).sum(),
             "verbosity": -1, "seed": seed}
        m4 = lgb.train(p, lgb.Dataset(Xsvd_f.toarray(), label=y_fit), num_boost_round=1000,
                       valid_sets=[lgb.Dataset(Xsvd_va.toarray(), label=y[va_idx])],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        oof["LGB"][va_idx] = m4.predict(Xsvd_va.toarray(), num_iteration=m4.best_iteration)
        tep["LGB"] += m4.predict(Xsvd_te.toarray(), num_iteration=m4.best_iteration) / 5

    oof_ens = np.mean([to_rank(oof[m]) for m in MEMBERS], axis=0)
    te_ens = np.mean([to_rank(tep[m]) for m in MEMBERS], axis=0)
    return oof_ens, te_ens


oof_list, te_list = [], []
for seed in SEEDS:
    oe, te = run_seed(seed)
    oof_list.append(oe); te_list.append(te)
    print(f"seed {seed}: OOF AUC={roc_auc_score(y, oe):.4f}", flush=True)

oof_final = np.mean([to_rank(o) for o in oof_list], axis=0)
te_final = np.mean([to_rank(t) for t in te_list], axis=0)
print(f"\n=== pseudo版 OOF AUC={roc_auc_score(y, oof_final):.4f} PR-AUC={average_precision_score(y, oof_final):.4f} ===")
for rate in [0.25, 0.28, 0.30, 0.35]:
    th = np.quantile(oof_final, 1 - rate)
    print(f"  OOF F1@rate{rate:.2f} = {f1_score(y, (oof_final >= th).astype(int)):.4f}")

np.save("../features/pseudo_test.npy", te_final)
for rate in [0.25, 0.30]:
    th = np.quantile(te_final, 1 - rate)
    pred = (te_final >= th).astype(int)
    tag = f"pseudo_rate{int(rate*100)}"
    pd.DataFrame({0: test_ids, 1: pred}).to_csv(f"../submission/{tag}.csv", index=False, header=False)
    print(f"saved {tag}.csv ({pred.sum()}件)")
