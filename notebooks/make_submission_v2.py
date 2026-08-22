"""Step1: 分散削減パイプライン
- transductive TF-IDF: 語彙とSVD基底を train+test 合算テキストでfit（ラベル不使用・正当）
  → test側のテキスト表現劣化を解消、全foldで同一表現
- seed averaging: fold分割seedを5通り変えて OOF/test を rank-average
- test予測は full-train再学習ではなく「各foldモデルの平均」（OOFと同じ条件）
評価は AUC / PR-AUC 重視。提出は rate固定(25/28/30%) + OOF最適閾値版。
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

SEEDS = [42, 123, 2026, 777, 555]
N_SVD = 40

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
y = train[target_col].values
test_ids = test[id_col].values
n_tr, n_te = len(train), len(test)

Xs = build_static(train)
Xs_test = build_static(test)
num_cols = [c for c in Xs.columns if c not in cat_cols]

# ---------- テキスト: transductive TF-IDF + SVD（1回だけfit, 決定的） ----------
tokenizer = Tokenizer()
def tok(t):
    if not isinstance(t, str) or t == "":
        return ""
    return " ".join(tokenizer.tokenize(t, wakati=True))

print("tokenizing...", flush=True)
all_tok = {}
for c in TFIDF_TEXT_COLS:
    all_tok[c] = pd.concat([train[c], test[c]]).fillna("").map(tok).tolist()

svd_train, svd_test = [], []     # dense SVD 表現
raw_train, raw_test = [], []     # sparse 生TF-IDF
for c in TFIDF_TEXT_COLS:
    vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2), max_features=20000)
    M = vec.fit_transform(all_tok[c])              # train+test 合算でfit
    raw_train.append(M[:n_tr]); raw_test.append(M[n_tr:])
    svd = TruncatedSVD(n_components=min(N_SVD, M.shape[1] - 1), random_state=42)
    S = svd.fit_transform(M)
    sc = StandardScaler().fit(S)
    S = sc.transform(S)
    svd_train.append(S[:n_tr]); svd_test.append(S[n_tr:])
svd_train = np.hstack(svd_train); svd_test = np.hstack(svd_test)
raw_train = sparse.hstack(raw_train).tocsr(); raw_test = sparse.hstack(raw_test).tocsr()
print(f"text ready: svd {svd_train.shape} / raw {raw_train.shape}", flush=True)

MEMBERS = ["LR_L2", "LinearSVC", "LR_raw", "LGB"]


def to_rank(s):
    return rankdata(s) / len(s)


def run_seed(seed):
    """1つのfold分割seedで OOF と test予測(fold平均) を返す。"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = {m: np.zeros(n_tr) for m in MEMBERS}
    tep = {m: np.zeros(n_te) for m in MEMBERS}
    for tr_idx, va_idx in skf.split(Xs, y):
        # 数値/カテゴリ/TE: fold内fit（TEはラベル使用のため必須）
        imp = SimpleImputer(strategy="median"); sc = StandardScaler()
        num_tr = sc.fit_transform(imp.fit_transform(Xs.iloc[tr_idx][num_cols]))
        num_va = sc.transform(imp.transform(Xs.iloc[va_idx][num_cols]))
        num_te = sc.transform(imp.transform(Xs_test[num_cols]))
        ohe = OneHotEncoder(handle_unknown="ignore")
        cat_tr = ohe.fit_transform(Xs.iloc[tr_idx][cat_cols].astype(str).fillna("missing"))
        cat_va = ohe.transform(Xs.iloc[va_idx][cat_cols].astype(str).fillna("missing"))
        cat_te = ohe.transform(Xs_test[cat_cols].astype(str).fillna("missing"))
        gmean = y[tr_idx].mean()
        te_tr_c, te_va_c, te_te_c = [], [], []
        for c in te_cols:
            va_e, test_e, tr_e = target_encode(
                train[c].iloc[tr_idx].astype(str), y[tr_idx],
                train[c].iloc[va_idx].astype(str), test[c].astype(str), gmean)
            te_tr_c.append(tr_e.reshape(-1, 1)); te_va_c.append(va_e.reshape(-1, 1))
            te_te_c.append(test_e.reshape(-1, 1))
        te_tr = np.hstack(te_tr_c); te_va = np.hstack(te_va_c); te_te = np.hstack(te_te_c)
        tesc = StandardScaler().fit(te_tr)
        te_tr = tesc.transform(te_tr); te_va = tesc.transform(te_va); te_te = tesc.transform(te_te)

        base_tr = [sparse.csr_matrix(num_tr), cat_tr, sparse.csr_matrix(te_tr)]
        base_va = [sparse.csr_matrix(num_va), cat_va, sparse.csr_matrix(te_va)]
        base_te = [sparse.csr_matrix(num_te), cat_te, sparse.csr_matrix(te_te)]
        Xsvd_tr = sparse.hstack(base_tr + [sparse.csr_matrix(svd_train[tr_idx])]).tocsr()
        Xsvd_va = sparse.hstack(base_va + [sparse.csr_matrix(svd_train[va_idx])]).tocsr()
        Xsvd_te = sparse.hstack(base_te + [sparse.csr_matrix(svd_test)]).tocsr()
        Xraw_tr = sparse.hstack(base_tr + [raw_train[tr_idx]]).tocsr()
        Xraw_va = sparse.hstack(base_va + [raw_train[va_idx]]).tocsr()
        Xraw_te = sparse.hstack(base_te + [raw_test]).tocsr()

        m1 = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, solver="liblinear").fit(Xsvd_tr, y[tr_idx])
        oof["LR_L2"][va_idx] = m1.predict_proba(Xsvd_va)[:, 1]
        tep["LR_L2"] += m1.predict_proba(Xsvd_te)[:, 1] / 5
        m2 = LinearSVC(C=0.01, class_weight="balanced", max_iter=5000).fit(Xsvd_tr, y[tr_idx])
        oof["LinearSVC"][va_idx] = m2.decision_function(Xsvd_va)
        tep["LinearSVC"] += m2.decision_function(Xsvd_te) / 5
        m3 = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, solver="liblinear").fit(Xraw_tr, y[tr_idx])
        oof["LR_raw"][va_idx] = m3.predict_proba(Xraw_va)[:, 1]
        tep["LR_raw"] += m3.predict_proba(Xraw_te)[:, 1] / 5
        p = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
             "num_leaves": 15, "min_child_samples": 10, "feature_fraction": 0.8,
             "bagging_fraction": 0.8, "bagging_freq": 1,
             "scale_pos_weight": (y[tr_idx] == 0).sum() / (y[tr_idx] == 1).sum(),
             "verbosity": -1, "seed": seed}
        m4 = lgb.train(p, lgb.Dataset(Xsvd_tr.toarray(), label=y[tr_idx]), num_boost_round=1000,
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
    print(f"seed {seed}: OOF AUC={roc_auc_score(y, oe):.4f} PR-AUC={average_precision_score(y, oe):.4f}", flush=True)

oof_final = np.mean([to_rank(o) for o in oof_list], axis=0)
te_final = np.mean([to_rank(t) for t in te_list], axis=0)

auc = roc_auc_score(y, oof_final)
prauc = average_precision_score(y, oof_final)
print(f"\n=== seed平均 OOF AUC={auc:.4f} PR-AUC={prauc:.4f} (現行champ AUC=0.9545 PR=0.8960) ===")

# rate固定のF1（比較・閾値ノイズ排除）
for rate in [0.21, 0.25, 0.28, 0.30, 0.35]:
    th = np.quantile(oof_final, 1 - rate)
    print(f"  OOF F1@rate{rate:.2f} = {f1_score(y, (oof_final >= th).astype(int)):.4f}")
bth, bf = 0, -1
for th in np.quantile(oof_final, np.linspace(0.05, 0.95, 100)):
    f = f1_score(y, (oof_final >= th).astype(int))
    if f > bf: bf, bth = f, th
print(f"  OOF best F1={bf:.4f}")

np.save("../features/v2_oof.npy", oof_final)
np.save("../features/v2_test.npy", te_final)

# 提出: rate固定 25/28/30 + 35(現LBベスト帯)
for rate in [0.25, 0.28, 0.30, 0.35]:
    th = np.quantile(te_final, 1 - rate)
    pred = (te_final >= th).astype(int)
    tag = f"v2_rate{int(rate*100)}"
    pd.DataFrame({0: test_ids, 1: pred}).to_csv(f"../submission/{tag}.csv", index=False, header=False)
    print(f"saved {tag}.csv 予測率={pred.mean():.3f} ({pred.sum()}件)")
