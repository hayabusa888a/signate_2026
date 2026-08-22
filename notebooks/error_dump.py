"""現行チャンピオン(アンサンブル)のOOFで外している企業を選び、生データを出力。"""
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
MEMBERS = ["LR_L2", "LinearSVC", "LR_raw", "LGB"]
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
    base_tr = [sparse.csr_matrix(num_tr), cat_tr, sparse.csr_matrix(te_tr)]
    base_va = [sparse.csr_matrix(num_va), cat_va, sparse.csr_matrix(te_va)]
    Xsvd_tr = sparse.hstack(base_tr + [sparse.csr_matrix(np.hstack(svd_tr))]).tocsr()
    Xsvd_va = sparse.hstack(base_va + [sparse.csr_matrix(np.hstack(svd_va))]).tocsr()
    Xraw_tr = sparse.hstack(base_tr + raw_tr).tocsr()
    Xraw_va = sparse.hstack(base_va + raw_va).tocsr()

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

def to_rank(s): return rankdata(s) / len(s)
ens = np.mean([to_rank(oof[m]) for m in MEMBERS], axis=0)
bth, bf = 0, -1
for th in np.quantile(ens, np.linspace(0.05, 0.95, 100)):
    f = f1_score(y, (ens >= th).astype(int))
    if f > bf: bf, bth = f, th
pred = (ens >= bth).astype(int)
print(f"OOF F1={bf:.4f} th={bth:.4f}")

train["_score"] = ens
train["_pred"] = pred
kind = np.where((y == 1) & (pred == 1), "TP",
        np.where((y == 0) & (pred == 0), "TN",
        np.where((y == 0) & (pred == 1), "FP", "FN")))
train["_区分"] = kind

# 確信度の高い誤り: FP(買わないのに高スコア) top3, FN(買うのに低スコア) top3
fp = train[train["_区分"] == "FP"].nlargest(3, "_score")
fn = train[train["_区分"] == "FN"].nsmallest(3, "_score")
picked = pd.concat([fp, fn])

train[["企業ID", "_score", "_pred", "購入フラグ", "_区分"]].to_csv("../features/oof_champion.csv", index=False)

# 生データを読みやすく出力
show_cols = [c for c in train.columns if c not in ["_score", "_pred", "_区分"]]
with open("../features/error_cases_raw.txt", "w", encoding="utf-8") as f:
    for _, row in picked.iterrows():
        f.write("=" * 90 + "\n")
        f.write(f"企業ID={row['企業ID']} 区分={row['_区分']} 実績購入={row['購入フラグ']} "
                f"予測={row['_pred']} スコア={row['_score']:.3f}\n")
        f.write("-" * 90 + "\n")
        for c in show_cols:
            val = row[c]
            if c in TFIDF_TEXT_COLS:
                f.write(f"\n■ {c}:\n{val}\n")
            else:
                f.write(f"{c}: {val}\n")
        f.write("\n\n")
print("saved features/error_cases_raw.txt / oof_champion.csv")
print("\nピックした企業:")
print(picked[["企業ID", "企業名", "業界", "上場種別", "_区分", "購入フラグ", "_score"]].round(3).to_string(index=False))
