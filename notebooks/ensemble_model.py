"""アンサンブル — 多様な線形/カーネル/木モデルのOOFをrank-averageで統合。
各モデルはfold内fitでOOFを作成。LGB v3は既存OOF(oof_pred_v3.csv, 同一fold seed)を再利用。
F1コンペ対応: スコアは順位平均→最後に閾値を1回探索。
"""
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from janome.tokenizer import Tokenizer

from feat_lib import (RANDOM_STATE, target_col, cat_cols, te_cols,
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


def make_fold_blocks(tr_idx, va_idx):
    """共有の前処理ブロックを1回だけ作る。svd/raw両方のテキスト表現を返す。"""
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
    tesc = StandardScaler().fit(te_tr)
    te_tr = tesc.transform(te_tr); te_va = tesc.transform(te_va)

    svd_tr, svd_va, raw_tr, raw_va = [], [], [], []
    for c in TFIDF_TEXT_COLS:
        vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2), max_features=20000)
        tm = vec.fit_transform(train_tok[c].iloc[tr_idx])
        vm = vec.transform(train_tok[c].iloc[va_idx])
        raw_tr.append(tm); raw_va.append(vm)
        n_comp = min(40, tm.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        tt = svd.fit_transform(tm); vt = svd.transform(vm)
        s = StandardScaler().fit(tt)
        svd_tr.append(s.transform(tt)); svd_va.append(s.transform(vt))

    base_tr = [sparse.csr_matrix(num_tr), cat_tr, sparse.csr_matrix(te_tr)]
    base_va = [sparse.csr_matrix(num_va), cat_va, sparse.csr_matrix(te_va)]
    X_svd_tr = sparse.hstack(base_tr + [sparse.csr_matrix(np.hstack(svd_tr))]).tocsr()
    X_svd_va = sparse.hstack(base_va + [sparse.csr_matrix(np.hstack(svd_va))]).tocsr()
    X_raw_tr = sparse.hstack(base_tr + raw_tr).tocsr()
    X_raw_va = sparse.hstack(base_va + raw_va).tocsr()
    return X_svd_tr, X_svd_va, X_raw_tr, X_raw_va


# メンバー定義: (名前, テキスト表現, 分類器ファクトリ, スコア関数)
def proba(clf, X): return clf.predict_proba(X)[:, 1]
def margin(clf, X): return clf.decision_function(X)

MEMBERS = {
    "LR_L2_svd":   ("svd", lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, solver="liblinear"), proba, False),
    "LR_L1_svd":   ("svd", lambda: LogisticRegression(C=0.5, penalty="l1", class_weight="balanced", max_iter=3000, solver="liblinear"), proba, False),
    "LinearSVC_svd": ("svd", lambda: LinearSVC(C=0.01, class_weight="balanced", max_iter=5000), margin, False),
    "SVC_rbf_svd": ("svd", lambda: SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced"), margin, True),
    "LR_raw":      ("raw", lambda: LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, solver="liblinear"), proba, False),
}

oof = {name: np.zeros(len(y)) for name in MEMBERS}

for tr_idx, va_idx in skf.split(Xs, y):
    Xsvd_tr, Xsvd_va, Xraw_tr, Xraw_va = make_fold_blocks(tr_idx, va_idx)
    for name, (mode, factory, scorer, dense) in MEMBERS.items():
        Xtr = Xsvd_tr if mode == "svd" else Xraw_tr
        Xva = Xsvd_va if mode == "svd" else Xraw_va
        if dense:
            Xtr, Xva = Xtr.toarray(), Xva.toarray()
        clf = factory(); clf.fit(Xtr, y[tr_idx])
        oof[name][va_idx] = scorer(clf, Xva)

# LGB v3 を既存OOFから追加（同一fold seed）
lgb_oof = pd.read_csv("../features/oof_pred_v3.csv").set_index("企業ID").loc[train["企業ID"], "oof_pred"].values
oof["LGB_v3"] = lgb_oof


def best_f1(scores):
    order = np.argsort(scores)
    ths = np.quantile(scores, np.linspace(0.05, 0.95, 80))
    bth, bf1 = 0, -1
    for th in ths:
        f = f1_score(y, (scores >= th).astype(int))
        if f > bf1:
            bf1, bth = f, th
    return bf1, bth


def to_rank(s):
    return rankdata(s) / len(s)


print("\n=== 個別モデル OOF F1 ===")
for name, s in oof.items():
    f1, _ = best_f1(s)
    print(f"  {name:16s}: {f1:.4f}")

# 予測相関（rank相関）
print("\n=== メンバー間 順位相関 ===")
rankdf = pd.DataFrame({n: to_rank(s) for n, s in oof.items()})
print(rankdf.corr(method="spearman").round(2).to_string())

# --- アンサンブル: rank-average ---
print("\n=== アンサンブル (rank-average) ===")
combos = {
    "全部(6)": list(oof.keys()),
    "線形3(LR_L2/L1/LinearSVC)": ["LR_L2_svd", "LR_L1_svd", "LinearSVC_svd"],
    "線形+RBF(4)": ["LR_L2_svd", "LR_L1_svd", "LinearSVC_svd", "SVC_rbf_svd"],
    "線形+LGB多様(4)": ["LR_L2_svd", "LinearSVC_svd", "LR_raw", "LGB_v3"],
    "全線形+RBF+LGB除くraw(5)": ["LR_L2_svd", "LR_L1_svd", "LinearSVC_svd", "SVC_rbf_svd", "LGB_v3"],
}
for tag, members in combos.items():
    ens = np.mean([to_rank(oof[m]) for m in members], axis=0)
    f1, th = best_f1(ens)
    print(f"  {tag:32s}: OOF F1={f1:.4f}")

# 参考: 単純平均(確率スケール混在)ではなくrank平均を採用している旨
print("\n(注) スコア尺度差を吸収するため rank-average を使用。閾値は最後に1回探索。")
