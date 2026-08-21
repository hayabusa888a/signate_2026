"""最良ロジックで提出ファイルを作成。
champion = 線形+LGB多様(4): LR_L2_svd + LinearSVC_svd + LR_raw + LGB_svd の rank-average (OOF F1=0.8278)。
- fold内OOFで閾値決定 / 全train再学習でtest予測 / rank-average→閾値適用。
- 全メンバー同一特徴パイプライン(fold内fit・全train fit)でtrain/test整合を担保。
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

from feat_lib import (RANDOM_STATE, target_col, id_col, cat_cols, te_cols,
                      TFIDF_TEXT_COLS, build_static, target_encode)

sys.stdout.reconfigure(encoding="utf-8")

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
y = train[target_col].values
test_ids = test[id_col].values

Xs = build_static(train)
Xs_test = build_static(test)
num_cols = [c for c in Xs.columns if c not in cat_cols]

tokenizer = Tokenizer()
def tok(t):
    if not isinstance(t, str) or t == "":
        return ""
    return " ".join(tokenizer.tokenize(t, wakati=True))

print("tokenizing...", flush=True)
train_tok = {c: train[c].fillna("").map(tok) for c in TFIDF_TEXT_COLS}
test_tok = {c: test[c].fillna("").map(tok) for c in TFIDF_TEXT_COLS}


def make_blocks(fit_idx, Xs_fit, Xs_out, tok_fit, tok_out, ids_fit, ids_out, out_is_test):
    """fit_idx(train内)で前処理をfitし、出力側(val or test)へtransform。
    svd/raw両方のテキスト表現を含む特徴行列を返す。"""
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
        s = StandardScaler().fit(ft)
        svd_f.append(s.transform(ft)); svd_o.append(s.transform(ot))

    base_f = [sparse.csr_matrix(num_f), cat_f, sparse.csr_matrix(te_f)]
    base_o = [sparse.csr_matrix(num_o), cat_o, sparse.csr_matrix(te_o)]
    X_svd_f = sparse.hstack(base_f + [sparse.csr_matrix(np.hstack(svd_f))]).tocsr()
    X_svd_o = sparse.hstack(base_o + [sparse.csr_matrix(np.hstack(svd_o))]).tocsr()
    X_raw_f = sparse.hstack(base_f + raw_f).tocsr()
    X_raw_o = sparse.hstack(base_o + raw_o).tocsr()
    return (X_svd_f, X_svd_o, X_raw_f, X_raw_o)


def fit_members(X_svd_f, X_svd_o, X_raw_f, X_raw_o, y_fit, lgb_rounds):
    """champion 4メンバーを学習し、出力側スコアを返す。"""
    scores = {}
    lr2 = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, solver="liblinear").fit(X_svd_f, y_fit)
    scores["LR_L2_svd"] = lr2.predict_proba(X_svd_o)[:, 1]
    svc = LinearSVC(C=0.01, class_weight="balanced", max_iter=5000).fit(X_svd_f, y_fit)
    scores["LinearSVC_svd"] = svc.decision_function(X_svd_o)
    lrr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, solver="liblinear").fit(X_raw_f, y_fit)
    scores["LR_raw"] = lrr.predict_proba(X_raw_o)[:, 1]
    dtr = lgb.Dataset(X_svd_f.toarray(), label=y_fit)
    params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
              "num_leaves": 15, "min_child_samples": 10, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1,
              "scale_pos_weight": (y_fit == 0).sum() / (y_fit == 1).sum(),
              "verbosity": -1, "seed": RANDOM_STATE}
    m = lgb.train(params, dtr, num_boost_round=lgb_rounds)
    scores["LGB_svd"] = m.predict(X_svd_o.toarray())
    return scores


MEMBERS = ["LR_L2_svd", "LinearSVC_svd", "LR_raw", "LGB_svd"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ===== 1) fold内OOF（閾値決定 + LGB rounds推定） =====
oof = {m: np.zeros(len(y)) for m in MEMBERS}
lgb_best_iters = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(Xs, y)):
    # LGB rounds を early stopping で推定
    Xsvd_f, Xsvd_v, Xraw_f, Xraw_v = make_blocks(
        tr_idx, Xs.iloc[tr_idx], Xs.iloc[va_idx],
        {c: train_tok[c].iloc[tr_idx] for c in TFIDF_TEXT_COLS},
        {c: train_tok[c].iloc[va_idx] for c in TFIDF_TEXT_COLS},
        None, None, out_is_test=False)
    dtr = lgb.Dataset(Xsvd_f.toarray(), label=y[tr_idx])
    dva = lgb.Dataset(Xsvd_v.toarray(), label=y[va_idx])
    params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
              "num_leaves": 15, "min_child_samples": 10, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1,
              "scale_pos_weight": (y[tr_idx] == 0).sum() / (y[tr_idx] == 1).sum(),
              "verbosity": -1, "seed": RANDOM_STATE}
    mlgb = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_best_iters.append(mlgb.best_iteration)

    sc = fit_members(Xsvd_f, Xsvd_v, Xraw_f, Xraw_v, y[tr_idx], mlgb.best_iteration)
    for m in MEMBERS:
        oof[m][va_idx] = sc[m]
    print(f"fold {fold} done (lgb best_iter={mlgb.best_iteration})", flush=True)

def to_rank(s):
    return rankdata(s) / len(s)

oof_ens = np.mean([to_rank(oof[m]) for m in MEMBERS], axis=0)
best_th, best_f1 = 0, -1
for th in np.quantile(oof_ens, np.linspace(0.05, 0.95, 100)):
    f = f1_score(y, (oof_ens >= th).astype(int))
    if f > best_f1:
        best_f1, best_th = f, th
print(f"\nOOF ensemble F1={best_f1:.4f} at rank-th={best_th:.4f}")

# ===== 2) 全trainで再学習 → test予測 =====
lgb_rounds = int(np.mean(lgb_best_iters))
print(f"full-train LGB rounds = {lgb_rounds}")
all_idx = np.arange(len(y))
Xsvd_f, Xsvd_te, Xraw_f, Xraw_te = make_blocks(
    all_idx, Xs, Xs_test, train_tok, test_tok, None, None, out_is_test=True)
test_sc = fit_members(Xsvd_f, Xsvd_te, Xraw_f, Xraw_te, y, lgb_rounds)

test_ens = np.mean([to_rank(test_sc[m]) for m in MEMBERS], axis=0)

# 連続スコアを保存（後で任意の閾値を試せるように）
np.save("../features/test_ens_score.npy", test_ens)
pd.DataFrame({"企業ID": test_ids, "score": test_ens}).to_csv(
    "../features/test_ens_score.csv", index=False)

# 本命: CV最適閾値
final_pred = (test_ens >= best_th).astype(int)
pd.DataFrame({0: test_ids, 1: final_pred}).to_csv(
    "../submission/ensemble_submission.csv", index=False, header=False)
print(f"\nsaved ensemble_submission.csv 予測率={final_pred.mean():.3f} ({final_pred.sum()}件)")

# ---- 陽性率を振ったバリエーション（LBで最適陽性率を探る） ----
# 現行25%を挟んで上下に振る
variants = {
    "ensemble_rate21": 0.21,
    "ensemble_rate30": 0.30,
    "ensemble_rate35": 0.35,
}
print("\n=== 閾値バリエーション（LB probe用） ===")
for tag, rate in variants.items():
    th = np.quantile(test_ens, 1 - rate)   # 上位rate割を陽性
    pred = (test_ens >= th).astype(int)
    pd.DataFrame({0: test_ids, 1: pred}).to_csv(
        f"../submission/{tag}.csv", index=False, header=False)
    # 参考: このrateならOOFではF1いくつか
    oof_th = np.quantile(oof_ens, 1 - rate)
    oof_f1_at = f1_score(y, (oof_ens >= oof_th).astype(int))
    print(f"  {tag}: 予測率{pred.mean():.3f} ({pred.sum()}件)  [参考OOF F1@同率={oof_f1_at:.4f}]")
