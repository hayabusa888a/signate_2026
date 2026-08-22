"""LR単体(L2, C=0.1, SVDテキスト)の提出ファイルを作成。
単純モデルの汎化を確認する狙い。OOFで閾値決定→全train再学習→test予測。
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
test_ids = test[id_col].values

Xs = build_static(train); Xs_test = build_static(test)
num_cols = [c for c in Xs.columns if c not in cat_cols]

tokenizer = Tokenizer()
def tok(t):
    if not isinstance(t, str) or t == "":
        return ""
    return " ".join(tokenizer.tokenize(t, wakati=True))
print("tokenizing...", flush=True)
train_tok = {c: train[c].fillna("").map(tok) for c in TFIDF_TEXT_COLS}
test_tok = {c: test[c].fillna("").map(tok) for c in TFIDF_TEXT_COLS}


def build_X(Xs_fit, Xs_out, fit_idx, tok_fit, tok_out, out_is_test):
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
    svd_f, svd_o = [], []
    for c in TFIDF_TEXT_COLS:
        vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2), max_features=20000)
        fm = vec.fit_transform(tok_fit[c]); om = vec.transform(tok_out[c])
        n_comp = min(40, fm.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        ft = svd.fit_transform(fm); ot = svd.transform(om)
        s = StandardScaler().fit(ft); svd_f.append(s.transform(ft)); svd_o.append(s.transform(ot))
    Xf = sparse.hstack([sparse.csr_matrix(num_f), cat_f, sparse.csr_matrix(te_f),
                        sparse.csr_matrix(np.hstack(svd_f))]).tocsr()
    Xo = sparse.hstack([sparse.csr_matrix(num_o), cat_o, sparse.csr_matrix(te_o),
                        sparse.csr_matrix(np.hstack(svd_o))]).tocsr()
    return Xf, Xo


def make_lr():
    return LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, solver="liblinear")


# ---- OOF で閾値決定 ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(y))
for tr_idx, va_idx in skf.split(Xs, y):
    Xtr, Xva = build_X(Xs.iloc[tr_idx], Xs.iloc[va_idx], tr_idx,
                       {c: train_tok[c].iloc[tr_idx] for c in TFIDF_TEXT_COLS},
                       {c: train_tok[c].iloc[va_idx] for c in TFIDF_TEXT_COLS}, out_is_test=False)
    oof[va_idx] = make_lr().fit(Xtr, y[tr_idx]).predict_proba(Xva)[:, 1]

best_th, best_f1 = 0.5, -1
for th in np.arange(0.05, 0.95, 0.01):
    f = f1_score(y, (oof >= th).astype(int))
    if f > best_f1:
        best_f1, best_th = f, th
print(f"LR単体 OOF F1={best_f1:.4f} at th={best_th:.2f}")

# ---- 全train再学習 → test予測 ----
all_idx = np.arange(len(y))
Xf, Xte = build_X(Xs, Xs_test, all_idx, train_tok, test_tok, out_is_test=True)
test_prob = make_lr().fit(Xf, y).predict_proba(Xte)[:, 1]
final_pred = (test_prob >= best_th).astype(int)

pd.DataFrame({0: test_ids, 1: final_pred}).to_csv(
    "../submission/lr_submission.csv", index=False, header=False)
print(f"saved submission/lr_submission.csv 予測率={final_pred.mean():.3f} ({final_pred.sum()}件)")
np.save("../features/test_prob_lr.npy", test_prob)
