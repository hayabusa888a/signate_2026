"""特徴量追加フェーズ v2 — EDA知見を反映。valid(OOF) F1 のみ確認（提出学習なし）。

label非依存の特徴は全体で計算。target encoding と TF-IDF+SVD は fold内fit でリーク防止。
"""
import re
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from janome.tokenizer import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
target_col = "購入フラグ"
id_col = "企業ID"
cat_cols = ["業界", "上場種別", "特徴"]
te_cols = ["業界", "上場種別", "特徴"]           # target encoding 対象
TFIDF_TEXT_COLS = ["企業概要", "今後のDX展望", "組織図"]
N_SVD = 40

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
y = train[target_col].values


def safe_div(a, b):
    return a / b.replace(0, np.nan)


# EDAで有効だったキーワードに限定
DX_KEYWORDS = ["生成AI", "AI", "IoT", "クラウド", "リスキリング", "デジタル", "人材育成"]
# 姿勢語彙（積極 vs 抑制）
POS_WORDS = ["圧倒的", "力強", "一段と", "強固", "昇華", "加速", "積極", "推進", "変革",
             "挑戦", "拡大", "成長", "抜本", "全社", "野心", "先進", "最先端", "テクノロジー", "ナレッジ", "ダッシュボード"]
NEG_WORDS = ["抑える", "小さく", "極力", "スモール", "短時間", "低い", "マニュアル", "紙",
             "慎重", "段階的", "最小限", "コスト削減", "見送", "縮小", "様子見"]
DX_DEPT = ["DX推進", "DX戦略", "デジタル推進", "IT戦略", "情報システム", "DX室", "デジタル戦略"]
dept_pat = re.compile("|".join(map(re.escape, DX_DEPT)))


def count_words(series, words):
    pat = re.compile("|".join(map(re.escape, words)))
    return series.fillna("").apply(lambda s: len(pat.findall(s)))


def build_static(df):
    """label非依存の特徴（財務比率・アンケート再設計・テキスト内容系）。"""
    num_cat_base = [c for c in df.columns
                    if c not in [target_col, id_col] + ["企業概要", "組織図", "今後のDX展望", "企業名"]]
    f = df[num_cat_base].copy()

    # --- 財務比率（EDAで効いた率系を中心に） ---
    f["自己資本比率"] = safe_div(df["自己資本"], df["総資産"])
    f["ROE"] = safe_div(df["当期純利益"], df["自己資本"])
    f["ROA"] = safe_div(df["当期純利益"], df["総資産"])
    f["売上高営業利益率"] = safe_div(df["営業利益"], df["売上"])
    f["売上高経常利益率"] = safe_div(df["経常利益"], df["売上"])
    f["流動資産比率"] = safe_div(df["流動資産"], df["総資産"])
    f["負債比率"] = safe_div(df["負債"], df["純資産"])
    有利子負債 = df["短期借入金"] + df["長期借入金"]
    f["有利子負債比率"] = safe_div(有利子負債, df["総資産"])
    f["一人当たり売上"] = safe_div(df["売上"], df["従業員数"])
    f["一人当たり営業利益"] = safe_div(df["営業利益"], df["従業員数"])
    f["事業所あたり従業員数"] = safe_div(df["従業員数"], df["事業所数"])
    f["フリーCF"] = df["営業CF"] + df["投資CF"]
    f["減価償却費率"] = safe_div(df["減価償却費"], df["売上"])
    f["無形固定資産変動率"] = safe_div(df["無形固定資産変動(ソフトウェア関連)"], df["総資産"])
    f["有形固定資産変動率"] = safe_div(df["有形固定資産変動"], df["総資産"])
    f["ソフト投資対有形比"] = safe_div(df["無形固定資産変動(ソフトウェア関連)"], df["有形固定資産変動"].abs() + 1)
    f["投資CF対売上"] = safe_div(df["投資CF"], df["売上"])

    # --- 規模（log変換） ---
    f["log従業員数"] = np.log1p(df["従業員数"])
    f["log総資産"] = np.log1p(df["総資産"].clip(lower=0))
    f["log売上"] = np.log1p(df["売上"].clip(lower=0))

    # --- アンケート再設計（EDA: 単純合算は無効、符号を揃える/軸を分ける） ---
    # 満足度・現状肯定 軸（高いほど買わない傾向）
    f["満足度軸"] = df[["アンケート２", "アンケート８"]].mean(axis=1)
    # 戦略・整備 軸（高いほど買う傾向）
    f["戦略整備軸"] = df[["アンケート１", "アンケート５"]].mean(axis=1)
    # ネガ要因（逆相関の強い項目を符号反転して合成）
    f["逆風スコア"] = df["アンケート７"].fillna(3) + df["アンケート１０"] + df["アンケート４"]
    # 非線形項目のビン化
    f["抵抗高フラグ"] = (df["アンケート４"] >= 4).astype(int)
    f["連携高フラグ"] = (df["アンケート１０"] >= 4).astype(int)
    # アンケート7（既存ツール満足度）: 欠損=未導入
    f["ツール未導入フラグ"] = df["アンケート７"].isna().astype(int)
    f["既存ツール満足度"] = df["アンケート７"]  # 欠損はそのままLGBに委譲
    # 最も熱いセグメント: 導入済み × 満足度低(<=2)
    f["導入済み不満フラグ"] = ((df["アンケート６"] == 1) & (df["アンケート７"] <= 2)).astype(int)
    f["導入済みフラグ"] = (df["アンケート６"] == 1).astype(int)

    # --- テキスト内容系（長さは除外＝EDAで無効） ---
    concat = df["企業概要"].fillna("") + " " + df["今後のDX展望"].fillna("")
    for kw in DX_KEYWORDS:
        f[f"kw_{kw}"] = concat.str.contains(re.escape(kw)).astype(int)
    f["姿勢_積極"] = count_words(concat, POS_WORDS)
    f["姿勢_抑制"] = count_words(concat, NEG_WORDS)
    f["姿勢差"] = f["姿勢_積極"] - f["姿勢_抑制"]
    f["DX部署フラグ"] = df["組織図"].fillna("").apply(lambda s: int(bool(dept_pat.search(s))))

    # ================= v3: エラー分析由来の特徴（すべてlabel非依存） =================
    ind = df["業界"].astype(str)

    # --- (1) 業界内相対位置：購入率中間の業界で絶対値の交絡を解く ---
    for col in ["売上高経常利益率", "ROA", "有利子負債比率", "log従業員数", "姿勢差", "自己資本比率"]:
        grp = f[col].groupby(ind)
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        f[f"{col}_業界内z"] = ((f[col] - mean) / std).fillna(0)
        f[f"{col}_業界内順位"] = f[col].groupby(ind).rank(pct=True)

    # --- (2) 財務総合スコア（全体順位ベース、外れ値に頑健）---
    prof_rank = f["売上高経常利益率"].rank(pct=True)
    roa_rank = f["ROA"].rank(pct=True)
    equity_rank = f["自己資本比率"].rank(pct=True)
    debt_rank = f["有利子負債比率"].rank(pct=True)  # 高いほど不健全
    f["財務総合スコア"] = (prof_rank + roa_rank + equity_rank + (1 - debt_rank)) / 4

    # --- (3) テキスト積極度と財務力の乖離：FN(財務地味だがテキスト前向き)の温床を可視化 ---
    posture_rank = f["姿勢差"].rank(pct=True)
    f["テキスト積極度順位"] = posture_rank
    f["テキスト財務乖離"] = posture_rank - f["財務総合スコア"]  # 高い=財務の割にテキストが前向き

    # --- (4) FP抑制：満足していて財務も良い企業は買わない ---
    f["満足度高フラグ"] = (df["アンケート７"] >= 4).astype(int)  # 導入済みで満足
    f["導入済み満足フラグ"] = ((df["アンケート６"] == 1) & (df["アンケート７"] >= 4)).astype(int)
    # 満足×財務優良（買わない方向のシグナルを明示）
    f["満足_財務優良"] = f["満足度高フラグ"] * f["財務総合スコア"]

    return f


X = build_static(train)
X_test = build_static(test)

# カテゴリ整列
for c in cat_cols:
    X[c] = X[c].fillna("missing").astype("category")
    X_test[c] = X_test[c].fillna("missing").astype("category")
    all_cats = pd.concat([X[c], X_test[c]]).unique()
    X[c] = X[c].cat.set_categories(all_cats)
    X_test[c] = X_test[c].cat.set_categories(all_cats)

# ---- テキスト事前トークン化（janome, 1回） ----
tokenizer = Tokenizer()
def tok(t):
    if not isinstance(t, str) or t == "":
        return ""
    return " ".join(tokenizer.tokenize(t, wakati=True))

print("tokenizing...", flush=True)
train_tok = {c: train[c].fillna("").map(tok) for c in TFIDF_TEXT_COLS}
test_tok = {c: test[c].fillna("").map(tok) for c in TFIDF_TEXT_COLS}


def target_encode(tr_series, y_tr, va_series, test_series, global_mean, smoothing=10):
    """fold内fitのスムージング付き target encoding。"""
    df = pd.DataFrame({"k": tr_series.values, "y": y_tr})
    agg = df.groupby("k")["y"].agg(["mean", "count"])
    enc = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
    return (
        va_series.map(enc).fillna(global_mean).values,
        test_series.map(enc).fillna(global_mean).values,
        tr_series.map(enc).fillna(global_mean).values,
    )


def build_tfidf_svd(tr_texts, va_texts, test_texts, col, seed):
    vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2), max_features=20000)
    tr_mat = vec.fit_transform(tr_texts)
    n_comp = min(N_SVD, tr_mat.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    tr_svd = svd.fit_transform(tr_mat)
    va_svd = svd.transform(vec.transform(va_texts))
    test_svd = svd.transform(vec.transform(test_texts))
    cols = [f"{col}_svd{i}" for i in range(n_comp)]
    return (pd.DataFrame(tr_svd, columns=cols),
            pd.DataFrame(va_svd, columns=cols),
            pd.DataFrame(test_svd, columns=cols))


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_pred = np.zeros(len(X))

params = {
    "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
    "num_leaves": 15, "min_child_samples": 10, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1,
    "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),
    "verbosity": -1, "seed": RANDOM_STATE,
}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    X_va = X.iloc[va_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    gmean = y_tr.mean()

    # target encoding（fold内fit）
    for c in te_cols:
        va_te, _, tr_te = target_encode(
            train[c].iloc[tr_idx].astype(str), y_tr,
            train[c].iloc[va_idx].astype(str), test[c].astype(str), gmean)
        X_tr[f"te_{c}"] = tr_te
        X_va[f"te_{c}"] = va_te

    # TF-IDF + SVD（fold内fit）
    tr_parts, va_parts = [X_tr], [X_va]
    for c in TFIDF_TEXT_COLS:
        tr_s, va_s, _ = build_tfidf_svd(
            train_tok[c].iloc[tr_idx].tolist(),
            train_tok[c].iloc[va_idx].tolist(),
            test_tok[c].tolist(), c, RANDOM_STATE + fold)
        tr_parts.append(tr_s)
        va_parts.append(va_s)
    X_tr_f = pd.concat(tr_parts, axis=1)
    X_va_f = pd.concat(va_parts, axis=1)

    dtrain = lgb.Dataset(X_tr_f, label=y_tr, categorical_feature=cat_cols)
    dvalid = lgb.Dataset(X_va_f, label=y[va_idx], categorical_feature=cat_cols, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(50, verbose=False)])

    oof_pred[va_idx] = model.predict(X_va_f, num_iteration=model.best_iteration)
    print(f"fold {fold}: best_iter={model.best_iteration}, "
          f"logloss={model.best_score['valid_0']['binary_logloss']:.4f}", flush=True)

best_th, best_f1 = 0.5, -1
for th in np.arange(0.05, 0.95, 0.01):
    f1 = f1_score(y, (oof_pred >= th).astype(int))
    if f1 > best_f1:
        best_f1, best_th = f1, th

print(f"\n=== v3 OOF best F1={best_f1:.4f} at threshold={best_th:.2f} ===")
print("(baseline 0.6379 / feature_eng 0.6635 / tfidf 0.7168 / v2 0.7442)")

# エラー分析用に OOF 予測を保存
np.save("../features/oof_pred_v3.npy", oof_pred)
pd.DataFrame({"企業ID": train[id_col], "oof_pred": oof_pred, "y": y,
              "threshold": best_th}).to_csv("../features/oof_pred_v3.csv", index=False)
print("saved features/oof_pred_v3.csv (for error analysis)")

# 重要度（最終fold）
imp = pd.Series(model.feature_importance(importance_type="gain"),
                index=X_tr_f.columns).sort_values(ascending=False)
imp.to_csv("../features/importance_v3.csv", encoding="utf-8-sig")
print("\nTop 30 importances:")
print(imp.head(30).to_string())
# v3で追加した特徴の順位を確認
v3_feats = ["財務総合スコア", "テキスト財務乖離", "テキスト積極度順位", "満足_財務優良",
            "満足度高フラグ", "導入済み満足フラグ", "売上高経常利益率_業界内z",
            "ROA_業界内順位", "有利子負債比率_業界内z", "log従業員数_業界内z", "姿勢差_業界内z"]
print("\n[v3追加特徴の重要度]")
for feat in v3_feats:
    if feat in imp.index:
        print(f"  {feat}: {imp[feat]:.1f} (順位 {list(imp.index).index(feat)+1})")
