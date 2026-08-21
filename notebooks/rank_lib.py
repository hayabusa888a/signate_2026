"""順位系特徴を分離し、fit集合(fold内 or 全train)から順位/zを計算できるようにする。
build_base: 行単位で安全な基本特徴（分布に依存しない）。
RankTransformer: 業界内z/順位・財務総合スコア・テキスト積極度順位 等を fit/apply。
"""
import re
import numpy as np
import pandas as pd

from feat_lib import (target_col, id_col, cat_cols, DX_KEYWORDS, POS_WORDS,
                      NEG_WORDS, dept_pat, safe_div, count_words)

# 分布依存の順位/zに使う元カラム
Z_COLS = ["売上高経常利益率", "ROA", "有利子負債比率", "log従業員数", "姿勢差", "自己資本比率"]
FIN_RANK_COLS = ["売上高経常利益率", "ROA", "自己資本比率", "有利子負債比率"]


def build_base(df):
    """分布に依存しない行単位の基本特徴のみ（rank/z系は含まない）。"""
    num_cat_base = [c for c in df.columns
                    if c not in [target_col, id_col] + ["企業概要", "組織図", "今後のDX展望", "企業名"]]
    f = df[num_cat_base].copy()

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

    f["log従業員数"] = np.log1p(df["従業員数"])
    f["log総資産"] = np.log1p(df["総資産"].clip(lower=0))
    f["log売上"] = np.log1p(df["売上"].clip(lower=0))

    f["満足度軸"] = df[["アンケート２", "アンケート８"]].mean(axis=1)
    f["戦略整備軸"] = df[["アンケート１", "アンケート５"]].mean(axis=1)
    f["逆風スコア"] = df["アンケート７"].fillna(3) + df["アンケート１０"] + df["アンケート４"]
    f["抵抗高フラグ"] = (df["アンケート４"] >= 4).astype(int)
    f["連携高フラグ"] = (df["アンケート１０"] >= 4).astype(int)
    f["ツール未導入フラグ"] = df["アンケート７"].isna().astype(int)
    f["既存ツール満足度"] = df["アンケート７"]
    f["導入済み不満フラグ"] = ((df["アンケート６"] == 1) & (df["アンケート７"] <= 2)).astype(int)
    f["導入済みフラグ"] = (df["アンケート６"] == 1).astype(int)

    concat = df["企業概要"].fillna("") + " " + df["今後のDX展望"].fillna("")
    for kw in DX_KEYWORDS:
        f[f"kw_{kw}"] = concat.str.contains(re.escape(kw)).astype(int)
    f["姿勢_積極"] = count_words(concat, POS_WORDS)
    f["姿勢_抑制"] = count_words(concat, NEG_WORDS)
    f["姿勢差"] = f["姿勢_積極"] - f["姿勢_抑制"]
    f["DX部署フラグ"] = df["組織図"].fillna("").apply(lambda s: int(bool(dept_pat.search(s))))

    f["満足度高フラグ"] = (df["アンケート７"] >= 4).astype(int)
    f["導入済み満足フラグ"] = ((df["アンケート６"] == 1) & (df["アンケート７"] >= 4)).astype(int)

    # 業界キーはrank計算で使うため保持
    f["_業界"] = df["業界"].astype(str).values
    return f


def _pct(sorted_arr, x):
    """train分布(sorted_arr)におけるxの percentile (0-1)。"""
    if len(sorted_arr) == 0:
        return np.full(len(x), 0.5)
    return np.searchsorted(sorted_arr, x, side="right") / len(sorted_arr)


class RankTransformer:
    """fit集合の分布から 業界内z/順位・財務総合スコア・テキスト積極度順位 を計算。"""

    def fit(self, f_fit):
        self.ind_stats = {}   # col -> {industry -> (mean,std, sorted_vals)}
        self.global_sorted = {}
        ind = f_fit["_業界"].values
        for col in Z_COLS:
            vals = f_fit[col].values.astype(float)
            self.global_sorted[col] = np.sort(vals[~np.isnan(vals)])
            d = {}
            for g in np.unique(ind):
                gv = vals[ind == g]
                gv = gv[~np.isnan(gv)]
                if len(gv) >= 5:
                    d[g] = (gv.mean(), gv.std() if gv.std() > 0 else np.nan, np.sort(gv))
            self.ind_stats[col] = d
        for col in FIN_RANK_COLS + ["姿勢差"]:
            vals = f_fit[col].values.astype(float)
            self.global_sorted.setdefault(col, np.sort(vals[~np.isnan(vals)]))
        return self

    def transform(self, f):
        f = f.copy()
        ind = f["_業界"].values
        for col in Z_COLS:
            vals = f[col].values.astype(float)
            z = np.zeros(len(f)); rnk = np.full(len(f), 0.5)
            gs = self.global_sorted[col]
            gmean = gs.mean() if len(gs) else 0.0
            gstd = gs.std() if len(gs) and gs.std() > 0 else 1.0
            for i in range(len(f)):
                g = ind[i]; v = vals[i]
                st = self.ind_stats[col].get(g)
                if st is not None and not np.isnan(st[1]):
                    z[i] = (v - st[0]) / st[1] if not np.isnan(v) else 0.0
                    rnk[i] = _pct(st[2], np.array([v]))[0] if not np.isnan(v) else 0.5
                else:  # 未知/小グループは全体分布でフォールバック
                    z[i] = (v - gmean) / gstd if not np.isnan(v) else 0.0
                    rnk[i] = _pct(gs, np.array([v]))[0] if not np.isnan(v) else 0.5
            f[f"{col}_業界内z"] = z
            f[f"{col}_業界内順位"] = rnk

        prof = _pct(self.global_sorted["売上高経常利益率"], f["売上高経常利益率"].values.astype(float))
        roa = _pct(self.global_sorted["ROA"], f["ROA"].values.astype(float))
        eq = _pct(self.global_sorted["自己資本比率"], f["自己資本比率"].values.astype(float))
        debt = _pct(self.global_sorted["有利子負債比率"], f["有利子負債比率"].values.astype(float))
        f["財務総合スコア"] = (prof + roa + eq + (1 - debt)) / 4

        posture = _pct(self.global_sorted["姿勢差"], f["姿勢差"].values.astype(float))
        f["テキスト積極度順位"] = posture
        f["テキスト財務乖離"] = posture - f["財務総合スコア"]
        f["満足_財務優良"] = f["満足度高フラグ"] * f["財務総合スコア"]

        return f.drop(columns=["_業界"])
