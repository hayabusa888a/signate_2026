"""共通の特徴生成ライブラリ（import時に副作用なし）。v3のbuild_staticを再利用可能にしたもの。"""
import re
import numpy as np
import pandas as pd

RANDOM_STATE = 42
target_col = "購入フラグ"
id_col = "企業ID"
cat_cols = ["業界", "上場種別", "特徴"]
te_cols = ["業界", "上場種別", "特徴"]
TFIDF_TEXT_COLS = ["企業概要", "今後のDX展望", "組織図"]

DX_KEYWORDS = ["生成AI", "AI", "IoT", "クラウド", "リスキリング", "デジタル", "人材育成"]
POS_WORDS = ["圧倒的", "力強", "一段と", "強固", "昇華", "加速", "積極", "推進", "変革",
             "挑戦", "拡大", "成長", "抜本", "全社", "野心", "先進", "最先端", "テクノロジー", "ナレッジ", "ダッシュボード"]
NEG_WORDS = ["抑える", "小さく", "極力", "スモール", "短時間", "低い", "マニュアル", "紙",
             "慎重", "段階的", "最小限", "コスト削減", "見送", "縮小", "様子見"]
DX_DEPT = ["DX推進", "DX戦略", "デジタル推進", "IT戦略", "情報システム", "DX室", "デジタル戦略"]
dept_pat = re.compile("|".join(map(re.escape, DX_DEPT)))


def safe_div(a, b):
    return a / b.replace(0, np.nan)


def count_words(series, words):
    pat = re.compile("|".join(map(re.escape, words)))
    return series.fillna("").apply(lambda s: len(pat.findall(s)))


def build_static(df):
    """label非依存の特徴（財務比率・アンケート再設計・テキスト内容系 + v3のエラー分析特徴）。"""
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

    ind = df["業界"].astype(str)
    for col in ["売上高経常利益率", "ROA", "有利子負債比率", "log従業員数", "姿勢差", "自己資本比率"]:
        grp = f[col].groupby(ind)
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        f[f"{col}_業界内z"] = ((f[col] - mean) / std).fillna(0)
        f[f"{col}_業界内順位"] = f[col].groupby(ind).rank(pct=True)

    prof_rank = f["売上高経常利益率"].rank(pct=True)
    roa_rank = f["ROA"].rank(pct=True)
    equity_rank = f["自己資本比率"].rank(pct=True)
    debt_rank = f["有利子負債比率"].rank(pct=True)
    f["財務総合スコア"] = (prof_rank + roa_rank + equity_rank + (1 - debt_rank)) / 4

    posture_rank = f["姿勢差"].rank(pct=True)
    f["テキスト積極度順位"] = posture_rank
    f["テキスト財務乖離"] = posture_rank - f["財務総合スコア"]

    f["満足度高フラグ"] = (df["アンケート７"] >= 4).astype(int)
    f["導入済み満足フラグ"] = ((df["アンケート６"] == 1) & (df["アンケート７"] >= 4)).astype(int)
    f["満足_財務優良"] = f["満足度高フラグ"] * f["財務総合スコア"]

    return f


def target_encode(tr_series, y_tr, va_series, test_series, global_mean, smoothing=10):
    """fold内fitのスムージング付き target encoding。"""
    d = pd.DataFrame({"k": tr_series.values, "y": y_tr})
    agg = d.groupby("k")["y"].agg(["mean", "count"])
    enc = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
    return (va_series.map(enc).fillna(global_mean).values,
            test_series.map(enc).fillna(global_mean).values,
            tr_series.map(enc).fillna(global_mean).values)
