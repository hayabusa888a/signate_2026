"""上場種別(PR/ST/GR)の深掘りEDA。"""
import re
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 240)

train = pd.read_csv("../data/train.csv")
y = train["購入フラグ"].values
print(f"全体購入率={y.mean():.3f}  n={len(train)}")


def sd(a, b): return a / b.replace(0, np.nan)
train["経常利益率"] = sd(train["経常利益"], train["売上"])
train["ROA"] = sd(train["当期純利益"], train["総資産"])
train["有利子負債比率"] = sd(train["短期借入金"] + train["長期借入金"], train["総資産"])
train["自己資本比率"] = sd(train["自己資本"], train["総資産"])
train["BtoCフラグ"] = train["特徴"].fillna("").str.contains("BtoC").astype(int)
POS = ["圧倒的", "力強", "一段と", "強固", "加速", "積極", "推進", "変革", "挑戦", "拡大", "成長", "先進", "最先端"]
NEG = ["抑える", "小さく", "極力", "スモール", "短時間", "低い", "マニュアル", "慎重", "段階的", "最小限", "見送", "縮小"]
def cnt(s, ws):
    pat = re.compile("|".join(map(re.escape, ws)))
    return s.fillna("").apply(lambda t: len(pat.findall(t)))
concat = train["企業概要"].fillna("") + " " + train["今後のDX展望"].fillna("")
train["姿勢差"] = cnt(concat, POS) - cnt(concat, NEG)

# ===== 上場種別プロファイル =====
g = train.groupby("上場種別")
prof = pd.DataFrame({
    "n": g.size(),
    "購入率": g["購入フラグ"].mean(),
    "従業員数_中央": g["従業員数"].median(),
    "総資産_中央": g["総資産"].median(),
    "経常利益率_中央": g["経常利益率"].median(),
    "有利子負債比率_中央": g["有利子負債比率"].median(),
    "自己資本比率_中央": g["自己資本比率"].median(),
    "BtoC比率": g["BtoCフラグ"].mean(),
    "姿勢差_中央": g["姿勢差"].median(),
    "Q7満足_平均": g["アンケート７"].mean(),
    "Q1戦略_平均": g["アンケート１"].mean(),
}).sort_values("購入率", ascending=False)
print("\n=== 上場種別プロファイル ===")
print(prof.round(3).to_string())

# ===== カイ二乗 =====
ct = pd.crosstab(train["上場種別"], y)
chi2, p, dof, _ = stats.chi2_contingency(ct)
n = ct.sum().sum()
print(f"\n上場種別×購入 カイ二乗: chi2={chi2:.1f} p={p:.2e} CramersV={np.sqrt(chi2/(n*(min(ct.shape)-1))):.3f}")

# ===== 交絡チェック: 業界を固定しても種別差は残るか =====
print("\n=== 交絡チェック: 主要業界内での 種別別購入率 ===")
for ind in ["機械", "IT", "金融", "建設・工事", "自動車・乗り物"]:
    sub = train[train["業界"] == ind]
    t = sub.groupby("上場種別")["購入フラグ"].agg(["count", "mean"])
    t = t[t["count"] >= 3]
    print(f"\n[{ind}] n={len(sub)}")
    print(t.round(3).to_string())

# ===== 種別内で購入を分ける特徴 =====
print("\n=== 種別ごと 購入 vs 非購入 中央値 ===")
FEATS = ["従業員数", "経常利益率", "有利子負債比率", "姿勢差", "アンケート１", "アンケート７", "アンケート１０"]
for k in ["PR", "ST", "GR"]:
    sub = train[train["上場種別"] == k]
    ys = sub["購入フラグ"].values
    print(f"\n■ {k} (n={len(sub)}, 購入{int(ys.sum())}, 購入率{ys.mean():.3f})")
    rows = []
    for c in FEATS:
        b = sub.loc[sub["購入フラグ"] == 1, c].median()
        nn = sub.loc[sub["購入フラグ"] == 0, c].median()
        rows.append({"特徴": c, "購入_中央": round(b, 3), "非購入_中央": round(nn, 3)})
    print(pd.DataFrame(rows).to_string(index=False))
