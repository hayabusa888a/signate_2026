"""業界の深掘りEDA: 購入率の違いが何に由来するかをプロファイル。"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 240)

train = pd.read_csv("../data/train.csv")
y = train["購入フラグ"]
base = y.mean()
print(f"全体購入率={base:.3f}  n={len(train)}")


def sd(a, b): return a / b.replace(0, np.nan)
train["経常利益率"] = sd(train["経常利益"], train["売上"])
train["ROA"] = sd(train["当期純利益"], train["総資産"])
train["有利子負債比率"] = sd(train["短期借入金"] + train["長期借入金"], train["総資産"])
train["自己資本比率"] = sd(train["自己資本"], train["総資産"])
train["BtoCフラグ"] = train["特徴"].fillna("").str.contains("BtoC").astype(int)
train["STフラグ"] = (train["上場種別"] == "ST").astype(int)

# ===== 業界プロファイル（n>=10のみ） =====
g = train.groupby("業界")
prof = pd.DataFrame({
    "n": g.size(),
    "購入率": g["購入フラグ"].mean(),
    "従業員数_中央": g["従業員数"].median(),
    "経常利益率_中央": g["経常利益率"].median(),
    "ROA_中央": g["ROA"].median(),
    "有利子負債比率_中央": g["有利子負債比率"].median(),
    "BtoC比率": g["BtoCフラグ"].mean(),
    "ST比率": g["STフラグ"].mean(),
    "Q7満足_平均": g["アンケート７"].mean(),
    "Q10連携_平均": g["アンケート１０"].mean(),
    "Q1戦略_平均": g["アンケート１"].mean(),
})
prof = prof[prof["n"] >= 10].sort_values("購入率", ascending=False)
print("\n=== 業界プロファイル（購入率降順, n>=10）===")
print(prof.round(3).to_string())

# ===== 高/中/低 購入率グループの特徴差 =====
print("\n=== 購入率 高(>=0.35) / 中(0.15-0.35) / 低(<0.15) グループの平均像 ===")
def tier(r):
    return "高" if r >= 0.35 else ("低" if r < 0.15 else "中")
prof["tier"] = prof["購入率"].apply(tier)
train2 = train.merge(prof[["tier"]], left_on="業界", right_index=True, how="inner")
tg = train2.groupby("tier")
summ = pd.DataFrame({
    "業界数": prof.groupby("tier").size(),
    "企業数": tg.size(),
    "購入率": tg["購入フラグ"].mean(),
    "従業員数_中央": tg["従業員数"].median(),
    "経常利益率_中央": tg["経常利益率"].median(),
    "BtoC比率": tg["BtoCフラグ"].mean(),
    "ST比率": tg["STフラグ"].mean(),
    "Q7満足_平均": tg["アンケート７"].mean(),
    "Q10連携_平均": tg["アンケート１０"].mean(),
}).reindex(["高", "中", "低"])
print(summ.round(3).to_string())

# ===== 業界効果は交絡か? 上場種別・特徴を固定した部分集団で購入率 =====
print("\n=== 交絡チェック: BtoB企業だけに絞った業界購入率（BtoC構成の影響を除く）===")
btob = train[train["特徴"] == "BtoB"]
gg = btob.groupby("業界")["購入フラグ"].agg(["count", "mean"])
print(gg[gg["count"] >= 10].sort_values("mean", ascending=False).round(3).to_string())

# ===== カイ二乗（業界 × 購入） =====
ct = pd.crosstab(train["業界"], y)
chi2, p, dof, _ = stats.chi2_contingency(ct)
n = ct.sum().sum()
print(f"\n業界×購入 カイ二乗: chi2={chi2:.1f} p={p:.2e} CramersV={np.sqrt(chi2/(n*(min(ct.shape)-1))):.3f}")
