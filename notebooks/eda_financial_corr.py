"""財務諸表 と 購入フラグ の相関分析（生科目 + 派生比率）"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 200)

train = pd.read_csv("../data/train.csv")
y = train["購入フラグ"]
print(f"全体購入率: {y.mean():.3f}  (購入{y.sum()} / 全{len(y)})")

# ---- 生の財務科目 ----
bs = ["資本金", "総資産", "流動資産", "固定資産", "負債", "短期借入金", "長期借入金", "純資産", "自己資本"]
pl = ["売上", "営業利益", "経常利益", "当期純利益"]
cf = ["営業CF", "減価償却費", "運転資本変動", "投資CF", "有形固定資産変動", "無形固定資産変動(ソフトウェア関連)"]
raw_cols = bs + pl + cf


def safe_div(a, b):
    return a / b.replace(0, np.nan)


# ---- 派生比率 ----
d = pd.DataFrame(index=train.index)
d["自己資本比率"] = safe_div(train["自己資本"], train["総資産"])
d["ROE"] = safe_div(train["当期純利益"], train["自己資本"])
d["ROA"] = safe_div(train["当期純利益"], train["総資産"])
d["売上高営業利益率"] = safe_div(train["営業利益"], train["売上"])
d["売上高経常利益率"] = safe_div(train["経常利益"], train["売上"])
d["流動資産比率"] = safe_div(train["流動資産"], train["総資産"])
d["負債比率"] = safe_div(train["負債"], train["純資産"])
d["有利子負債比率"] = safe_div(train["短期借入金"] + train["長期借入金"], train["総資産"])
d["一人当たり売上"] = safe_div(train["売上"], train["従業員数"])
d["一人当たり営業利益"] = safe_div(train["営業利益"], train["従業員数"])
d["減価償却費率"] = safe_div(train["減価償却費"], train["売上"])
d["無形固定資産変動率"] = safe_div(train["無形固定資産変動(ソフトウェア関連)"], train["総資産"])
d["有形固定資産変動率"] = safe_div(train["有形固定資産変動"], train["総資産"])
d["ソフト投資対有形比"] = safe_div(train["無形固定資産変動(ソフトウェア関連)"], train["有形固定資産変動"].abs() + 1)
d["フリーCF"] = train["営業CF"] + train["投資CF"]
ratio_cols = list(d.columns)

work = pd.concat([train[raw_cols + ["購入フラグ"]], d], axis=1)


def analyze(cols, title):
    rows = []
    for c in cols:
        sub = work[[c, "購入フラグ"]].dropna()
        if sub[c].nunique() < 3:
            continue
        r, p = stats.pointbiserialr(sub["購入フラグ"], sub[c])
        g1 = sub.loc[sub["購入フラグ"] == 1, c]
        g0 = sub.loc[sub["購入フラグ"] == 0, c]
        u, pmw = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        rows.append({
            "変数": c,
            "非購入_median": g0.median(),
            "購入_median": g1.median(),
            "r": r,
            "p(r)": p,
            "MW_p": pmw,
            "|r|": abs(r),
        })
    df = pd.DataFrame(rows).sort_values("|r|", ascending=False)
    print(f"\n{'='*80}\n### {title}（|point-biserial r| 降順）")
    print(df.round(4).to_string(index=False))
    return df


d1 = analyze(raw_cols, "生の財務科目")
d2 = analyze(ratio_cols, "派生比率")

print(f"\n{'='*80}\n### 全体トップ15（効き順）")
alldf = pd.concat([d1, d2]).sort_values("|r|", ascending=False)
print(alldf[["変数", "非購入_median", "購入_median", "r", "MW_p"]].head(15).round(4).to_string(index=False))
