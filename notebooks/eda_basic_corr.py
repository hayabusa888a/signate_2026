"""企業基本情報 と 購入フラグ の相関分析"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 200)

train = pd.read_csv("../data/train.csv")
y = train["購入フラグ"]
base_rate = y.mean()

print(f"全体購入率: {base_rate:.3f}  (購入{y.sum()} / 全{len(y)})")
print("=" * 70)

cat_cols = ["業界", "上場種別", "特徴"]
num_cols = ["従業員数", "事業所数", "工場数", "店舗数"]

out = []

# ========== カテゴリ変数 ==========
for c in cat_cols:
    print(f"\n### {c} — 購入率（件数>=10のみ表示、カイ二乗検定）")
    tab = train.groupby(c)["購入フラグ"].agg(["count", "mean"]).rename(columns={"mean": "購入率"})
    tab = tab.sort_values("購入率", ascending=False)
    print(tab[tab["count"] >= 10].round(3).to_string())

    ct = pd.crosstab(train[c], y)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    # Cramér's V
    n = ct.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    print(f"  chi2={chi2:.2f}, p={p:.4f}, CramersV={cramers_v:.3f}")
    out.append((c, "cat", cramers_v, p))

# ========== 数値変数 ==========
print("\n" + "=" * 70)
for c in num_cols:
    g1 = train.loc[y == 1, c].dropna()
    g0 = train.loc[y == 0, c].dropna()
    # point-biserial（欠損を除いた行で）
    sub = train[[c, "購入フラグ"]].dropna()
    r, p = stats.pointbiserialr(sub["購入フラグ"], sub[c])
    # Mann-Whitney（分布の差、外れ値に頑健）
    u, pmw = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    print(f"\n### {c}")
    print(f"  非購入 median={g0.median():.1f} mean={g0.mean():.1f} (n={len(g0)})")
    print(f"  購入   median={g1.median():.1f} mean={g1.mean():.1f} (n={len(g1)})")
    print(f"  point-biserial r={r:+.3f} (p={p:.4f}) / Mann-Whitney p={pmw:.4f}")
    out.append((c, "num", abs(r), p))

# ========== まとめ（効き順） ==========
print("\n" + "=" * 70)
print("### 効き順まとめ（カテゴリ=CramersV, 数値=|point-biserial r|）")
summ = pd.DataFrame(out, columns=["変数", "型", "効果量", "p値"]).sort_values("効果量", ascending=False)
print(summ.round(4).to_string(index=False))
