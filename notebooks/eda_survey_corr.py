"""アンケート項目 と 購入フラグ の相関分析"""
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

# アンケートの意味
LABELS = {
    "アンケート１": "DX戦略の明確さ(1-5)",
    "アンケート２": "DX化満足度(1-5)",
    "アンケート３": "最新技術導入状況(1-5)",
    "アンケート４": "DX変革への抵抗感(1-5, 高いほど抵抗)",
    "アンケート５": "セキュリティ整備状況(1-5)",
    "アンケート６": "業務改善ツール導入済み(1=はい,2=いいえ)",
    "アンケート７": "既存ツール満足度(1-5, 6=はい時のみ)",
    "アンケート８": "DX成果の実感(1-5)",
    "アンケート９": "技術イベント参加率(1-5)",
    "アンケート１０": "外部パートナー連携(1-5)",
    "アンケート１１": "情報収集の度合い(1-5)",
}
survey_cols = list(LABELS.keys())

# ---- 各項目: 購入率テーブル + point-biserial ----
rows = []
for c in survey_cols:
    print(f"\n{'='*70}\n### {c}: {LABELS[c]}")
    tab = train.groupby(c)["購入フラグ"].agg(["count", "mean"]).rename(columns={"mean": "購入率"})
    print(tab.round(3).to_string())
    if train[c].isna().any():
        na_rate = train.loc[train[c].isna(), "購入フラグ"].mean()
        print(f"  [欠損 {train[c].isna().sum()}件] 購入率={na_rate:.3f}")

    sub = train[[c, "購入フラグ"]].dropna()
    r, p = stats.pointbiserialr(sub["購入フラグ"], sub[c])
    # spearman（順序尺度として）
    rho, prho = stats.spearmanr(sub[c], sub["購入フラグ"])
    rows.append({"変数": c, "意味": LABELS[c], "r": r, "spearman": rho, "p": p, "|r|": abs(r)})

# ---- アンケート7欠損フラグ（=アンケート6でいいえ）自体の効果 ----
q7na = train["アンケート７"].isna().astype(int)
r7, p7 = stats.pointbiserialr(y, q7na)
print(f"\n{'='*70}\n### アンケート7欠損フラグ(=ツール未導入) の効果: r={r7:+.3f} p={p7:.4f}")

# ---- DX前向きスコア合成（4は逆転） ----
pos = ["アンケート１", "アンケート２", "アンケート３", "アンケート５", "アンケート８", "アンケート９", "アンケート１０", "アンケート１１"]
score = train[pos].sum(axis=1)
rs, ps = stats.pointbiserialr(y, score)
print(f"### DX前向きスコア合計(4除く8項目) の効果: r={rs:+.3f} p={ps:.4f}")

# ---- まとめ ----
print(f"\n{'='*70}\n### 効き順まとめ")
summ = pd.DataFrame(rows).sort_values("|r|", ascending=False)
print(summ[["変数", "意味", "r", "spearman", "p"]].round(4).to_string(index=False))
