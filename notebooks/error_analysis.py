"""エラー分析 — v2のOOF予測で外している企業(FP/FN)の特徴を調べる"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 200)

ver = sys.argv[1] if len(sys.argv) > 1 else "v3"
train = pd.read_csv("../data/train.csv")
oof = pd.read_csv(f"../features/oof_pred_{ver}.csv")
th = float(oof["threshold"].iloc[0])
print(f"[version={ver}]")

df = train.merge(oof[["企業ID", "oof_pred", "y"]], on="企業ID", how="inner")
df["pred"] = (df["oof_pred"] >= th).astype(int)

# 混同行列の区分
def kind(r):
    if r["y"] == 1 and r["pred"] == 1: return "TP"
    if r["y"] == 0 and r["pred"] == 0: return "TN"
    if r["y"] == 0 and r["pred"] == 1: return "FP"  # 買わないのに買うと予測
    return "FN"                                      # 買うのに見逃し
df["区分"] = df.apply(kind, axis=1)

print(f"閾値={th:.2f}")
print(df["区分"].value_counts().to_string())
tp = (df["区分"] == "TP").sum(); fp = (df["区分"] == "FP").sum()
fn = (df["区分"] == "FN").sum(); tn = (df["区分"] == "TN").sum()
prec = tp / (tp + fp); rec = tp / (tp + fn)
print(f"Precision={prec:.3f}  Recall={rec:.3f}  F1={2*prec*rec/(prec+rec):.3f}")
print(f"FP={fp}件(買わないのに買う予測)  FN={fn}件(買うのに見逃し)")

# ========== 業界別のエラー傾向 ==========
print(f"\n{'='*70}\n### 業界別 誤分類率（件数>=15）")
g = df.groupby("業界").agg(
    n=("y", "size"), 購入率=("y", "mean"),
    誤り率=("区分", lambda s: s.isin(["FP", "FN"]).mean()),
    FP=("区分", lambda s: (s == "FP").sum()),
    FN=("区分", lambda s: (s == "FN").sum()),
)
print(g[g["n"] >= 15].sort_values("誤り率", ascending=False).round(3).to_string())

# ========== FN vs TP（買う企業の中で、見逃しと正解の違い） ==========
print(f"\n{'='*70}\n### FN(見逃し) vs TP(正解): 買う企業の中での違い")
buyers = df[df["y"] == 1]
compare_cols = ["従業員数", "売上高経常利益率_x" if "売上高経常利益率_x" in df else None]
# 派生比率を再計算
def sd(a, b): return a / b.replace(0, np.nan)
buyers = buyers.copy()
buyers["経常利益率"] = sd(buyers["経常利益"], buyers["売上"])
buyers["ROA"] = sd(buyers["当期純利益"], buyers["総資産"])
buyers["有利子負債比率"] = sd(buyers["短期借入金"] + buyers["長期借入金"], buyers["総資産"])
buyers["自己資本比率"] = sd(buyers["自己資本"], buyers["総資産"])
for c in ["oof_pred", "従業員数", "経常利益率", "ROA", "有利子負債比率", "自己資本比率",
          "アンケート１", "アンケート７", "アンケート１０"]:
    fn_v = buyers.loc[buyers["区分"] == "FN", c]
    tp_v = buyers.loc[buyers["区分"] == "TP", c]
    print(f"  {c:14s} FN median={fn_v.median():.3f} / TP median={tp_v.median():.3f}")

# ========== FP vs TN（買わない企業の中で、誤検知と正解の違い） ==========
print(f"\n{'='*70}\n### FP(誤検知) vs TN(正解): 買わない企業の中での違い")
nonbuyers = df[df["y"] == 0].copy()
nonbuyers["経常利益率"] = sd(nonbuyers["経常利益"], nonbuyers["売上"])
nonbuyers["有利子負債比率"] = sd(nonbuyers["短期借入金"] + nonbuyers["長期借入金"], nonbuyers["総資産"])
for c in ["oof_pred", "従業員数", "経常利益率", "有利子負債比率",
          "アンケート１", "アンケート７", "アンケート１０"]:
    fp_v = nonbuyers.loc[nonbuyers["区分"] == "FP", c]
    tn_v = nonbuyers.loc[nonbuyers["区分"] == "TN", c]
    print(f"  {c:14s} FP median={fp_v.median():.3f} / TN median={tn_v.median():.3f}")

# ========== 最も確信して外したケース ==========
print(f"\n{'='*70}\n### 最も確信して外したケース")
print("-- 自信満々のFP（買わないのに高確率で買う予測） top8 --")
cols_show = ["企業ID", "業界", "上場種別", "特徴", "従業員数", "oof_pred"]
print(df[df["区分"] == "FP"].nlargest(8, "oof_pred")[cols_show].round(3).to_string(index=False))
print("\n-- 完全に見逃したFN（買うのに低確率） top8 --")
print(df[df["区分"] == "FN"].nsmallest(8, "oof_pred")[cols_show].round(3).to_string(index=False))

df[["企業ID", "業界", "上場種別", "y", "pred", "oof_pred", "区分"]].to_csv(
    f"../features/error_analysis_{ver}.csv", index=False, encoding="utf-8-sig")
print(f"\nsaved features/error_analysis_{ver}.csv")
