"""業界セグメント別に、購入企業 vs 非購入企業 の他情報の違いを見る。"""
import re
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 240)

train = pd.read_csv("../data/train.csv")
y = train["購入フラグ"].values


def sd(a, b): return a / b.replace(0, np.nan)
train["経常利益率"] = sd(train["経常利益"], train["売上"])
train["ROA"] = sd(train["当期純利益"], train["総資産"])
train["有利子負債比率"] = sd(train["短期借入金"] + train["長期借入金"], train["総資産"])
POS = ["圧倒的", "力強", "一段と", "強固", "加速", "積極", "推進", "変革", "挑戦", "拡大", "成長", "先進", "最先端"]
NEG = ["抑える", "小さく", "極力", "スモール", "短時間", "低い", "マニュアル", "慎重", "段階的", "最小限", "見送", "縮小", "様子見"]
def cnt(s, ws):
    pat = re.compile("|".join(map(re.escape, ws)))
    return s.fillna("").apply(lambda t: len(pat.findall(t)))
concat = train["企業概要"].fillna("") + " " + train["今後のDX展望"].fillna("")
train["姿勢差"] = cnt(concat, POS) - cnt(concat, NEG)

FEATS = ["従業員数", "経常利益率", "ROA", "有利子負債比率",
         "アンケート１", "アンケート４", "アンケート７", "アンケート１０", "姿勢差"]

# 主要業界（n>=30）
sizes = train["業界"].value_counts()
major = sizes[sizes >= 30].index.tolist()

lines = []
for ind in major:
    sub = train[train["業界"] == ind]
    ys = sub["購入フラグ"].values
    nb, nn = int(ys.sum()), int((ys == 0).sum())
    lines.append(f"\n{'='*80}\n■ {ind}  (n={len(sub)}, 購入{nb} / 非購入{nn}, 購入率{ys.mean():.2f})")
    rows = []
    for c in FEATS:
        buy = sub.loc[sub["購入フラグ"] == 1, c].median()
        non = sub.loc[sub["購入フラグ"] == 0, c].median()
        diff = "→買う方が高" if buy > non else ("→買う方が低" if buy < non else "＝")
        rows.append({"特徴": c, "購入_中央": round(buy, 3), "非購入_中央": round(non, 3), "傾向": diff})
    lines.append(pd.DataFrame(rows).to_string(index=False))

with open(r"C:/Users/hayat/AppData/Local/Temp/claude/C--Users-hayat/9bfca7fd-7b7d-4eb4-b20c-7df76eea227e/scratchpad/seg_out.txt", "w", encoding="utf-8") as f:
    f.write("主要業界(n>=30)の 購入 vs 非購入 中央値比較\n")
    f.write("\n".join(lines))
print("done")
