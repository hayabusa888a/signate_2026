"""テキスト列 と 購入フラグ の相関分析"""
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from janome.tokenizer import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 200)

train = pd.read_csv("../data/train.csv")
y = train["購入フラグ"].values
base = y.mean()
print(f"全体購入率: {base:.3f}  (購入{y.sum()} / 全{len(y)})")

text_cols = ["企業概要", "組織図", "今後のDX展望", "企業名"]

# ========== 1) 文字数・行数の相関 ==========
print(f"\n{'='*70}\n### 1) テキスト長 と 購入フラグ")
rows = []
for c in text_cols:
    length = train[c].fillna("").str.len()
    r, p = stats.pointbiserialr(y, length)
    rows.append({"変数": f"{c}_文字数", "非購入_median": length[y == 0].median(),
                 "購入_median": length[y == 1].median(), "r": r, "p": p})
lines = train["組織図"].fillna("").str.count("\n") + 1
r, p = stats.pointbiserialr(y, lines)
rows.append({"変数": "組織図_行数", "非購入_median": lines[y == 0].median(),
             "購入_median": lines[y == 1].median(), "r": r, "p": p})
print(pd.DataFrame(rows).round(4).to_string(index=False))

# ========== 2) DXキーワード出現の効果 ==========
print(f"\n{'='*70}\n### 2) DX関連キーワードの出現有無 × 購入率")
DX_KEYWORDS = ["DX", "AI", "IoT", "クラウド", "デジタル", "自動化", "データ活用", "RPA",
               "人材育成", "教育", "研修", "リスキリング", "スキル", "内製化", "生成AI"]
concat_text = (train["企業概要"].fillna("") + " " + train["今後のDX展望"].fillna(""))
krows = []
for kw in DX_KEYWORDS:
    present = concat_text.str.contains(re.escape(kw))
    if present.sum() < 10:
        continue
    pr = y[present.values].mean()
    pr_no = y[~present.values].mean()
    krows.append({"キーワード": kw, "出現数": int(present.sum()),
                  "出現時購入率": pr, "非出現時購入率": pr_no, "差": pr - pr_no})
kdf = pd.DataFrame(krows).sort_values("差", ascending=False)
print(kdf.round(3).to_string(index=False))

# ========== 3) 組織図のDX部署 ==========
print(f"\n{'='*70}\n### 3) 組織図内のDX関連部署")
DX_DEPT = ["DX推進", "DX戦略", "デジタル推進", "IT戦略", "情報システム", "DX室", "デジタル戦略"]
pat = re.compile("|".join(map(re.escape, DX_DEPT)))
has_dx_dept = train["組織図"].fillna("").apply(lambda s: bool(pat.search(s)))
print(f"  DX部署あり: {has_dx_dept.sum()}社 購入率={y[has_dx_dept.values].mean():.3f}")
print(f"  DX部署なし: {(~has_dx_dept).sum()}社 購入率={y[~has_dx_dept.values].mean():.3f}")
r, p = stats.pointbiserialr(y, has_dx_dept.astype(int))
print(f"  point-biserial r={r:+.3f} p={p:.4f}")

# ========== 4) 購入を分ける単語（log-odds, janome） ==========
print(f"\n{'='*70}\n### 4) 購入を分ける単語 top（企業概要+DX展望, janome分かち書き）")
tokenizer = Tokenizer()
def tok(t):
    if not isinstance(t, str) or t == "":
        return ""
    return " ".join(tokenizer.tokenize(t, wakati=True))

docs = concat_text.map(tok).tolist()
vec = CountVectorizer(min_df=15, token_pattern=r"(?u)\b\w+\b")
Xc = (vec.fit_transform(docs) > 0).astype(int).toarray()
vocab = np.array(vec.get_feature_names_out())

# 各単語: 出現群の購入率、log-odds（+1スムージング）
n1 = y.sum()
n0 = len(y) - n1
res = []
for j, w in enumerate(vocab):
    present = Xc[:, j] == 1
    cnt = present.sum()
    if cnt < 15:
        continue
    a = y[present].sum()        # 購入 & 出現
    b = cnt - a                 # 非購入 & 出現
    # smoothed log-odds of purchase given word present vs absent
    a2 = y[~present].sum()
    b2 = (~present).sum() - a2
    lo = np.log(((a + 0.5) / (b + 0.5)) / ((a2 + 0.5) / (b2 + 0.5)))
    res.append({"単語": w, "出現数": int(cnt), "出現時購入率": y[present].mean(), "log_odds": lo})
rdf = pd.DataFrame(res)
print("\n-- 購入と結びつく単語 top15 --")
print(rdf.sort_values("log_odds", ascending=False).head(15).round(3).to_string(index=False))
print("\n-- 非購入と結びつく単語 top15 --")
print(rdf.sort_values("log_odds").head(15).round(3).to_string(index=False))
