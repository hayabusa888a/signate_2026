"""train/test 分布シフトの精査。姿勢差など生特徴が本当にシフトしているか。"""
import sys
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
from feat_lib import POS_WORDS, NEG_WORDS

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")


def count_words(series, words):
    pat = re.compile("|".join(map(re.escape, words)))
    return series.fillna("").apply(lambda s: len(pat.findall(s)))


for name, df in [("train", train), ("test", test)]:
    concat = df["企業概要"].fillna("") + " " + df["今後のDX展望"].fillna("")
    pos = count_words(concat, POS_WORDS)
    neg = count_words(concat, NEG_WORDS)
    print(f"\n[{name}] 姿勢_積極: mean={pos.mean():.2f} median={pos.median():.0f} / "
          f"姿勢_抑制: mean={neg.mean():.2f} median={neg.median():.0f} / "
          f"姿勢差: mean={(pos-neg).mean():.2f}")
    print(f"  姿勢差 分布: {np.percentile(pos-neg, [0,10,25,50,75,90,100])}")

# 生テキスト長も比較
print("\n=== テキスト長 train vs test ===")
for col in ["企業概要", "今後のDX展望", "組織図"]:
    tl = train[col].fillna("").str.len(); el = test[col].fillna("").str.len()
    print(f"  {col}: train median={tl.median():.0f} mean={tl.mean():.0f} / test median={el.median():.0f} mean={el.mean():.0f}")

# 生の数値特徴だけで adversarial（rank/zを排除）
print("\n=== 生特徴のみ adversarial validation ===")
raw_num = [c for c in train.columns if c not in ["企業ID", "購入フラグ", "企業名", "企業概要", "組織図", "今後のDX展望", "業界", "上場種別", "特徴"]]
# テキスト由来の姿勢差も追加
def add_posture(df):
    concat = df["企業概要"].fillna("") + " " + df["今後のDX展望"].fillna("")
    return (count_words(concat, POS_WORDS) - count_words(concat, NEG_WORDS))

Xtr = train[raw_num].copy(); Xtr["姿勢差"] = add_posture(train).values
Xte = test[raw_num].copy(); Xte["姿勢差"] = add_posture(test).values
allX = pd.concat([Xtr, Xte], ignore_index=True)
yadv = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(allX)); imps = np.zeros(allX.shape[1])
params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31,
          "min_child_samples": 20, "feature_fraction": 0.8, "verbosity": -1, "seed": 42}
for tr, va in skf.split(allX, yadv):
    m = lgb.train(params, lgb.Dataset(allX.iloc[tr], label=yadv[tr]), num_boost_round=300,
                  valid_sets=[lgb.Dataset(allX.iloc[va], label=yadv[va])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    oof[va] = m.predict(allX.iloc[va], num_iteration=m.best_iteration)
    imps += m.feature_importance(importance_type="gain") / 5
print(f"生特徴のみ adversarial AUC = {roc_auc_score(yadv, oof):.4f}")
print("上位10:")
print(pd.Series(imps, index=allX.columns).sort_values(ascending=False).head(10).round(1).to_string())

# 姿勢差を除いたら？
print("\n=== 姿勢差を除いた adversarial ===")
allX2 = allX.drop(columns=["姿勢差"])
oof2 = np.zeros(len(allX2))
for tr, va in skf.split(allX2, yadv):
    m = lgb.train(params, lgb.Dataset(allX2.iloc[tr], label=yadv[tr]), num_boost_round=300,
                  valid_sets=[lgb.Dataset(allX2.iloc[va], label=yadv[va])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    oof2[va] = m.predict(allX2.iloc[va], num_iteration=m.best_iteration)
print(f"姿勢差なし adversarial AUC = {roc_auc_score(yadv, oof2):.4f}")
