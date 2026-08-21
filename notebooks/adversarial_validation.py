"""Adversarial Validation: train と test を識別できるか？
AUC≒0.5 なら同一分布(乖離は純粋な過学習)。AUC高なら分布シフト(その特徴を要注意)。
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

from feat_lib import build_static, cat_cols

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

Xtr = build_static(train); Xtr["is_test"] = 0
Xte = build_static(test); Xte["is_test"] = 1
allX = pd.concat([Xtr, Xte], ignore_index=True)
yadv = allX.pop("is_test").values

for c in cat_cols:
    allX[c] = allX[c].astype("category")

feat = [c for c in allX.columns]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(allX))
imps = np.zeros(len(feat))
params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31,
          "min_child_samples": 20, "feature_fraction": 0.8, "verbosity": -1, "seed": 42}
for tr, va in skf.split(allX, yadv):
    d = lgb.Dataset(allX.iloc[tr], label=yadv[tr], categorical_feature=cat_cols)
    dv = lgb.Dataset(allX.iloc[va], label=yadv[va], categorical_feature=cat_cols)
    m = lgb.train(params, d, num_boost_round=300, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    oof[va] = m.predict(allX.iloc[va], num_iteration=m.best_iteration)
    imps += m.feature_importance(importance_type="gain") / 5

auc = roc_auc_score(yadv, oof)
print(f"=== Adversarial AUC = {auc:.4f} ===")
print("(0.5=train/test見分けつかない=同一分布 / 高いほど分布シフト大)")
print("\ntrain/testを見分ける特徴 top15 (gain):")
s = pd.Series(imps, index=feat).sort_values(ascending=False)
print(s.head(15).round(1).to_string())

# シフトしている特徴の train vs test 平均比較
print("\n上位シフト特徴の train vs test 中央値:")
for c in s.head(8).index:
    if c in cat_cols:
        continue
    print(f"  {c:24s} train={Xtr[c].median():.3f}  test={Xte[c].median():.3f}")
