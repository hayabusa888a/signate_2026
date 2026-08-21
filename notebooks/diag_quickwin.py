"""簡単にスコアを上げられる「見落とし」を探す診断。"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
y = train["購入フラグ"].values

print(f"train {train.shape} / test {test.shape} / 購入率 {y.mean():.3f}")

# ========== 1) 企業ID 並び順リーク ==========
print("\n=== 1) 企業ID と 購入フラグ ===")
r, p = stats.pointbiserialr(y, train["企業ID"])
print(f"ID vs 購入 相関 r={r:+.3f} p={p:.4f}")
# IDを10分割して購入率
train["_idbin"] = pd.qcut(train["企業ID"], 10, labels=False)
print(train.groupby("_idbin")["購入フラグ"].agg(["count", "mean"]).round(3).to_string())

# ========== 2) train/test 重複・近傍 ==========
print("\n=== 2) train/test の重複チェック ===")
key_cols = [c for c in train.columns if c not in ["企業ID", "購入フラグ", "_idbin"]]
# 企業名の重複
common_names = set(train["企業名"]) & set(test["企業名"])
print(f"企業名の重複: {len(common_names)}件")
# 数値プロファイル(主要財務)が完全一致する行
prof_cols = ["従業員数", "売上", "総資産", "資本金", "純資産"]
tr_key = train[prof_cols].fillna(-999).astype(str).agg("_".join, axis=1)
te_key = test[prof_cols].fillna(-999).astype(str).agg("_".join, axis=1)
overlap = set(tr_key) & set(te_key)
print(f"財務プロファイル完全一致(train∩test): {len(overlap)}件")
dup_in_train = tr_key.duplicated().sum()
print(f"train内の財務プロファイル重複: {dup_in_train}件")

# ========== 3) 単独で強い特徴（決定的ルール）を AUC で探す ==========
print("\n=== 3) 単独AUCが高い特徴 top15 ===")
from sklearn.metrics import roc_auc_score
num_cols = train.select_dtypes(include=[np.number]).columns
num_cols = [c for c in num_cols if c not in ["企業ID", "購入フラグ", "_idbin"]]
aucs = []
for c in num_cols:
    v = train[c]
    mask = v.notna()
    if mask.sum() < 50 or v[mask].nunique() < 3:
        continue
    try:
        a = roc_auc_score(y[mask], v[mask])
        aucs.append((c, max(a, 1 - a), mask.sum()))
    except Exception:
        pass
adf = pd.DataFrame(aucs, columns=["feature", "auc(方向自由)", "n"]).sort_values("auc(方向自由)", ascending=False)
print(adf.head(15).round(3).to_string(index=False))

# ========== 4) 業界×上場種別 などのセグメント購入率（決定的セグメント探し） ==========
print("\n=== 4) 高純度セグメント（業界別 購入率 0 or 高） ===")
g = train.groupby("業界")["購入フラグ"].agg(["count", "mean"])
print("購入率0%の業界:", g[(g["mean"] == 0) & (g["count"] >= 5)].index.tolist())
print("購入率>=45%の業界:", g[(g["mean"] >= 0.45) & (g["count"] >= 10)].index.tolist())

# ========== 5) 現行OOFの陽性率と閾値感度 ==========
print("\n=== 5) 陽性率・閾値の感度（現行OOF v3利用） ===")
try:
    oof = pd.read_csv("../features/oof_pred_v3.csv")
    from sklearn.metrics import f1_score
    s = oof.set_index("企業ID").loc[train["企業ID"], "oof_pred"].values
    print("閾値ごとの F1 と 予測陽性率:")
    for q in [0.15, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40]:
        th = np.quantile(s, 1 - q)  # 上位q割を陽性に
        f = f1_score(y, (s >= th).astype(int))
        print(f"  予測陽性率={q:.2f} → F1={f:.4f}")
except FileNotFoundError:
    print("oof_pred_v3.csv なし")
