import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

RANDOM_STATE = 42

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

target_col = "購入フラグ"
id_col = "企業ID"

y = train[target_col].values
train_ids = train[id_col].values
test_ids = test[id_col].values

cat_cols = ["業界", "上場種別", "特徴"]
text_cols = ["企業概要", "組織図", "今後のDX展望", "企業名"]

drop_cols = [target_col, id_col] + text_cols
num_cat_cols = [c for c in train.columns if c not in drop_cols]

def text_len_features(df):
    feats = pd.DataFrame(index=df.index)
    for c in text_cols:
        s = df[c].fillna("")
        feats[f"{c}_len"] = s.str.len()
    return feats

X = train[num_cat_cols].copy()
X_test = test[num_cat_cols].copy()

X = pd.concat([X, text_len_features(train)], axis=1)
X_test = pd.concat([X_test, text_len_features(test)], axis=1)

for c in cat_cols:
    X[c] = X[c].fillna("missing").astype("category")
    X_test[c] = X_test[c].fillna("missing").astype("category")
    all_cats = pd.concat([X[c], X_test[c]]).unique()
    X[c] = X[c].cat.set_categories(all_cats)
    X_test[c] = X_test[c].cat.set_categories(all_cats)

feature_cols = list(X.columns)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),
    "verbosity": -1,
    "seed": RANDOM_STATE,
}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dvalid = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_pred[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_pred += model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits

    print(f"fold {fold}: best_iter={model.best_iteration}, logloss={model.best_score['valid_0']['binary_logloss']:.4f}")

best_th, best_f1 = 0.5, -1
for th in np.arange(0.05, 0.95, 0.01):
    f1 = f1_score(y, (oof_pred >= th).astype(int))
    if f1 > best_f1:
        best_f1, best_th = f1, th

print(f"OOF best F1={best_f1:.4f} at threshold={best_th:.2f}")

final_pred = (test_pred >= best_th).astype(int)

submission = pd.DataFrame({0: test_ids, 1: final_pred})
submission.to_csv("../submission/baseline_submission.csv", index=False, header=False)
print("saved submission/baseline_submission.csv")
print(submission[1].value_counts())

importance = pd.Series(model.feature_importance(importance_type="gain"), index=feature_cols).sort_values(ascending=False)
print("\nTop feature importances (last fold model, gain):")
print(importance.head(15))
