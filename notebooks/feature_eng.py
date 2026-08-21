import re

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42

target_col = "購入フラグ"
id_col = "企業ID"

cat_cols = ["業界", "上場種別", "特徴"]
text_cols = ["企業概要", "組織図", "今後のDX展望", "企業名"]

DX_KEYWORDS = ["DX", "AI", "IoT", "クラウド", "デジタル", "自動化", "データ活用", "RPA"]
DX_DEPT_KEYWORDS = ["DX推進", "DX戦略", "デジタル推進", "IT戦略", "情報システム"]


def safe_div(a, b):
    return a / b.replace(0, np.nan)


def add_financial_features(df):
    f = pd.DataFrame(index=df.index)

    f["自己資本比率"] = safe_div(df["自己資本"], df["総資産"])
    f["ROE"] = safe_div(df["当期純利益"], df["自己資本"])
    f["ROA"] = safe_div(df["当期純利益"], df["総資産"])
    f["売上高営業利益率"] = safe_div(df["営業利益"], df["売上"])
    f["売上高経常利益率"] = safe_div(df["経常利益"], df["売上"])
    f["流動資産比率"] = safe_div(df["流動資産"], df["総資産"])
    f["負債比率"] = safe_div(df["負債"], df["純資産"])

    有利子負債 = df["短期借入金"] + df["長期借入金"]
    f["有利子負債"] = 有利子負債
    f["有利子負債比率"] = safe_div(有利子負債, df["総資産"])

    f["一人当たり売上"] = safe_div(df["売上"], df["従業員数"])
    f["一人当たり営業利益"] = safe_div(df["営業利益"], df["従業員数"])
    f["事業所あたり従業員数"] = safe_div(df["従業員数"], df["事業所数"])

    f["フリーCF"] = df["営業CF"] + df["投資CF"]
    f["減価償却費率"] = safe_div(df["減価償却費"], df["売上"])
    f["無形固定資産変動率"] = safe_div(df["無形固定資産変動(ソフトウェア関連)"], df["総資産"])
    f["有形固定資産変動率"] = safe_div(df["有形固定資産変動"], df["総資産"])
    f["ソフトウェア投資対有形固定資産比"] = safe_div(
        df["無形固定資産変動(ソフトウェア関連)"], df["有形固定資産変動"].abs() + 1
    )

    return f


def add_survey_features(df):
    f = pd.DataFrame(index=df.index)

    positive_cols = ["アンケート１", "アンケート２", "アンケート３", "アンケート８", "アンケート９", "アンケート１０", "アンケート１１"]
    f["DX前向きスコア合計"] = df[positive_cols].sum(axis=1)
    f["DX前向きスコア平均"] = df[positive_cols].mean(axis=1)
    f["DX抵抗感"] = df["アンケート４"]
    f["戦略実感ギャップ"] = df["アンケート２"] - df["アンケート１"]
    f["導入済みフラグ"] = (df["アンケート６"] == 1).astype(int)
    f["既存ツール満足度欠損フラグ"] = df["アンケート７"].isna().astype(int)
    f["セキュリティ整備度"] = df["アンケート５"]

    return f


def count_keywords(series, keywords):
    pattern = re.compile("|".join(map(re.escape, keywords)))
    return series.fillna("").apply(lambda s: len(pattern.findall(s)))


def add_text_features(df):
    f = pd.DataFrame(index=df.index)
    for c in text_cols:
        s = df[c].fillna("")
        f[f"{c}_len"] = s.str.len()

    f["DXキーワード数_企業概要"] = count_keywords(df["企業概要"], DX_KEYWORDS)
    f["DXキーワード数_DX展望"] = count_keywords(df["今後のDX展望"], DX_KEYWORDS)
    f["DX部署有無_組織図"] = (count_keywords(df["組織図"], DX_DEPT_KEYWORDS) > 0).astype(int)
    f["組織図行数"] = df["組織図"].fillna("").str.count("\n") + 1

    return f


def add_structure_features(df):
    f = pd.DataFrame(index=df.index)
    f["工場数欠損フラグ"] = df["工場数"].isna().astype(int)
    f["店舗数欠損フラグ"] = df["店舗数"].isna().astype(int)
    f["店舗数_事業所数比"] = safe_div(df["店舗数"].fillna(0), df["事業所数"])
    f["工場数_事業所数比"] = safe_div(df["工場数"].fillna(0), df["事業所数"])
    return f


def build_features(df):
    num_cat_cols = [c for c in df.columns if c not in [target_col, id_col] + text_cols]
    base = df[num_cat_cols].copy()
    parts = [
        base,
        add_financial_features(df),
        add_survey_features(df),
        add_text_features(df),
        add_structure_features(df),
    ]
    return pd.concat(parts, axis=1)


def align_categories(X, X_test):
    for c in cat_cols:
        X[c] = X[c].fillna("missing").astype("category")
        X_test[c] = X_test[c].fillna("missing").astype("category")
        all_cats = pd.concat([X[c], X_test[c]]).unique()
        X[c] = X[c].cat.set_categories(all_cats)
        X_test[c] = X_test[c].cat.set_categories(all_cats)
    return X, X_test


if __name__ == "__main__":
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")

    y = train[target_col].values
    test_ids = test[id_col].values

    X = build_features(train)
    X_test = build_features(test)
    X, X_test = align_categories(X, X_test)

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

    print(f"\nOOF best F1={best_f1:.4f} at threshold={best_th:.2f}  (baseline was 0.6379 @ 0.28)")

    final_pred = (test_pred >= best_th).astype(int)
    submission = pd.DataFrame({0: test_ids, 1: final_pred})
    submission.to_csv("../submission/feature_eng_submission.csv", index=False, header=False)
    print("saved submission/feature_eng_submission.csv")
    print(submission[1].value_counts())

    importance = pd.Series(model.feature_importance(importance_type="gain"), index=feature_cols).sort_values(ascending=False)
    print("\nTop 20 feature importances (last fold model, gain):")
    print(importance.head(20))
