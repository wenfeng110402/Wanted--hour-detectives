import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb

TARGET = "hours_spent"
ID = "project_id"
LABEL_COLS = [TARGET, 'log_hours', 'hours_bucket']
DROP = [ID] + LABEL_COLS

# 上限
HOURS_CAP = 140.7

FAKE_BOOL = ['has_custom_art',
 'is_mobile_responsive',
 'is_multiplayer',
 'mentions_ai_api',
 'is_clone_or_tutorial']

ORDINAL = {
    "author_prior_projects": {"0": 0, "1-2": 1, "3-5": 2, "6+": 3},
    "author_account_age_years": {"<1": 0, "1-2": 1, "3-5": 2, "6+": 3},
}

CATS = ["commit_cadence", "primary_language", "framework", "hosting_type", "project_category"]

def build_features(df, categories=None):
    """把原始表变成模型能吃的特征矩阵。

    categories: {列名: 训练集里的类别列表}，保证 test 的类别编码和 train 一致。
    """
    X = df.drop(columns=[c for c in DROP if c in df.columns]).copy()

    # 1. 一次性上传信号
    X["is_one_shot"] = (X["pct_code_in_first_commit"] > 0.9).astype(int)

    # 2. 假布尔（因为有 NaN，pandas 读成了字符串）
    for col in FAKE_BOOL:
        if col in X.columns:
            X[col] = (
                X[col]
                .map({True: 1, False: 0, "True": 1, "False": 0})
                .fillna(0)
                .astype(int)
            )

    # 3. 序数
    for col, mapping in ORDINAL.items():
        if col in X.columns:
            X[col] = X[col].map(mapping)

    # 4. 真布尔
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    # 5. 注水信号
    X["is_zero_span"] = (X["project_span_days"] == 0).astype(int)
    X["low_commit_high_loc"] = (
        (X["num_commits"] <= 3) & (X["lines_of_code"] > 2000)
    ).astype(int)
    X["suspicious_speed"] = X["lines_of_code"] / X["active_commit_days"].clip(lower=1)

    # 6. 分类列，对齐到训练集的类别
    for col in CATS:
        if col not in X.columns:
            continue
        if categories is not None and col in categories:
            X[col] = pd.Categorical(X[col], categories=categories[col])
        else:
            X[col] = X[col].astype("category")

    return X

def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    y = train[TARGET].values
    X_train = build_features(train)

    categories = {
        col: X_train[col].cat.categories
        for col in CATS
        if col in X_train.columns
    }
    X_test = build_features(test, categories=categories)

    # 列顺序必须和训练时完全一致
    X_test = X_test.reindex(columns=X_train.columns)

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=20,
        verbose=-1,
        random_state=42,
    )
    model.fit(X_train, np.log1p(y))

    pred = np.expm1(model.predict(X_test))
    pred = np.clip(pred, 0.1, HOURS_CAP)   # 下限防 0

    out = pd.DataFrame({ID: test[ID].values, TARGET: pred})
    out.to_csv("predictions.csv", index=False)
    print(f"wrote predictions.csv  rows={len(out)}")
    print(out.head())


if __name__ == "__main__":
    main()