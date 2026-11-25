import numpy as np
import pandas as pd
import lightgbm as lgb

TARGET = "hours_spent"
ID = "project_id"
HOURS_CAP = 140.7

FAKE_BOOL = [
    "has_custom_art",
    "is_mobile_responsive",
    "is_multiplayer",
    "mentions_ai_api",
    "is_clone_or_tutorial",
]

ORDINAL = {
    "author_prior_projects": {"0": 0, "1-2": 1, "3-5": 2, "6+": 3},
    "author_account_age_years": {"<1": 0, "1-2": 1, "3-5": 2, "6+": 3},
}

CATS = ["commit_cadence", "primary_language", "framework", "hosting_type", "project_category"]

# 按特征重要性选出的 top 40 特征（全量数据上 LightGBM 计算）
TOP40 = [
    "description_word_count", "project_span_days", "median_lines_per_commit",
    "active_commit_days", "primary_language", "suspicious_speed",
    "comment_line_ratio", "pct_commits_weekend", "pct_code_in_first_commit",
    "num_commits", "avg_commit_msg_length", "hosting_type", "avg_file_length",
    "lines_of_code", "page_weight_kb", "num_source_files",
    "pct_generic_commit_msgs", "readme_word_count", "pct_primary_language",
    "asset_size_mb", "framework", "churn_ratio", "max_lines_in_one_commit",
    "num_image_files", "lines_added_total", "pct_commits_late_night",
    "readme_num_headings", "num_merge_commits", "load_time_ms",
    "num_commit_authors", "stars", "num_internal_pages",
    "author_prior_projects", "project_category", "num_dependencies",
    "num_directories", "lines_deleted_total", "num_js_files_loaded",
    "max_dir_depth", "readme_num_images",
]


def build_features(df, feature_cols, categories=None):
    """把原始表变成模型能吃的特征矩阵。

    feature_cols: 特征列名列表，由 test.csv 决定，自动排除所有 label 列。
    categories:   {列名: 训练集里的类别列表}，保证 test 的类别编码和 train 一致。
    """
    X = df[feature_cols].copy()

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

    # 7. 兜底：任何残留的 object 列转成 category
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category")

    return X


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # 特征列由 test.csv 决定：它没有 label 列，所以自动排除。
    # 再和 train 的列取交集，防止本地假 test.csv 多出派生列。
    feature_cols = [
        c for c in test.columns if c != ID and c in train.columns
    ]

    y = train[TARGET].values
    X_train = build_features(train, feature_cols)

    categories = {
        col: X_train[col].cat.categories
        for col in CATS
        if col in X_train.columns
    }
    X_test = build_features(test, feature_cols, categories=categories)

    # 列顺序必须和训练时完全一致
    X_test = X_test.reindex(columns=X_train.columns)

    # 只用 top 40 特征，减少噪声、提升泛化
    X_train = X_train[TOP40]
    X_test = X_test[TOP40]

    # 5 个 seed 各训一个模型再平均，降低随机波动
    SEEDS = [42, 7, 2024, 123, 999]
    preds = []
    for seed in SEEDS:
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=7,
            min_child_samples=20,
            colsample_bytree=0.5,   # 每棵树只看一半特征，抗过拟合
            n_jobs=-1,
            verbose=-1,
            random_state=seed,
        )
        model.fit(X_train, np.log1p(y))
        preds.append(np.expm1(model.predict(X_test)))

    pred = np.mean(preds, axis=0)
    pred = np.clip(pred, 0.1, HOURS_CAP)

    out = pd.DataFrame({ID: test[ID].values, TARGET: pred})
    out.to_csv("predictions.csv", index=False)
    print(f"wrote predictions.csv  rows={len(out)}")
    print(out.head())


if __name__ == "__main__":
    main()
