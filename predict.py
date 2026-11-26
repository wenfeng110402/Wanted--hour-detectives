"""Hack Club Hour Detective - prediction script.

用法:
    python predict.py

要求 train.csv 和 test.csv 与本文件同目录。
输出 predictions.csv，列名严格为 project_id,hours_spent
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import StratifiedKFold

TARGET = "hours_spent"
ID = "project_id"
HOURS_CAP = 140.7

PRIOR_MAP = {"0": 0, "1-2": 1, "3-5": 2, "6+": 3}
AGE_MAP = {"<1": 0, "1-2": 1, "3-5": 2, "6+": 3}
BOOL_MAP = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}

CAT_COLS = ["primary_language", "framework", "hosting_type", "commit_cadence", "project_category"]
BOOL_COLS = [
    "uses_game_engine", "has_tests", "has_ci_config", "has_dockerfile", "has_backend",
    "has_database", "has_custom_art", "readme_has_install_instructions", "has_license",
    "has_docs_folder", "url_is_live", "is_mobile_responsive", "has_custom_domain",
    "is_multiplayer", "mentions_ai_api", "is_clone_or_tutorial", "is_forked_from_template",
]


def clean(X):
    X = X.copy()
    X["author_prior_projects"] = X["author_prior_projects"].astype(str).map(PRIOR_MAP)
    X["author_account_age_years"] = X["author_account_age_years"].astype(str).map(AGE_MAP)
    for c in BOOL_COLS:
        if c in X.columns:
            X[c] = X[c].map(BOOL_MAP)
    for c in CAT_COLS:
        X[c] = X[c].fillna("missing").astype(str)
    return X


def div(a, b):
    return a / b.replace(0, np.nan)


def add_features(X):
    X = X.copy()
    X["lines_per_active_day"] = div(X.lines_added_total, X.active_commit_days)
    X["loc_per_commit"] = div(X.lines_of_code, X.num_commits)
    X["commits_per_active_day"] = div(X.num_commits, X.active_commit_days)
    X["active_frac"] = div(X.active_commit_days, X.project_span_days + 1)
    X["total_lines_touched"] = np.log1p(X.lines_added_total + X.lines_deleted_total)
    X["biggest_commit_share"] = div(X.max_lines_in_one_commit, X.lines_added_total)
    X["bulk_upload"] = (X.pct_code_in_first_commit > 0.8).astype(float).where(X.pct_code_in_first_commit.notna())
    X["few_commits"] = (X.num_commits <= 3).astype(float).where(X.num_commits.notna())
    X["no_git"] = X.num_commits.isna().astype(float)
    X["n_assets"] = X[["num_image_files", "num_audio_files", "num_3d_model_files", "num_font_files"]].sum(axis=1, min_count=1)
    X["readme_richness"] = X[["readme_num_headings", "readme_num_images", "readme_num_code_blocks"]].sum(axis=1, min_count=1)
    X["loc_x_days"] = np.log1p(X.lines_of_code) * np.log1p(X.active_commit_days)
    X["loc_template_adj"] = np.log1p(X.lines_of_code) * (1 - 0.5 * X.is_forked_from_template.fillna(0))
    return X


def prepare(X):
    # project_id 可能已经在 feature_cols 里被排除了，这里安全地 drop
    if ID in X.columns:
        X = X.drop(columns=[ID])
    return add_features(clean(X))


def eval_log_rmsle(y_true, log_pred):
    pred = np.clip(np.expm1(log_pred), 0.1, HOURS_CAP)
    return root_mean_squared_log_error(y_true, pred)


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    y = train[TARGET].values
    y_log = np.log1p(y)
    # 特征列 = test.csv 的列（去掉 project_id），再和 train 取交集
    feature_cols = [c for c in test.columns if c != ID and c in train.columns]
    X_train = prepare(train[feature_cols])
    X_test = prepare(test)

    # 列顺序对齐
    X_test = X_test.reindex(columns=X_train.columns)

    X_train_tree = X_train.copy()
    X_test_tree = X_test.copy()
    for c in CAT_COLS:
        X_train_tree[c] = X_train_tree[c].astype("category")
        X_test_tree[c] = X_test_tree[c].astype("category")

    stratify_col = (
        train["hours_bucket"]
        if "hours_bucket" in train.columns
        else pd.qcut(y_log, q=5, labels=False)
    )
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_cb = np.zeros(len(X_train))
    oof_lgb = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))
    test_cb = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, stratify_col)):
        cb = CatBoostRegressor(
            cat_features=CAT_COLS,
            iterations=3000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5,
            rsm=0.5,
            random_seed=42,
            verbose=0,
        )
        cb.fit(
            X_train.iloc[tr_idx], y_log[tr_idx],
            eval_set=(X_train.iloc[va_idx], y_log[va_idx]),
            early_stopping_rounds=200,
        )
        oof_cb[va_idx] = cb.predict(X_train.iloc[va_idx])
        test_cb += cb.predict(X_test) / kf.n_splits

        lg = lgb.LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=42,
            verbose=-1,
        )
        lg.fit(
            X_train_tree.iloc[tr_idx], y_log[tr_idx],
            eval_set=[(X_train_tree.iloc[va_idx], y_log[va_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_lgb[va_idx] = lg.predict(X_train_tree.iloc[va_idx])
        test_lgb += lg.predict(X_test_tree) / kf.n_splits

        xg = xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            max_depth=5,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=1.0,
            reg_lambda=5.0,
            enable_categorical=True,
            tree_method="hist",
            early_stopping_rounds=100,
            random_state=42,
        )
        xg.fit(
            X_train_tree.iloc[tr_idx], y_log[tr_idx],
            eval_set=[(X_train_tree.iloc[va_idx], y_log[va_idx])],
            verbose=False,
        )
        oof_xgb[va_idx] = xg.predict(X_train_tree.iloc[va_idx])
        test_xgb += xg.predict(X_test_tree) / kf.n_splits

    w_cb, w_lgb, w_xgb = 0.40, 0.05, 0.55
    oof_blend = w_cb * oof_cb + w_lgb * oof_lgb + w_xgb * oof_xgb
    test_blend = w_cb * test_cb + w_lgb * test_lgb + w_xgb * test_xgb

    print(f"OOF RMSLE - CB: {eval_log_rmsle(y, oof_cb):.5f} | LGB: {eval_log_rmsle(y, oof_lgb):.5f} | XGB: {eval_log_rmsle(y, oof_xgb):.5f}")
    print(f"OOF RMSLE - Ensemble (0.40 CB + 0.05 LGB + 0.55 XGB): {eval_log_rmsle(y, oof_blend):.5f}")

    pred = np.clip(np.expm1(test_blend), 0.1, HOURS_CAP)

    out = pd.DataFrame({ID: test[ID].values, TARGET: pred})
    out.to_csv("predictions.csv", index=False)
    print(f"wrote predictions.csv  rows={len(out)}")
    print(out.head())


if __name__ == "__main__":
    main()
