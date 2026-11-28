import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

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

# 稀疏/低价值列，删除以减少随机列采样干扰
DROP_COLS = [
    "no_git", "has_dockerfile", "few_commits", "has_docs_folder",
    "mentions_ai_api", "uses_game_engine", "has_ci_config", "has_database",
    "has_tests", "num_issues", "forks",
    "num_3d_model_files", "num_audio_files", "num_font_files",
]


def div(a, b):
    return a / b.replace(0, np.nan)


def prepare_base(X):
    X = X.copy()
    if ID in X.columns:
        X = X.drop(columns=[ID])

    X["author_prior_projects"] = X["author_prior_projects"].astype(str).map(PRIOR_MAP)
    X["author_account_age_years"] = X["author_account_age_years"].astype(str).map(AGE_MAP)
    for c in BOOL_COLS:
        if c in X.columns:
            X[c] = X[c].map(BOOL_MAP)
    for c in CAT_COLS:
        X[c] = X[c].fillna("missing").astype(str)

    # 核心特征
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

    # 真实工作信号（删改量、有效提交、文档投入）
    X["del_x_days"] = np.log1p(X.lines_deleted_total) * np.log1p(X.active_commit_days)
    X["commits_x_days"] = np.log1p(X.num_commits) * np.log1p(X.active_commit_days)
    X["genuine_commits"] = X.num_commits * (1 - X.pct_generic_commit_msgs.fillna(0))
    X["del_to_add_ratio"] = div(X.lines_deleted_total, X.lines_added_total + 1)
    X["loc_per_file"] = div(X.lines_of_code, X.num_source_files)
    X["docs_x_desc"] = np.log1p(X.readme_word_count) * np.log1p(X.description_word_count)

    # 战线拉长但实际投入少 = 弃坑/水项目信号（高估组 span 中位数 134 天 vs 对照组 52 天）
    #X["span_per_active_day"] = div(X.project_span_days, X.active_commit_days)
    #X["commits_per_span"] = div(X.num_commits, X.project_span_days + 1)

    # 复制粘贴嫌疑：文件极少但代码量巨大（生成/粘贴代码，非真实工作）
    X["copy_paste"] = ((X.num_source_files <= 3) & (X.lines_of_code > 3000)).astype(float)

    # README 模板嫌疑：README 词数相对代码量过高（模板 README 很长但项目很小）
    X["readme_to_loc"] = div(X.readme_word_count, X.lines_of_code + 1)

    # 文件组织度：文件多但每个小 = 精心组织；文件少但每个大 = 堆砌
    X["files_per_loc"] = div(X.num_source_files, X.lines_of_code + 1)

    X = X.drop(columns=DROP_COLS, errors="ignore")
    return X


def add_group_features(X_tr, X_va):
    """用训练集的分组中位数算相对特征，防止数据泄露"""
    X_tr, X_va = X_tr.copy(), X_va.copy()
    for grp in ["primary_language", "project_category"]:
        for col in ["lines_of_code", "active_commit_days", "lines_deleted_total"]:
            med = X_tr.groupby(grp)[col].median()
            global_med = X_tr[col].median()
            X_tr[f"{col}_vs_{grp}"] = X_tr[col] / (X_tr[grp].map(med).fillna(global_med) + 1)
            X_va[f"{col}_vs_{grp}"] = X_va[col] / (X_va[grp].map(med).fillna(global_med) + 1)
    return X_tr, X_va


def build_features(train, test):
    """训练/推理共用的特征构建入口。

    train: 含 TARGET 列的训练 DataFrame；test: 不含 TARGET 的 DataFrame。
    分组中位数只从 train 计算，防止泄露。
    """
    feature_cols = [c for c in test.columns if c != ID and c in train.columns]
    X_tr_base = prepare_base(train[feature_cols])
    X_te_base = prepare_base(test[feature_cols]).reindex(columns=X_tr_base.columns)
    return add_group_features(X_tr_base, X_te_base)
