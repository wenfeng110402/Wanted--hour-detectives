"""训练 AutoGluon（good_quality）+ 手工特征工程。

用法:
    python train_ag.py

输出模型到 ./models，验证后替换 ./models。
"""

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from features import build_features, TARGET

MODEL_PATH = "./models"


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    X_tr, X_te = build_features(train, test)
    train_ag = X_tr.copy()
    train_ag[TARGET] = np.log1p(train[TARGET].values)

    predictor = TabularPredictor(
        label=TARGET,
        eval_metric="root_mean_squared_error",
        path=MODEL_PATH,
    ).fit(
        train_ag,
        presets="good_quality",
        time_limit=1800,
    )

    print(predictor.leaderboard(silent=True).to_string(index=False))

    log_pred = predictor.predict(X_te)
    pred = np.clip(np.expm1(log_pred), 0.1, 140.7)
    print("test pred stats:", pd.Series(pred).describe().to_dict())


if __name__ == "__main__":
    main()