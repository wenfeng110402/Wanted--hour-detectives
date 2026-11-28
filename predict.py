import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from features import build_features, TARGET, ID

HOURS_CAP = 140.7


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # 分组中位数特征需要 train 作参照，与训练时保持一致
    _, X_test = build_features(train, test)

    # 裁判机 Python 版本可能与训练环境不同（3.12 vs 3.13），放宽版本断言
    predictor = TabularPredictor.load(
        "./models",
        require_version_match=False,
        require_py_version_match=False,
    )

    log_pred = predictor.predict(X_test)
    pred = np.clip(np.expm1(log_pred), 0.1, HOURS_CAP)

    out = pd.DataFrame({ID: test[ID].values, TARGET: pred})
    out.to_csv("predictions.csv", index=False)
    print(f"wrote predictions.csv  rows={len(out)}")


if __name__ == "__main__":
    main()