import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

TARGET = "hours_spent"
ID = "project_id"
HOURS_CAP = 140.7

def main():
    test = pd.read_csv("test.csv")

    predictor = TabularPredictor.load("./models")

    # 与训练时保持一致：去掉 ID；训练特征里没有的列 AutoGluon 会自动忽略
    X_test = test.drop(columns=[ID])

    log_pred = predictor.predict(X_test)
    pred = np.clip(np.expm1(log_pred), 0.1, HOURS_CAP)

    out = pd.DataFrame({ID: test[ID].values, TARGET: pred})
    out.to_csv("predictions.csv", index=False)
    print(f"wrote predictions.csv  rows={len(out)}")

if __name__ == "__main__":
    main()