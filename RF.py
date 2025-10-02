# rf_prior.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

def train_rf_prior(data_path, target_col="y", test_size=0.2, random_state=42, save_model=True):
    print("正在读取数据...")
    df = pd.read_excel(data_path) if data_path.endswith(".xlsx") else pd.read_csv(data_path)
    print(f"数据读取成功: {df.shape}")

    # Separating features and labels
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Split training/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Defining the RF Model
    rf = RandomForestRegressor(
        n_estimators=300,   # Number of trees
        max_depth=None,     # No depth limit
        random_state=random_state
    )

    print("正在训练 RF 模型...")
    rf.fit(X_train, y_train)

    # predict
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    # Calculation indicators
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("\n===== RF 模型结果 =====")
    print(f"训练集 R²: {train_r2:.4f}")
    print(f"测试集  R²: {test_r2:.4f}")
    print(f"测试集 RMSE: {test_rmse:.4f}")

    if save_model:
        joblib.dump(rf, "rf_prior_model.pkl")
        print("模型已保存为 rf_prior_model.pkl")

    return rf, (train_r2, test_r2, test_rmse)


if __name__ == "__main__":
    data_path = "D:/PycharmProject/Train_Data.xlsx"
    target_col = "Irregularity"
    train_rf_prior(data_path, target_col)

