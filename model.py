import math
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import BaseEnsemble

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data slipt proportions
STATE = 42
TRAIN = 0.66
TEST = 0.34

# Train a specific model adjust to reservoir data
def train_reservoir_model(df: pd.DataFrame, reservoir: str) -> BaseEnsemble:
    # Obtain data from reservoir
    df = df[df["lake"] == reservoir]

    # Process dataset
    x_train, x_test, y_train, y_test = train_test_split(df[["snap_chla"]], df["real_chla"], train_size=TRAIN, random_state=STATE)

    # Variables to optimize
    min_metric = float('inf')
    best_lr = None
    best_estimators = None
    best_model = None
    best_mae = None

    # Test with differents estimators and learning rates 
    for estimators in range(1, 100):
        for lr in [i/1000 for i in range(1, 1000, 10)]:
            model = GradientBoostingRegressor(n_estimators=estimators, learning_rate=lr, random_state=STATE)
            model.fit(x_train, y_train)

            # Calculate upgrade of this batch
            model_mae = mean_absolute_error(y_test, model.predict(x_test))
            if model_mae < min_metric:
                min_metric = model_mae
                best_lr = lr
                best_model = model
                best_estimators = estimators
                best_mae = mean_absolute_error(y_test, model.predict(x_test))

    print(f"Best model adjust to {reservoir} with MAE: {best_mae}, LR: {best_lr} and estimators: {best_estimators}")
    return best_model


if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("dataset_adjusted.csv", sep=';')

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(df[["snap_chla"]], df["real_chla"], train_size=TRAIN, random_state=STATE)

    # Build VotingEnsemble with GradientBoostingRegressor
    models = []
    for reservoir, lr, estimators in [("alange", 0.731, 1), ("orellana", 0.001, 1), ("villar_del_rey", 0.501, 1), ("tentudia", 0.721, 3), ("zujar", 0.491, 2)]:
        models.append((reservoir, GradientBoostingRegressor(n_estimators=estimators, learning_rate=lr, random_state=STATE)))
    model = VotingRegressor(models)

    # Train VotingEnsemble
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    
    # Show metrics
    r_pearson, pvalue = pearsonr(y_test, y_predict)
    print(f"--- Voting Ensemble with GradientBoostingRegressor ---\n" \
          f"MAE: {mean_absolute_error(y_test, y_predict)}\n" \
          f"RSME: {math.sqrt(mean_squared_error(y_test, y_predict))}\n" \
          f"R2: {r2_score(y_test, y_predict)}\n" \
          f"r Pearson: {r_pearson}\n" \
          f"p-value: {pvalue}")

    # Build Dataframe to represent data
    df_gradient = pd.DataFrame({
        "x": x_test.iloc[:, 0].values,
        "y_true": y_test.values,
        "y_pred": y_predict
    }).sort_values("x")

    df_gradient.to_csv("plot_data.csv", sep=',', index=False)

    # Show results
    plt.figure(figsize=(8, 5))
    plt.plot(df_gradient["x"], df_gradient["y_true"], label="Real values", marker="o", linestyle="-")
    plt.plot(df_gradient["x"], df_gradient["y_pred"], label="Predicted values with GradientBoosting", marker="x", linestyle="--")
    plt.xlabel("SNAP processed chl-a values")
    plt.ylabel("Real measured chl-a values")
    plt.title("Comparison between real data and model behavior on test data")
    plt.legend()
    plt.grid(True)
    plt.show()
