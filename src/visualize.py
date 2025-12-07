import matplotlib as plt

def plot_xgboost(y_test, preds):
    plt.scatter(y_test, preds)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("XGBoost Prediction")
    plt.show()