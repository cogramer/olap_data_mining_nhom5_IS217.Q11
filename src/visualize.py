import matplotlib.pyplot as plt

def plot_xgboost(y_test, preds):
    errors = preds - y_test

    # Scatter colored by error
    plt.scatter(y_test, preds, c=errors, cmap="coolwarm", alpha=0.8)

    # Perfect prediction line (error = 0)
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label="Error = 0")

    plt.colorbar(label="Prediction Error")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("XGBoost Prediction Error Map")
    plt.legend()
    plt.show()