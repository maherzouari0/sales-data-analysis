import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. Load Data ────────────────────────────────────────────
df = pd.read_csv("superstore.csv")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Sales", "Category", "Sub-Category", "Region"])

# ── 2. Feature Engineering ──────────────────────────────────
df_model = pd.get_dummies(df, columns=["Category", "Sub-Category", "Region"], drop_first=True)
X = df_model.drop(columns=["Sales"])
y = df_model["Sales"]
X = X.select_dtypes(include=["number"]).fillna(0)

# ── 3. Train/Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Train Model ──────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── 5. Evaluate ─────────────────────────────────────────────
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("── Model Evaluation ──────────────────────────")
print(f"R² Score : {r2:.4f}")
print(f"MSE      : {mse:.2f}")
print(f"RMSE     : {rmse:.2f}")
print("──────────────────────────────────────────────")

# ── 6. Actual vs Predicted ──────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, alpha=0.4, color="#e74c3c", s=20)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "k--", linewidth=1.5, label="Perfect Prediction")
plt.title("Actual vs Predicted Sales", fontsize=13)
plt.xlabel("Actual Sales ($)")
plt.ylabel("Predicted Sales ($)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# ── 7. Top 10 Feature Importances ───────────────────────────
importance = pd.Series(model.coef_, index=X.columns)
top_features = importance.abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
top_features.plot(kind="bar", color="#f39c12")
plt.title("Top 10 Features Influencing Sales", fontsize=13)
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

print("\n📊 Key Insights:")
print("- Sales are influenced by product category and region.")
print("- Model provides a baseline for sales prediction.")
print("- Performance can be improved with additional variables.")
print("\n✅ Model complete. Charts saved.")
