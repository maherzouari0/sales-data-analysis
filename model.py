import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. Load Data ────────────────────────────────────────────
df = pd.read_csv("superstore.csv")

# Clean column names
df.columns = df.columns.str.strip()

# ── 2. Data Cleaning ────────────────────────────────────────
df = df.dropna(subset=["Sales", "Category", "Sub-Category", "Region"])

# ── 3. Feature Engineering ──────────────────────────────────
df = pd.get_dummies(df, columns=["Category", "Sub-Category", "Region"], drop_first=True)

X = df.drop(columns=["Sales"])
y = df["Sales"]

# Keep only numeric columns
X = X.select_dtypes(include=["number"])

# ✅ REMOVE ANY REMAINING NaN
X = X.fillna(0)

# ── 4. Train/Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 5. Train Model ──────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── 6. Evaluate ─────────────────────────────────────────────
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("── Model Evaluation ──────────────────────────")
print(f"R² Score : {r2:.4f}")
print(f"MSE      : {mse:.2f}")
print(f"RMSE     : {rmse:.2f}")
print("──────────────────────────────────────────────")

# ── 7. Visualization ────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, alpha=0.4)

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# ── 8. Business Insights ────────────────────────────────────
print("\n📊 Key Insights:")
print("- Sales are influenced by product category and region.")
print("- Model provides a baseline for sales prediction using categorical features.")
print("- Performance can be improved with additional variables (discount, quantity, etc.).")

print("\n✅ Model complete. Chart saved.")
