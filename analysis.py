import pandas as pd
import matplotlib.pyplot as plt

# ── 1. Load Data ────────────────────────────────────────────
df = pd.read_csv("superstore.csv")

# Clean column names
df.columns = df.columns.str.strip()

# ── 2. Basic Info ───────────────────────────────────────────
print(f"Dataset shape: {df.shape}")
print(df.head())

# ── 3. Data Cleaning ────────────────────────────────────────
df = df.dropna(subset=["Sales", "Category", "Sub-Category", "Region"])

# ── 4. Sales by Category ────────────────────────────────────
sales_category = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sales_category.plot(kind="bar")
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales ($)")
plt.tight_layout()
plt.savefig("sales_by_category.png", dpi=150)
plt.show()

# ── 5. Sales by Region ──────────────────────────────────────
sales_region = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(8, 5))
sales_region.plot(kind="bar")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales ($)")
plt.tight_layout()
plt.savefig("sales_by_region.png", dpi=150)
plt.show()

# ── 6. Top Sub-Categories ───────────────────────────────────
top_sub = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
top_sub.plot(kind="bar")
plt.title("Top 10 Sub-Categories by Sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top_subcategories.png", dpi=150)
plt.show()

# ── 7. Business Insights ────────────────────────────────────
print("\n📊 Key Insights:")
print("- Certain categories dominate overall sales performance.")
print("- Regional differences highlight potential market opportunities.")
print("- A small number of sub-categories generate most revenue.")

print("\n✅ Analysis complete. Charts saved.")
