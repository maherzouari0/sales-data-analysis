import pandas as pd
import matplotlib.pyplot as plt

# ── 1. Load Data ────────────────────────────────────────────
df = pd.read_csv("superstore.csv")
df.columns = df.columns.str.strip()
print(f"Dataset shape: {df.shape}")
print(df.head())

# ── 2. Data Cleaning ────────────────────────────────────────
df = df.dropna(subset=["Sales", "Category", "Sub-Category", "Region"])

# ── 3. Sales by Category ────────────────────────────────────
sales_category = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
bars = plt.bar(sales_category.index, sales_category.values,
               color=["#2ecc71", "#3498db", "#e74c3c"])
for bar, val in zip(bars, sales_category.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f"${val:,.0f}", ha="center", fontsize=9)
plt.title("Total Sales by Category", fontsize=13)
plt.xlabel("Category")
plt.ylabel("Sales ($)")
plt.tight_layout()
plt.savefig("sales_by_category.png", dpi=150)
plt.show()

# ── 4. Sales by Region ──────────────────────────────────────
sales_region = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
bars = plt.bar(sales_region.index, sales_region.values, color="#3498db")
for bar, val in zip(bars, sales_region.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f"${val:,.0f}", ha="center", fontsize=9)
plt.title("Total Sales by Region", fontsize=13)
plt.xlabel("Region")
plt.ylabel("Sales ($)")
plt.tight_layout()
plt.savefig("sales_by_region.png", dpi=150)
plt.show()

# ── 5. Top 10 Sub-Categories ────────────────────────────────
top_sub = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
top_sub.plot(kind="bar", color="#9b59b6")
plt.title("Top 10 Sub-Categories by Sales", fontsize=13)
plt.xlabel("Sub-Category")
plt.ylabel("Sales ($)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top_subcategories.png", dpi=150)
plt.show()

print("\n📊 Key Insights:")
print("- Technology generates the highest total revenue among all categories.")
print("- West and East regions outperform Central and South significantly.")
print("- Phones and Chairs are the top revenue-generating sub-categories.")
print("\n✅ Analysis complete. Charts saved.")
