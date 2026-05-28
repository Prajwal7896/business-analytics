
# =========================================================
# ENTERPRISE E-COMMERCE ANALYTICS DASHBOARD
# =========================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
df = pd.read_csv("Ecommerce.csv")

sns.set_style("whitegrid")

# =========================================================
# VISUALIZATION 1
# REVENUE BY MARKETING CHANNEL
# =========================================================

plt.figure(figsize=(14, 6))

channel_revenue = df.groupby(
    'marketing_channel'
)['revenue'].mean().sort_values(ascending=False)

sns.barplot(
    x=channel_revenue.index,
    y=channel_revenue.values
)

plt.title(
    " Insight 1: Average Revenue by Marketing Channel",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel("Marketing Channel")
plt.ylabel("Average Revenue")

plt.show()

# =========================================================
# VISUALIZATION 2
# CONVERSION FUNNEL
# =========================================================

funnel = [
    len(df),
    df['added_to_cart'].sum(),
    df['purchased'].sum()
]

stages = [
    'Visitors',
    'Added To Cart',
    'Purchased'
]

plt.figure(figsize=(10, 5))

sns.barplot(
    x=stages,
    y=funnel
)

plt.title(
    " Insight 2: Customer Conversion Funnel",
    fontsize=18,
    fontweight='bold'
)

plt.ylabel("Users")

plt.show()

# =========================================================
# VISUALIZATION 3
# CART ABANDONMENT BY DEVICE TYPE
# =========================================================

plt.figure(figsize=(12, 6))

sns.barplot(
    x='device_type',
    y='cart_abandoned',
    data=df
)

plt.title(
    " Insight 3: Cart Abandonment by Device Type",
    fontsize=18,
    fontweight='bold'
)

plt.ylabel("Abandonment Rate")

plt.show()

# =========================================================
# VISUALIZATION 4
# DISCOUNT VS REVENUE
# =========================================================

plt.figure(figsize=(12, 6))

sns.scatterplot(
    x='discount_percent',
    y='revenue',
    data=df,
    alpha=0.6
)

plt.title(
    " Insight 4: Discount vs Revenue",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel("Discount Percent")
plt.ylabel("Revenue")

plt.show()

# =========================================================
# VISUALIZATION 5
# PURCHASE VS TIME ON SITE
# =========================================================

plt.figure(figsize=(12, 6))

sns.boxplot(
    x='purchased',
    y='time_on_site_sec',
    data=df
)

plt.title(
    " Insight 5: Purchase vs Time on Site",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel("Purchased")
plt.ylabel("Time on Site (sec)")

plt.show()

# =========================================================
# VISUALIZATION 6
# MONTHLY REVENUE TREND
# =========================================================

monthly_revenue = df.groupby(
    'visit_month'
)['revenue'].sum()

plt.figure(figsize=(14, 6))

sns.lineplot(
    x=monthly_revenue.index,
    y=monthly_revenue.values,
    marker='o'
)

plt.title(
    " Insight 6: Monthly Revenue Trend",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel("Month")
plt.ylabel("Revenue")

plt.show()

# =========================================================
# VISUALIZATION 7
# PRODUCT CATEGORY REVENUE
# =========================================================

plt.figure(figsize=(15, 6))

category_revenue = df.groupby(
    'product_category'
)['revenue'].sum().sort_values(ascending=False)

sns.barplot(
    x=category_revenue.index,
    y=category_revenue.values
)

plt.title(
    " Insight 7: Revenue by Product Category",
    fontsize=18,
    fontweight='bold'
)

plt.xticks(rotation=45)

plt.show()

# =========================================================
# VISUALIZATION 8
# CUSTOMER SEGMENT HEATMAP
# =========================================================

pivot = pd.pivot_table(
    df,
    values='revenue',
    index='user_type',
    columns='device_type',
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))

sns.heatmap(
    pivot,
    annot=True,
    fmt=".2f",
    cmap='coolwarm'
)

plt.title(
    " Insight 8: Customer Revenue Heatmap",
    fontsize=18,
    fontweight='bold'
)

plt.show()

# =========================================================
# VISUALIZATION 9
# TOP 20 CUSTOMERS
# =========================================================

top_customers = df.groupby(
    'customer_id'
)['revenue'].sum().sort_values(
    ascending=False
).head(20)

plt.figure(figsize=(15, 6))

top_customers.plot(kind='bar')

plt.title(
    " Insight 9: Top 20 Customers by Revenue",
    fontsize=18,
    fontweight='bold'
)

plt.ylabel("Revenue")

plt.show()

# =========================================================
#  VISUALIZATION 10
# PAYMENT METHOD RISK ANALYSIS
# =========================================================

plt.figure(figsize=(12, 6))

sns.barplot(
    x='payment_method',
    y='cart_abandoned',
    data=df
)

plt.title(
    " Insight 10: Cart Abandonment by Payment Method",
    fontsize=18,
    fontweight='bold'
)

plt.ylabel("Abandonment Rate")

plt.show()
