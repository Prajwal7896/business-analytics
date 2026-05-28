
import opendatasets as od
import pandas as pd
import numpy as np


od.download(
    "https://www.kaggle.com/datasets/kundanbedmutha/indian-e-commerce-customer-behavior-and-purchase"
)


df = pd.read_csv("Ecommerce.csv")

print("=" * 60)
print("DATASET SHAPE")
print(df.shape)

print("=" * 60)
print("DATASET INFO")
print(df.info())


df['visit_date'] = pd.to_datetime(
    df['visit_date'],
    format='%d-%m-%Y',
    dayfirst=True
)


df['visit_hour'] = df['visit_date'].dt.hour
df['visit_week'] = df['visit_date'].dt.isocalendar().week.astype(int)
df['visit_quarter'] = df['visit_date'].dt.quarter

df['is_weekend'] = (
    df['visit_weekday'].isin([5, 6])
).astype(int)


df.drop(columns=['visit_date'], inplace=True)


categorical_cols = [
    'device_type',
    'user_type',
    'marketing_channel',
    'product_category',
    'payment_method',
    'visit_season',
    'location',
    'session_duration_bucket'
]

numerical_cols = [
    'unit_price',
    'quantity',
    'discount_percent',
    'discount_amount',
    'pages_viewed',
    'time_on_site_sec',
    'rating',
    'review_helpful_votes'
]


before = len(df)

df.drop_duplicates(inplace=True)

after = len(df)

print("=" * 60)
print(f"Duplicates Removed: {before - after}")


for col in numerical_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.clip(
        df[col],
        lower,
        upper
    )


df['revenue_per_page'] = (
    df['revenue'] /
    (df['pages_viewed'] + 1)
)

df['engagement_score'] = (
    df['pages_viewed'] *
    df['time_on_site_sec']
)

df['discount_efficiency'] = (
    df['revenue'] /
    (df['discount_amount'] + 1)
)

df['cart_intent_score'] = (
    df['added_to_cart'] *
    df['pages_viewed']
)

df['session_value'] = (
    df['quantity'] *
    df['unit_price']
)

df['discount_sensitivity'] = (
    df['discount_percent'] *
    df['quantity']
)


df['customer_avg_revenue'] = df.groupby(
    'customer_id'
)['revenue'].transform('mean')

df['customer_purchase_frequency'] = df.groupby(
    'customer_id'
)['purchased'].transform('sum')


df['high_intent_user'] = (
    (
        df['pages_viewed'] > 10
    ) &
    (
        df['added_to_cart'] == 1
    )
).astype(int)


df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)


target = 'purchased'


leakage_cols = [
    'purchased',
    'cart_abandoned',
    'revenue_normalized'
]

X = df.drop(columns=leakage_cols)

y = df[target]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scale_cols = [
    'unit_price',
    'discount_amount',
    'pages_viewed',
    'time_on_site_sec',
    'revenue_per_page',
    'engagement_score',
    'session_value',
    'discount_sensitivity'
]

X_train[scale_cols] = scaler.fit_transform(
    X_train[scale_cols]
)

X_test[scale_cols] = scaler.transform(
    X_test[scale_cols]
)


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)


print("=" * 60)
print("FINAL TRAIN SHAPE")
print(X_train.shape)

print("=" * 60)
print("FINAL TEST SHAPE")
print(X_test.shape)

print("=" * 60)
print("TARGET DISTRIBUTION AFTER SMOTE")
print(y_train.value_counts())

df.to_csv(
    "cleaned_ecommerce_data.csv",
    index=False
)

print("=" * 60)
print("CLEANED DATASET SAVED SUCCESSFULLY")
print("=" * 60)

print("File Name: cleaned_ecommerce_data.csv")