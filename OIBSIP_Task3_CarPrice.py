import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("car data.csv")

df.columns = df.columns.str.strip()

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nNull Values:")
print(df.isnull().sum())

df = df.drop_duplicates()

df["Car_Age"] = 2026 - df["Year"]

df["Brand"] = df["Car_Name"].str.split().str[0]

print("\nUpdated Dataset:")
print(df.head())


sns.histplot(df["Selling_Price"], kde=True)
plt.title("Selling Price Distribution")
plt.xlabel("Selling Price")
plt.ylabel("Number of Cars")
plt.show()


sns.boxplot(x="Fuel_Type", y="Selling_Price", data=df)
plt.title("Fuel Type vs Selling Price")
plt.show()


plt.scatter(df["Car_Age"], df["Selling_Price"])
plt.xlabel("Car Age")
plt.ylabel("Selling Price")
plt.title("Car Age vs Selling Price")
plt.show()


X = df[
    [
        "Brand",
        "Car_Age",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner"
    ]
]

y = df["Selling_Price"]


categorical = [
    "Brand",
    "Fuel_Type",
    "Seller_Type",
    "Transmission"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        )
    ],
    remainder="passthrough"
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model1 = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

print("\n----- Linear Regression -----")
print("MAE:", mean_absolute_error(y_test, pred1))
print("RMSE:", mean_squared_error(y_test, pred1) ** 0.5)
print("R2 Score:", r2_score(y_test, pred1))


model2 = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)

print("\n----- Random Forest Regression -----")
print("MAE:", mean_absolute_error(y_test, pred2))
print("RMSE:", mean_squared_error(y_test, pred2) ** 0.5)
print("R2 Score:", r2_score(y_test, pred2))


feature_names = model2.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = model2.named_steps[
    "model"
].feature_importances_

importance = pd.Series(
    importances,
    index=feature_names
).sort_values(
    ascending=False
).head(15)

importance.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Top Feature Importance - Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


residuals = y_test - pred2

plt.scatter(pred2, residuals)
plt.axhline(0)
plt.xlabel("Predicted Selling Price")
plt.ylabel("Residuals")
plt.title("Residual Plot - Random Forest")
plt.show()


r2_1 = r2_score(y_test, pred1)
r2_2 = r2_score(y_test, pred2)

print("\n----- Model Comparison -----")

if r2_1 >= r2_2:
    print("Best Model: Linear Regression")
    print("Reason: Linear Regression has the higher R2 Score.")
else:
    print("Best Model: Random Forest Regression")
    print("Reason: Random Forest Regression has the higher R2 Score.")