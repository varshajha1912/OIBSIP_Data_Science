import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load Dataset
df = pd.read_csv("Advertising.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nNull Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe())


# 2. Pairplot
sns.pairplot(df)
plt.show()


# 3. Scatter Plots

plt.scatter(df["TV"], df["Sales"])
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("TV Advertising vs Sales")
plt.show()

plt.scatter(df["Radio"], df["Sales"])
plt.xlabel("Radio Advertising")
plt.ylabel("Sales")
plt.title("Radio Advertising vs Sales")
plt.show()

plt.scatter(df["Newspaper"], df["Sales"])
plt.xlabel("Newspaper Advertising")
plt.ylabel("Sales")
plt.title("Newspaper Advertising vs Sales")
plt.show()


# 4. Correlation Heatmap
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# 5. Features and Target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]


# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7. Linear Regression
model1 = LinearRegression()
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

print("\n----- Linear Regression -----")
print("MAE:", mean_absolute_error(y_test, pred1))
print("RMSE:", mean_squared_error(y_test, pred1) ** 0.5)
print("R2 Score:", r2_score(y_test, pred1))


# 8. Random Forest Regression
model2 = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)

print("\n----- Random Forest Regression -----")
print("MAE:", mean_absolute_error(y_test, pred2))
print("RMSE:", mean_squared_error(y_test, pred2) ** 0.5)
print("R2 Score:", r2_score(y_test, pred2))


# 9. Feature Importance
importance = pd.Series(
    model2.feature_importances_,
    index=X.columns
)

importance.plot(kind="bar")
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()


# 10. Residual Plot
residuals = y_test - pred2

plt.scatter(pred2, residuals)
plt.axhline(0)
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# 11. Compare Models
r2_1 = r2_score(y_test, pred1)
r2_2 = r2_score(y_test, pred2)

print("\n----- Model Comparison -----")

if r2_1 >= r2_2:
    print("Best Model: Linear Regression")
    print("Reason: It has the higher R2 Score.")
else:
    print("Best Model: Random Forest Regression")
    print("Reason: It has the higher R2 Score.")