# House_price_Predictio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

df.head()
 df.info()

df.describe()

df.isnull().sum()
 df.hist(figsize=(12,10))
plt.show()
X = df.drop("Price", axis=1)

y = df["Price"]
: X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()

model.fit(X_train, y_train)

df.info()

df.describe()

df.isnull().sum()
df.hist(figsize=(12,10))
plt.show()
 X = df.drop("Price", axis=1)

y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()

model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted")

plt.show()
 import pickle

pickle.dump(
    model,
    open("house_price_model.pkl", "wb")
)

print("Model Saved Successfully")
