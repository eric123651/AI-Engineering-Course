#Predict petal length using both petal width AND sepal width
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)

X = iris[["petal_width","sepal_width" ]]
y = iris["petal_length"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error:{mse}")
print(f"Model Coef:{model.coef_}")
print(f"Model intercept:{model.intercept_}")

#Mean squared error:0.2133167123375325
#Model Coef:[ 2.16601484 -0.34526279]
#Model intercept:2.2206286857953654