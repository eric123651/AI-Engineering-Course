from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.metrics import r2_score 

# Setosa-only regression: predict sepal length from sepal width

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)

#Fliter to setosa 
iris_setosa = iris[iris["species"] == "setosa"]
print(f"Dataset size: {len(iris_setosa)}rows")

X = iris_setosa[["sepal_width"]]
y = iris_setosa["sepal_length"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error:{mse}")
print(f"Model coef:{model.coef_[0]}")
print(f"Model inyercept:{model.intercept_}")
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2:3f}")