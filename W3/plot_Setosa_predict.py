from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
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

#Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7, color='green', s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
plt.xlabel("Actual Sepal Length")
plt.ylabel("Predicted Sepal Length")
plt.title("Setosa: Actual vs. Predicted Sepal Length")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()





