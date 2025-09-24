from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd


#linear regression to predict sepal length from width in Iris dataset
 
#Load iris
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)

#prepare data
X = iris[["sepal_width"]]
y = iris["sepal_length"]

#Split data(80% train 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train model
model = LinearRegression()
model.fit(X_train, y_train)

#Predict and evaluate 
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squard Error:{mse}")
print(f"Model slope: {model.coef_[0]}")
print(f"Model intercept: {model.intercept_}")
