import pandas as pd

# Load Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)

# Explore data
print("First 5 rows:")
print(iris.head())
print("\nSummary statistics:")
print(iris.describe())
print("\nAverage petal length:")
print(iris["petal_length"].mean())
print("\nAverage sepal length:")
print(iris["sepal_length"].mean())
print("\nSetosa rows (first 5):")
print(iris[iris["species"] == "setosa"].head())
print("\nAverage sepal width")
print(iris["sepal_width"].mean())
print("\nVersicolor rows (first 5):")
print(iris[iris["species"] == "versicolor"].head())
print("\ngroupy data")
print(iris.groupby("species")["sepal_width"].mean())