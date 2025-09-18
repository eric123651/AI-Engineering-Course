import pandas as pd
import matplotlib.pyplot as plt 

#load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)

#Scatter plot of petal length vs petal width 
plt.scatter(iris["petal_length"], iris["petal_width"], c=iris["species"].map({"setosa": "red", "versicolor" :"blue", "virginica" :"green"}))
plt.xlabel("Petal Length (cm)") 
plt.ylabel("Petal Width (cm)") 
plt.title("Iris Dataset: Petal Length vs. Petal Width")
plt.legend(["setosa", "versicolor", "virginica"])
plt.grid(True)
plt.show()
