#Compute average sepal length, sepal width, and petal length per species.
#Plot sepal width histograms (as an alternative to petal length from previous).

import pandas as pd 
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)

#Compute average sepal length, sepal width, and petal length per species.
setosa_avg = iris[iris["species"] == "setosa"][["sepal_length","sepal_width","petal_length"]].mean()
versicolor_avg = iris[iris["species"] == "versicolor"][["sepal_length","sepal_width","petal_length"]].mean() 
virginica_avg = iris[iris["species"] == "virginica"][["sepal_length","sepal_width","petal_length"]].mean()  
print ("\n AVG for Setosa")
print(setosa_avg)
print ("\n AVG for versicolor")
print(versicolor_avg)
print ("\n AVG for virginica")
print(virginica_avg)

# Or either is this method
avg_all = iris.groupby("species")[["sepal_length","sepal_width","petal_length"]].mean()
print (avg_all)

#Plot sepal length histograms (as an alternative to petal length from previous).
plt.figure (figsize=(8,6))
for species in iris ["species"].unique():
    subset = iris[iris["species"] == species]
    plt.hist(subset["sepal_length"], alpha=0.5, label=species)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("count")
plt.title("Sepal Length Distribution by Species")
plt.legend()
plt.grid(True)
plt.show()

#Plot sepal width histograms (as an alternative to petal length from previous).
plt.figure (figsize = "10,6")
for species in iris ["species"].unique():
    subset = iris[iris["species"] == species]
    plt.hist(subset["sepal_width"], alpha=0.5, label=species)
plt.xlabel("Sepal width")
plt.ylabel("count")
plt.title("Sepal width distribution by Species")
plt.legend()
plt.grid(True)
plt.show()