from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score #cross validation
import pandas as pd
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)

X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = iris["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print("\n==CROSS-VALIDATION ANALYSIS")


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print (f"Accuracy:{accuracy}")
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("Sample predictions:")
print (f"Actual:{y_test.iloc[:5].values}")
print(f"Predicted: {y_pred[:5]}")