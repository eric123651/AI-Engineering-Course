from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score #cross validation
import pandas as pd

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)

X = iris[["petal_width", "petal_length", "sepal_width", "sepal_length"]]
y = iris["species"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model 1 Linear regression
print("====Linear Regression====")
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)


lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy:{lr_accuracy}")
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

#Model 2 Decision Tree
print("\n===DECISION TREE")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"Decision Tree Accuracy: {dt_accuracy}")
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

#Cross validation for LR reliably estimate model performance
print("\n==CROSS-VALIDATION ANALYSIS")
lr_cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring="accuracy")
print(f"Logistic Regression Cv Scores: {lr_cv_scores}")
print(f"LR CV Mean Accuracy:{lr_cv_scores.mean():.3f}")
print(f"LR CV Standard Deviation:{lr_cv_scores.std():.3f}")
print(f"LR CV RANGE:{lr_cv_scores.min():.3f}-{lr_cv_scores.max():.3f}")
#Cross validation for DT
dt_cv_scores = cross_val_score(dt_model, X, y, cv=5, scoring="accuracy")
print(f"Decision Tree Cv Scores: {dt_cv_scores}")
print(f"DT CV Mean Accuracy:{dt_cv_scores.mean():.3f}")
print(f"DT CV Standard Deviation:{dt_cv_scores.std():.3f}")
print(f"DT CV RANGE:{dt_cv_scores.min():.3f}-{dt_cv_scores.max():.3f}")

#Comparison
print("\n===Model Comparison")
print(f"Logistic Regression:{lr_accuracy:.3f}")
print(f"Decision Tree:{dt_accuracy:.3f}")
if lr_accuracy > dt_accuracy:
    print("Logistic wins!")
elif dt_accuracy > lr_accuracy:
    print("Decision Tree wins")
else:
    print("Tie Both perform equally")

print("\n===CV Comparison")
print(f"Logistic Regression CV: {lr_cv_scores.mean():.3f}+-{lr_cv_scores.std():.3f}")
print(f"Decision Tree CV: {dt_cv_scores.mean():.3f}+-{dt_cv_scores.std():.3f}")

if lr_cv_scores.mean() >dt_cv_scores.mean():
    print ("Logistic Regression is more robust")
elif dt_cv_scores.mean() > lr_cv_scores.mean():
    print("Decision Tree is more robust")
else:
    print("Both models are equally robust")
#CV gap >0.10 is a red flag.

#Confidence analysis LR
print("\n===PREDICTION CONFIDENCE ANALYSIS===")
lr_probabilities = lr_model.predict_proba(X_test)
print ("Logistic Regression - First 5 Test Samples:")
for i in range(5):
    print(f"Sample {i+1}: Actual ='{y_test.iloc[i]}', Predicted= '{lr_pred[i]}'")
    probs = lr_probabilities[i]
    print(f"Probabailities: Setosa={probs[0]:.3f}, Versicolor={probs[1]:.3f}, Virginica={probs[2]:.3f}")
    print(f"Max Confidence: {max(probs):.3f}")
    print()

#Confidence analysis DT
dt_probabilities = dt_model.predict_proba(X_test)
print("Decision Tree - First 5 Test Samples:")
for i in range(5):
    probs = dt_probabilities[i]
    print(f"Sample{i+1}: Acutal='{y_test.iloc[i]}', Predicted='{dt_pred[i]}")
    print(f"Probabilities: Setosa='{probs[0]:.3f}, Versicolor={probs[1]:.3f}, Virginica={probs[2]:.3f}")
    print(f"Max confidience:{max(probs):.3f}")
    print()