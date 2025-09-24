import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import joblib
import numpy as np

#Set up logging
logging.basicConfig(filename='model_deployment.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


"""Production-ready Iris clasifier with confidence evaluation."""

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)
X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = iris["species"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#train model
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

#Train evaluation
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print (f"Test Accuracy:{test_accuracy:.3f}({test_accuracy:.1%})\n")
print("Test confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nclassification report")
print (classification_report(y_test, y_pred))

#CV
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"\nCross-Validation Scores:{cv_scores}")
print(f"CV Mean Accuracy:{cv_scores.mean():.3f}({cv_scores.mean():.1%})")
print(f"CV Std Dev: {cv_scores.std():.3f}")
print(f"95% CI:{cv_scores.mean() - 2*cv_scores.std():.3f} - {cv_scores.mean() + 2*cv_scores.std():.3f}")

#Feature importance
print(f"\nFeature Importance (Absolute Coefiicients):")
coeficients =model.coef_[0]
for name, coef in zip(feature_names, coeficients):
    print (f"{name:15}:{abs(coef):.3f}")

#Coefficient analysis
print(f"\n===CONFIDENCE ANALYSIS(First 5 Samples)===")
probabilities = model.predict_proba(X_test)
for i in range(5):
    probs = probabilities[i]
    print(f"Sample {i+1}:Actual='{y_test.iloc[i]}', Predicted='{y_pred[i]}'")
    print(f" Probilities: Setosa={probs[0]:.3f}, Versicolor={probs[1]:.3f}, Virginica={probs[2]:.3f}")
    print(f"Max Confidence: {max(probs):.3f}")
    print()

#Save model
model_filename = "iris_production_classifier_v1.pk1"
joblib.dump(model, model_filename)
print(f"\nModel saved:{model_filename}")

#Test deploment with feature names
print("\n====Deployment Test")
new_flowers_data = {
    "sepal_length": [ 5.1, 6.0, 6.7],
    "sepal_width": [3.5, 2.7, 3.0],
    "petal_length": [1.4, 5.1, 5.2],
    "petal_width": [0.2, 1.6, 2.3]
    }
new_flowers_df = pd.DataFrame(new_flowers_data, index=["Setosa", "Versicolor", "Virginica"])
CONFIDENCE_THRESHOLD =0.9
for name in new_flowers_df.index:
    flower = new_flowers_df.loc[[name]]
    pred = model.predict(flower)[0]
    conf = model.predict_proba(flower)[0]
    max_conf = max(conf)
    print(f"{name:12}: Predicted='{pred}', Confidence={max(conf):.3f}")
    print(f"Probabilities: Setosa={conf[0]:.3f}, Versicolor={conf[1]:.3f}, Virginica={conf[2]:.3f}")
    if max_conf < CONFIDENCE_THRESHOLD:
        logging.warning(f"Low confidence prediction for {name}: {max_conf:.3f}, Predicted={pred}, Expected={name}")
    print()

#Check log file for warnings
with open ('model_deployment.log', 'r') as log_file:
    print ("\n===Deployment Log===")
    print(log_file.read())



