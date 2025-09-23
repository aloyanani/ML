import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score


train_data = pd.read_csv("Titanic/Datasets/train_preprocessed.csv")
test_data = pd.read_csv("Titanic/Datasets/test_preprocessed.csv")

Passenger_id = test_data['PassengerId']
test_data = test_data.drop(['PassengerId'], axis=1)
train_data = train_data.drop(['PassengerId'], axis=1)

X = train_data.drop('Survived', axis=1)
Y = train_data['Survived']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.14, random_state=42)

# Create a Random Forest Classifier model
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=42) # 100 trees in the forest

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred_classifier = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_classifier)
print(f"Random Forest Classifier Accuracy: {accuracy}")

f1 = f1_score(y_test, y_pred_classifier) 
print(f"F1 Score : {f1}")

precision = precision_score(y_test, y_pred_classifier)
print(f"Precision Score: {precision}")

predictions = rf_classifier.predict(test_data)

output = pd.DataFrame({'PassengerId': Passenger_id, 'Survived': predictions})
output.to_csv('Titanic/submission.csv', index=False)


