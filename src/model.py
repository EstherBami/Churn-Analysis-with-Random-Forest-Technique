import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

df = pd.read_csv('data\data_for_modelling.csv')

# Make a copy of the data
train_data = df.copy()

# Separate target variable from independent variables
y = df['churn']
X = df.drop(columns=['Unnamed: 0', 'id', 'churn'])
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model training
class_weight = {0: 1, 1: 10}  # Weight for churn (1) is 10 times higher than retained (0)
model = RandomForestClassifier(class_weight=class_weight, random_state=42) 
classifier = model.fit(X_train, y_train)

# Predictions

y_pred = classifier.predict(X_test)

y_pred[:5]

# Accuracy: defined as the ratio of the no. of correct predictions to the total no. of prediction
print('Accuracy:', accuracy_score(y_test,y_pred))

# Precision
print('Precision:', precision_score(y_test,y_pred,average='weighted'))

# Recall
print('Recall:', recall_score(y_test,y_pred,average='weighted'))

# F1 score: mean of precision and recall
print('F1 score:', f1_score(y_test,y_pred,average='weighted'))

# Area under curve
auc=np.round(roc_auc_score(y_test,y_pred),3)

print('AUC:',auc)

# Confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
print("confusion_matrix:", confusion_matrix)