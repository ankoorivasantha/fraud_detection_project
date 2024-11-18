import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load the dataset
try:
    df = pd.read_csv('FastagFraudDetection.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset not found.")
    exit()

# Preprocessing
df['Fraud_indicator'] = df['Fraud_indicator'].map({'Fraud': 1, 'Not Fraud': 0})  # Encode target
categorical_cols = ['Vehicle_Type', 'Lane_Type']

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target (reduced set of features)
X = df[['Transaction_Amount', 'Amount_paid', 'Vehicle_Type', 'Lane_Type', 'Vehicle_Speed']]
y = df['Fraud_indicator']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Additional evaluation: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Save the model
with open('fraud_detection_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Trained model saved as 'fraud_detection_model.pkl'.")

# Save the label encoders for consistent encoding during predictions
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)
print("Label encoders saved as 'label_encoders.pkl'.")
