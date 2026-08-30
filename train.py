import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading Dataset...")
df = pd.read_csv("Telecom_Tower_Failure_Dataset_10000-1.csv")

# Identify the target column name dynamically
# In this dataset, the target column is 'Failure_Within_48Hrs' (or adjust if named differently)
if "Failure_Within_48Hrs" in df.columns:
    target_col = "Failure_Within_48Hrs"
elif "hardware_failure" in df.columns:
    target_col = "hardware_failure"
else:
    target_col = df.columns[-1]  # Selects the last column by default

print(f"Target column selected: {target_col}")

# Separate features (X) and target label (y)
X = df.drop(columns=[target_col])
y = df[target_col]

# Split into 75% training and 25% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("Training Model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model performance
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save the trained model artifact as model.pkl (matching buildspec.yml)
joblib.dump(model, "model.pkl")
print("Training Completed Successfully and model.pkl saved!")