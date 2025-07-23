import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Updated sample dataset with extra columns
data = pd.DataFrame({
    'Experience': [1, 3, 5, 7, 10],
    'Education': ['Bachelors', 'Masters', 'PhD', 'Masters', 'Bachelors'],
    'Role': ['Developer', 'Manager', 'Developer', 'Analyst', 'Manager'],
    'Location': ['Hyderabad', 'Bangalore', 'Mumbai', 'Pune', 'Delhi'],
    'Skills': ['Python', 'Excel', 'Java', 'SQL', 'Python'],
    'Department': ['IT', 'Finance', 'IT', 'Marketing', 'HR'],
    'JobLevel': ['Junior', 'Senior', 'Mid', 'Junior', 'Senior'],
    'Certifications': ['Yes', 'No', 'No', 'Yes', 'Yes'],
    'RemoteWork': ['No', 'Yes', 'Yes', 'No', 'No'],
    'Salary': [30000, 50000, 80000, 60000, 90000]
})

# One-hot encoding
df_encoded = pd.get_dummies(data.drop('Salary', axis=1))
X = df_encoded
y = data['Salary']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'encoder_scaler.pkl')
print("✅ Model and scaler saved with extended features.")
