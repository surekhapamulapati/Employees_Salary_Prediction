import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import joblib

# Load dataset
df = pd.read_csv('data.csv')

# Features to use
features = ['Experience', 'Education', 'Role', 'Location', 'Skills', 'Department',
            'JobLevel', 'Certifications', 'RemoteWork']
target = 'Salary'

# One-hot encode categorical features
df_encoded = pd.get_dummies(df[features])
REQUIRED_FEATURES = df_encoded.columns.tolist()

# Save for app usage
joblib.dump(REQUIRED_FEATURES, 'required_features.pkl')

# Scale numeric features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Train KNN model
y = df[target]
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_scaled, y)

# Save model and scaler
joblib.dump(knn, 'model.pkl')
joblib.dump(scaler, 'encoder_scaler.pkl')
