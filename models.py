import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# 1. Load the Kaggle dataset
df = pd.read_csv('train.csv')

# 2. Select useful features and drop rows with missing values
df = df[['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'SalePrice']].dropna()

# 3. Define input (X) and target (y)
X = df[['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF']]
y = df['SalePrice']

# 4. Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 6. Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 7. Save model and scaler using pickle
with open('house_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model trained and saved successfully!")
