import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Load dataset
data = Dataset("dataset/Abalone.csv")
df = data.transactions

# Encode categorical 'Sex' column if present
if 'Sex' in df.columns:
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# Convert age (Rings) into categories (e.g., young, middle-aged, old)
df['AgeCategory'] = pd.cut(df['Rings'], bins=[0, 8, 11, 30], labels=[0, 1, 2])  # Adjust bins if needed
df.drop(columns=['Rings'], inplace=True)

# Features and target
X = df.drop(columns=['AgeCategory'])
y = df['AgeCategory']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optimize KNN using GridSearchCV
param_grid = {'n_neighbors': range(1, 20, 2)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Best K
best_k = grid_search.best_params_['n_neighbors']
print(f"Best k: {best_k}")

# Train optimized KNN
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Accuracy: {accuracy:.2f}")
