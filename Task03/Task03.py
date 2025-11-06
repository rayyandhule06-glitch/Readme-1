# Decision Tree Classifier on Bank Marketing Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset (replace with your dataset path)
DATA_PATH = "bank.csv"
df = pd.read_csv(DATA_PATH)
print("Dataset Loaded:", df.shape)

# Check for missing values
print(df.isnull().sum())

# Split features and target
X = df.drop(columns=["y"])
y = df["y"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify column types
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Model pipeline
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', DecisionTreeClassifier(random_state=42, max_depth=5))
])

# Train
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importances
model = clf.named_steps['model']
ohe = clf.named_steps['preprocessor'].named_transformers_['cat']
encoded_cols = ohe.get_feature_names_out(cat_cols)
all_features = np.concatenate([num_cols, encoded_cols])
importances = model.feature_importances_

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=all_features)
plt.title("Feature Importances (Decision Tree)")
plt.tight_layout()
plt.show()

# Visualize Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(model, filled=True, feature_names=all_features, class_names=model.classes_, rounded=True)
plt.title("Decision Tree (Depth=5)")
plt.show()
