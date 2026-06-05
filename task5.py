# ===============================
# TASK 5: Decision Trees & Random Forests
# ===============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
# =========================
# 2. LOAD DATASET
# =========================
df = pd.read_csv("heart.csv")

print("Dataset Shape:", df.shape)
print(df.head())      # =========================
# 3. SPLIT FEATURES & TARGET
# =========================
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# =========================
# 4. DECISION TREE CLASSIFIER
# =========================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\n=== Decision Tree Accuracy ===")
print(accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
# =========================
# 5. VISUALIZE DECISION TREE
# =========================
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
# =========================
# 6. CONTROL OVERFITTING (PRUNING)
# =========================
dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_pruned.fit(X_train, y_train)

y_pred_pruned = dt_pruned.predict(X_test)

print("\n=== Pruned Tree Accuracy (max_depth=4) ===")
print(accuracy_score(y_test, y_pred_pruned))
# =========================
# 7. RANDOM FOREST CLASSIFIER
# =========================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n=== Random Forest Accuracy ===")
print(accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
# =========================
# 8. FEATURE IMPORTANCE
# =========================
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()
# =========================
# 9. CROSS VALIDATION
# =========================

dt_scores = cross_val_score(dt, X, y, cv=5)
rf_scores = cross_val_score(rf, X, y, cv=5)

print("\n=== Cross Validation Results ===")
print("Decision Tree CV Accuracy:", dt_scores.mean())
print("Random Forest CV Accuracy:", rf_scores.mean())
