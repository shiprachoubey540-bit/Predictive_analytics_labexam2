
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('C:\\Users\\Shipra\\OneDrive\\Desktop\\predictive_analytics2\\Lab_Exam_binary_classification_dataset.csv')
df = df.dropna(subset=[df.columns[-1]])# Drop rows with missing values in the target variable
print(df.isnull().sum())
print(df.info())


#class distribution
sns.countplot(x=df.iloc[:, -1])
plt.title("Class Distribution")
plt.show()

#label encoding
df['Target'] = df['Target'].map({'No': 0, 'Yes': 1}).astype(int)


#feature visualization
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=df.iloc[:,-1], cmap='coolwarm')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Data Visualization")
plt.show()

#correlation matrix
sns.heatmap(df.corr(), annot=True)
plt.show()


#build classification model
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

#decision boundary visualization

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Select only 2 features
X = df.iloc[:, :2].values
y = df.iloc[:, -1].values

# Scale features (VERY IMPORTANT)
sc = StandardScaler()
X = sc.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Create meshgrid (SAFE step size)
X1, X2 = np.meshgrid(
    np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.1),
    np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.1)
)

# Predict on grid
Z = model.predict(np.array([X1.ravel(), X2.ravel()]).T)
Z = Z.reshape(X1.shape)

# Plot decision boundary
plt.contourf(X1, X2, Z, alpha=0.5, cmap=ListedColormap(('red', 'blue')))

# Plot actual points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(('red', 'blue')))

plt.title("Decision Boundary (Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


#model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#confusion matrix visualization
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()