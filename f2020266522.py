# -*- coding: utf-8 -*-
"""F2020266522.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Oexc0PJ8Sgzg187krkYI19JwPoCU__Bz
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# User defined plotting function
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
      ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
    levels=[-1, 0, 1], alpha=0.5,
    linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
      ax.scatter(model.support_vectors_[:, 0],
      model.support_vectors_[:, 1],
      s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)











"""## Perfectly Linearly Separable Dataset"""

from sklearn import datasets
X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=0.6, random_state=0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter');
print("Shape of X array: ",X.shape)
print("Shape of y array: ",y.shape)











from sklearn.svm import SVC
model = SVC(kernel='linear', C=0.1)
model.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter')
plot_svc_decision_function(model)









from sklearn.svm import SVC
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter')
plot_svc_decision_function(model)













"""## Almost Linearly Separable Dataset"""

from sklearn import datasets
X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.2, random_state=0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter');
print("Shape of X array: ",X.shape)
print("Shape of y array: ",y.shape)











# With a small value of C, the model will ignore classification errors
model = SVC(kernel='linear', C=0.01)
model.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter')
plot_svc_decision_function(model)











model = SVC(kernel='linear', C=100)
model.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter')
plot_svc_decision_function(model)











"""## What about Not Linearly Separable Dataset?"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X, y = datasets.make_circles(100, factor=.4, noise=.2, random_state=54)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='bwr')
df = pd.DataFrame(dict(x1=X[:, 0], x2=X[:, 1], y=y))
df









"""## SVC Linear Kernel"""

model = SVC(kernel='linear')
model.fit(X, y)
y_pred = model.predict(X)
print("Accuracy score: ", accuracy_score(y, y_pred))
print("F1 score: ", f1_score(y, y_pred))
ConfusionMatrixDisplay.from_predictions(y, y_pred);









from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X, y=y, clf=model)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Support Vector Classifier with Linear Kernel')
plt.show();













"""## SVC Polynomial Kernel"""

model = SVC(kernel='poly', degree=2) # Try other values of degree
model.fit(X, y)
y_pred = model.predict(X)
print("Accuracy score: ", accuracy_score(y, y_pred))
print("F1 score: ", f1_score(y, y_pred))
ConfusionMatrixDisplay.from_predictions(y, y_pred);









from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X, y=y, clf=model)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Support Vector Classifier with Polynomial Kernel')
plt.show();













"""## SVC with RBF Kernel"""

model = SVC(kernel='rbf', gamma=50)
model.fit(X, y)
y_pred = model.predict(X)
print("Accuracy score: ", accuracy_score(y, y_pred))
print("F1 score: ", f1_score(y, y_pred))
ConfusionMatrixDisplay.from_predictions(y, y_pred);











from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X, y=y, clf=model)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Support Vector Classifier with RBF Kernel')
plt.show();









"""---   
 <img align="left" width="75" height="75"  src="https://upload.wikimedia.org/wikipedia/commons/c/c8/Umt_logo.png">

<h1 align="center">Department of Computer Science</h1>
<h1 align="center">Course: Machine Learning</h1>

---
<h3><div align="right">Instructor: Hafiz Abdul Rehman</div></h3>

<h1 align="center">Assignment 3: SVM Kernel Selection</h1>
<h1 align="center">Submitted by: F2020266522</h1>
"""

import pandas as pd
data = pd.read_csv('User_Data.csv')

print(data.head())
print(data.info())
print(data.describe())

print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['Purchased'] = encoder.fit_transform(data['Purchased'])

from sklearn.preprocessing import LabelEncoder, StandardScaler

encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])


data['Purchased'] = encoder.fit_transform(data['Purchased'])


scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('Purchased', axis=1))

print(scaled_features[:5])

from sklearn.model_selection import train_test_split


X = scaled_features
y = data['Purchased']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['linear', 'rbf', 'poly']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

best_model = grid.best_estimator_
best_model.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = best_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib
joblib.dump(best_model, 'svm_model.pkl')