import csv
import math
import numpy as np
import pandas as pd
import pylab as pl
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Suppress warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

# Read in dataset
print('# Read in the dataset')
df = pd.read_csv('../../dataset.csv')

# Drop index column
df = df.iloc[:, 1:]

# Separate features and target
X = df.iloc[:, :-1]
Y = df.iloc[:, -1:]

# Standardize the features
print('# Standardize the features')
X = pd.DataFrame(data=StandardScaler().fit_transform(X))

# Perform a PCA
pca_features = ['PC 1', 'PC 2']

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
X = pd.DataFrame(data=principal_components, columns=pca_features)

# Split train/test data 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=0)
y_train_converted = y_train.values.ravel()

# Create and train the SVCs
print('# Create and train the SVCs')

clf_p = svm.SVC(C=0.6, gamma=0.06, kernel='rbf')
clf_r = svm.SVC(C=80, kernel='linear')

clf_p.fit(x_train, y_train_converted)
clf_r.fit(x_train, y_train_converted)

# Giant list of all coarse grid parameters
coarse_grid_params = [
  [
    {'C': 1, 'kernel': 'linear'},
    {'C': 10, 'kernel': 'linear'},
    {'C': 100, 'kernel': 'linear'},
    {'C': 1000, 'kernel': 'linear'},
    {'C': 1, 'degree': 2, 'kernel': 'poly'},
    {'C': 1, 'degree': 3, 'kernel': 'poly'},
    {'C': 1, 'degree': 4, 'kernel': 'poly'},
    {'C': 10, 'degree': 2, 'kernel': 'poly'},
    {'C': 10, 'degree': 3, 'kernel': 'poly'},
    {'C': 10, 'degree': 4, 'kernel': 'poly'},
    {'C': 100, 'degree': 2, 'kernel': 'poly'},
    {'C': 100, 'degree': 3, 'kernel': 'poly'},
    {'C': 100, 'degree': 4, 'kernel': 'poly'},
    {'C': 1000, 'degree': 2, 'kernel': 'poly'},
    {'C': 1000, 'degree': 3, 'kernel': 'poly'},
    {'C': 1000, 'degree': 4, 'kernel': 'poly'}
  ], [
    {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'},
    {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'},
    {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'},
    {'C': 1, 'gamma': 0.0001, 'kernel': 'rbf'},
    {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'},
    {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'},
    {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'},
    {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'},
    {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'},
    {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'},
    {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'},
    {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'},
    {'C': 1000, 'gamma': 0.1, 'kernel': 'rbf'},
    {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'},
    {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'},
    {'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'},
  ]
]

# Make giant plot with all coarse grid searches
# create a mesh to plot in
h=.02 # step size in the mesh
x_min, x_max = X[pca_features[0]].min()-1, X[pca_features[0]].max()+1
y_min, y_max = X[pca_features[1]].min()-1, X[pca_features[1]].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

pl.figure(figsize=(12, 8), dpi=140)
pl.set_cmap(pl.cm.Dark2)

print('## Making grid search plot')

for j in range(len(coarse_grid_params)):
  param_list = coarse_grid_params[j]
  for i, params in enumerate(param_list):
    params_title = 'C={}'.format(params['C'])
    if params['kernel'] == 'poly':
      params_title += '; Degree={}'.format(params['degree'])
    elif params['kernel'] == 'rbf':
      params_title += '; Gamma={}'.format(params['gamma'])
    title = '{} ({})'.format(params['kernel'].capitalize(), params_title)

    pl.subplot(4, 4, i+1)
    pl.axis('off')

    clf = svm.SVC(**params).fit(x_train, y_train_converted)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    pl.contourf(xx, yy, Z)
    pl.axis('tight')

    pl.scatter(X[pca_features[0]], X[pca_features[1]], c=Y.values, edgecolors='black')

    pl.title(title)
    print('### Added subplot ' + title)

  pl.axis('tight')
  pl.tight_layout()
  pl.savefig('grid-search-{}.png'.format(j+1), transparent=True)
