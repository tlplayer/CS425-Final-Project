import csv
import math
import numpy as np
import pandas as pd
import pylab as pl
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Suppress warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

# Calculate finer-tuned parameters
def get_fine_tuned_parameters(params, step = 0.2):
  tuned_params = {}
  for key in params:
    if key == 'kernel':
      tuned_params[key] = [params[key]]
      continue
    val = params[key]
    tuned_params[key] = []
    step_val = step * val
    for k in range(-4, 5):
      tuned_params[key].append(val + (step_val * k))
  return tuned_params


# Read in dataset
df = pd.read_csv('../../dataset.csv')

# Drop index column
df = df.iloc[:, 1:]

# Separate features and target
X = df.iloc[:, :-1]
Y = df.iloc[:, -1:]

# Standardize the features
X = pd.DataFrame(data=StandardScaler().fit_transform(X))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=0)
y_train_converted = y_train.values.ravel()

tuned_parameters = [
  {
    'kernel': ['linear'], 
    'C': [1, 10, 100, 1000]
  },
  {
    'kernel': ['poly'], 
    'degree': [2, 3, 4],
    'C': [1, 10, 100, 1000]
  },
  {
    'kernel': ['rbf'], 
    'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
    'C': [1, 10, 100, 1000]
  }
]

scores = ['precision', 'recall']

for score in scores:
  tested_params = []
  clf = GridSearchCV(
    SVC(), tuned_parameters, scoring='%s_macro' % score)
  clf.fit(x_train, y_train_converted)

  means = clf.cv_results_['mean_test_score']
  stds = clf.cv_results_['std_test_score']
  for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    obj = {
      'best': 0,
      'score': mean,
      'std': 2 * std,
      'params': params
    }
    if params == clf.best_params_:
      obj['best'] = 1
    tested_params.append(obj)

  # Fine grid param calculation
  fine_tuned_parameters = get_fine_tuned_parameters(clf.best_params_)

  # Fine grid search
  clf = GridSearchCV(
      SVC(), fine_tuned_parameters, scoring='%s_macro' % score
  )
  clf.fit(x_train, y_train)

  means = clf.cv_results_['mean_test_score']
  stds = clf.cv_results_['std_test_score']
  for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    obj = {
      'best': 0,
      'score': mean,
      'std': 2 * std,
      'params': params
    }
    if params == clf.best_params_:
      obj['best'] = 2
    tested_params.append(obj)

  print('##########################################')
  print('##       Best params for {: <14} ##'.format(score))
  print('##########################################')
  print('  Score:  {:.3f}'.format(clf.best_score_ * 100))
  print('  Params: {}'.format(clf.best_params_))
  print()

  y_true, y_pred = y_test, clf.predict(x_test)
  print(classification_report(y_true, y_pred, digits=4))
  print()

  print('# All parameters tested:')
  for test in tested_params:
    prefix = ''
    if test['best'] == 2:
      prefix = ' ** '
    elif test['best'] == 1:
      prefix = '  * '
    else:
      prefix = '    '
    print('{}{:.3f} (+/-{:.03f}) for {}'.format(prefix, test['score'] * 100, test['std'], test['params']))
  print()