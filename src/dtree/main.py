from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in master dataset
df = pd.read_csv("../../dataset.csv")

# get rid of first column
df = df.iloc[:, 1:]

# split up features from target
X = df.iloc[:, :-1]
Y = df.iloc[:, -1:]


# comment/uncomment these two lines out depending on what tree you want
# X = X.drop(columns="cases")
# X = X.drop(columns="deaths")

# split into train and test using kfold
kf = StratifiedKFold(n_splits=5)

train_indices = []
test_indices = []
scores = []
feature_importance = []

for train, test in kf.split(X, Y):
    train_indices.append(train)
    test_indices.append(test)

total_tn = 0
total_fp = 0
total_fn = 0
total_tp = 0

for i in range(len(train_indices)):
    dt = tree.DecisionTreeClassifier(criterion="entropy")
    x_train = X.iloc[train_indices[i]]
    x_test = X.iloc[test_indices[i]]
    y_train = Y.iloc[train_indices[i]]
    y_test = Y.iloc[test_indices[i]]

    dt.fit(x_train, y_train)

    y_pred = dt.predict(x_test)
    row1, row2 = confusion_matrix(y_test, y_pred)
    total_tn += row1[0]
    total_fp += row1[1]
    total_fn += row2[0]
    total_tp += row2[1]

    disp = plot_confusion_matrix(dt, x_test, y_test,
                                 cmap=plt.cm.Blues)
    plt.show()

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    scores.append(dt.score(x_test, y_test))
    feature_importance.append(dt.feature_importances_)
    print()


# get the avg of each feature's importance for all the k folds
avg_importance = [np.mean(k) for k in zip(*(row for row in feature_importance))]
features = list(X.columns)

print("true negatives = " + str(total_tn / 5))
print("true positives = " + str(total_tp / 5))
print("false negatives = " + str(total_fn / 5))
print("false positives = " + str(total_fp / 5))

plt.bar(features, avg_importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()