# This program creates a decision tree
# based on dataset.csv and creates multiple
# plots from it

# UNCOMMENT plt.savefig LINES IF YOU WANT TO PLOT

from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read in master dataset
df = pd.read_csv("../../dataset.csv")

# get rid of first column
df = df.iloc[:, 1:]

# split up features from target
X = df.iloc[:, :-1]
Y = df.iloc[:, -1:]


############################
#                          #
#   COMMENT/UNCOMMENT      #
#   BASED ON FEATURES      #
#   YOU WANT               #
#                          #
############################

X = X.drop(columns="cases")
X = X.drop(columns="deaths")



# split into train and test using kfold
kf = StratifiedKFold(n_splits=5)

# split training and test data
train_indices = []
test_indices = []

for train, test in kf.split(X, Y):
    train_indices.append(train)
    test_indices.append(test)

# initialize variables to keep up with data
classification_reports = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
    10: []
}

feature_importance = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
    10: []
}
total_tn = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0
}
total_fp = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0
}
total_fn = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0
}
total_tp = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0
}

scores_by_depth = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
    10: []
}


############################
#                          #
#  ACTUAL CLASSIFICATION   #
#                          #
############################

for i in range(len(train_indices)):

    # test different depths up to 5
    for j in range(10):
        dt = tree.DecisionTreeClassifier(criterion="entropy", max_depth=j+1)

        # set up variables
        x_train = X.iloc[train_indices[i]]
        x_test = X.iloc[test_indices[i]]
        y_train = Y.iloc[train_indices[i]]
        y_test = Y.iloc[test_indices[i]]

        # create the tree
        dt.fit(x_train, y_train)

        # make the prediction
        y_pred = dt.predict(x_test)

        # record the statistics
        row1, row2 = confusion_matrix(y_test, y_pred)
        total_tn[j+1] += row1[0]
        total_fp[j+1] += row1[1]
        total_fn[j+1] += row2[0]
        total_tp[j+1] += row2[1]
        scores_by_depth[j+1].append(dt.score(x_test, y_test))
    
        # note the feature importance of tree
        feature_importance[j+1].append(dt.feature_importances_)

        # get classification report
        classification_reports[j+1].append(classification_report(y_test, y_pred, output_dict=True, zero_division=0))


############################
#                          #
#   ACCURACY BY DEPTH      #
#                          #
############################

# let's see the accuracy score for each depth
accuracy_by_depth = []
for val in scores_by_depth.values():
    accuracy_by_depth.append(np.mean(val))

# let's plot a bar graph to show accuracy by depth
depth = [i+1 for i in range(10)]

plt.figure(figsize=(12.7,6.2))
plt.plot(depth, accuracy_by_depth)
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
if len(X.columns) < 17:
    plt.title("Accuracy By Max Depth Without Cases & Deaths")
else:
    plt.title("Accuracy By Max Depth With Cases & Deaths")
#plt.savefig("depth_accuracy_without_cases.png")


# Now that we know which depth is best,
# get avg confusion matrix and feature
# importance of the best depth
best_depth = accuracy_by_depth.index(max(accuracy_by_depth)) + 1

############################
#                          #
#   AVG CONFUSION MATRIX   #
#                          #
############################

cf_matrix = np.array([[total_tn[best_depth]/5, total_fp[best_depth]/5], [total_fn[best_depth]/5, total_tp[best_depth]/5]])

# code used from https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
# to make this custom confusion matrix plot
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.2f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

plt.figure(figsize=(12.7,6.2))
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
if len(X.columns) < 17:
    plt.title("Average Confusion Matrix Without Cases & Deaths")
else:
    plt.title("Average Confusion Matrix With Cases & Deaths")
#plt.savefig("confusion_matrix_without_cases.png")



############################
#                          #
#   FEATURE IMPORTANCE     #
#                          #
############################

# get the avg of each feature's importance for all the k folds
avg_importance = [np.mean(k) for k in zip(*(row for row in feature_importance[best_depth]))]
features = list(X.columns)

# replace _ and 'and' of feature names with \n
for i in range (len(features)):
    if '_and_' in features[i]:
        features[i] = features[i].replace('_and_', '\n')
    if '_' in features[i]:
        features[i] = features[i].replace('_', '\n')

# show off the bar graph of feature importance
plt.close()
plt.figure(figsize=(12.7,6.2))
plt.bar(features, avg_importance)
plt.xlabel("Features")
plt.ylabel("Importance")
if len(X.columns) < 17:
    plt.title("Feature Importance Without Cases & Deaths")
else:
    plt.title("Feature Importance With Cases & Deaths")
plt.xticks(rotation=30)
#plt.savefig("feature_importance_without_cases.png")

############################
#                          #
#  Classification Report   #
#                          #
############################

clf_report = classification_reports[best_depth][2]

plt.figure(figsize=(12.7,6.2))
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap='Blues')
plt.yticks(rotation=0)
if len(X.columns) < 17:
    plt.title("Classification Report Without Cases & Deaths")
else:
    plt.title("Classification Report With Cases & Deaths")
#plt.savefig("clf_report_without_cases.png")