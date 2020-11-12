from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd

# read in master dataset
df = pd.read_csv("../dataset.csv")

# get rid of first column
df = df.iloc[:, 1:]

# split up features from target
X = df.iloc[:, :-1]
Y = df.iloc[:, -1:]

# X = X.drop(columns="cases")
# X = X.drop(columns="pop_total")
# X = X.drop(columns="deaths")
# print(X)

# split into train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)


dt = tree.DecisionTreeClassifier(criterion="gini")
dt.fit(x_train, y_train)
#print(dt.decision_path(x_test))
print(dt.feature_importances_)

score = dt.score(x_test, y_test)
print(score)