import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing

X = pd.read_csv("dataset.csv")
X=X.values
names = ['index' ,'fips' ,'cases','deaths','never','rarely','sometimes','frequently','always','retail','grocery','parks','transit','workplaces','residential','pop_total', 'popdensity','candidate','risk']
print(len(names))

X = preprocessing.StandardScaler().fit_transform(X)
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title('Hierarchical Clustering Dendrogram')

print(len(X.T))
print()
linked = linkage(X.T, 'single')
plt.tight_layout()
dendrogram(linked, labels = names, leaf_font_size = 10)
plt.show()
