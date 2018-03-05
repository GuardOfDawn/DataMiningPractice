import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('F:\zhangyi\course\data mining\HR_comma_sep.csv')
split_target = 'satisfaction_level'
split_target2 = 'last_evaluation'

x1 = np.array(df.loc[(df['left'] == 1),[split_target]])
x2 = np.array(df.loc[(df['left'] == 1),[split_target2]])
x = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

kmeans = KMeans(n_clusters=4)  # n_clusters:number of cluster
kmeans.fit(x)
label = kmeans.labels_

plt.figure()
plt.title('Split')
plt.xlabel(split_target)
plt.ylabel(split_target2)
colors = ['b', 'g', 'r', 'k', 'y']
markers = ['o', 's', 'D', 'd', 'h']
for i, l in enumerate(kmeans.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
plt.show()
