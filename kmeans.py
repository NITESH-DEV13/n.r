import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    'Feature1': [1, 2, 1, 5, 6, 5, 9, 8, 9],
    'Feature2': [2, 1, 3, 6, 5, 7, 9, 8, 10]
}
df = pd.DataFrame(data)
X = df.values

print(X)

kmeans=KMeans(n_clusters=3, max_iter=10, n_init=1, random_state=42)

mse_list=[]
for i in range(1,11):
    temp_kmeans=KMeans(n_clusters=3, max_iter=i, n_init=1, random_state=42)
    temp_kmeans.fit(X)
    centers=temp_kmeans.cluster_centers_
    mse = np.mean([np.min([np.linalg.norm(x - c) ** 2 for c in centers]) for x in X])
    mse_list.append(mse)

plt.plot(range(1,11), mse_list, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Mean squared error(MSE)')
plt.title('MSE over iteration on your dataset')
plt.grid(True)
plt.show()


