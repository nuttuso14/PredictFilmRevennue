import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
       }
  
df = DataFrame(Data,columns=['x','y'])
  
print(df)

kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
#print(centroids)

results = kmeans.labels_
print(results) 
print(type(results))

#cluster_map = df
#cluster_map['x'] = df['x']
#cluster_map['cluster'] = kmeans.labels_

val = df['y'].values[18]

print(val)
print(len(df))


for n in range(0,len(df)):
    print(results[n],df['x'].values[n] ,df['y'].values[n])


#for cc in cluster_map:
#    print(cc['cluster'])
#print(x,row['x'], row['y'])

#for x, j in zip(x, y):
    
#for row in df.iterrows():
#print(row['x'], row['y'])




plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50,  cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)