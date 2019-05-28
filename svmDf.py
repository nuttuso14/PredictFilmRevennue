import numpy as np
import pandas as pd
from sklearn import svm   


#X = [[100,15,20,50,70], [80,10,7.5,3,0.5],[20,0.5,30,10,2.5],[8.8,1,0.5,20,30],[100.5,15.6,2,0.782,0.25]]
#y = [1,0,-1,0,1]
#clf = svm.SVC(gamma='scale')
#sv = clf.fit(X, y) 
#results = clf.predict([[200,0.5,100,150,0.8],[30,0.58,5.4,20,8],[5,0.75,0.75,8,15]]) 

Data = {'m': [[100,15,20,50,70], [80,10,7.5,3,0.5],[20,0.5,30,10,2.5],[8.8,1,0.5,20,30],[100.5,15.6,2,0.782,0.25]],
        'n': [1,-1,-1,20,1]
       }
  
df = pd.DataFrame(Data,columns=['m','n'])
m = df.m.apply(lambda m: pd.Series(list(m)))

#print(df.y)
print(m)
print(type(m))
print(df.n)
print(type(df.n))

clf = svm.SVC(gamma='scale')
sv = clf.fit(m,df.n) 
results2 = clf.predict([[50,0.5,100,150,0.8],[30,0.58,5.4,20,8],[5,0.75,0.75,8,15]])

print(results2)
