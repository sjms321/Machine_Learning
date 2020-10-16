from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
iris = load_iris()
it = iris.target.reshape((150,1))
ir = np.concatenate([iris.data,it],axis=1)
df=pd.DataFrame(ir,columns=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)','target'])
df['targetNmae']=np.where(df['target']==0,"setosa",np.where(df['target']==1,"versicolor","virginica"))
print(df.head())

fig = plt.figure()
x1=df.iloc[:50,:1]
x2=df.iloc[50:100,:1]
y1=df.iloc[:50,1:2]
y2=df.iloc[50:100,1:2]
x3=df.iloc[100:150,:1]
y3=df.iloc[100:150,1:2]
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.plot(x1,y1,'o',label = 'setosa')
plt.plot(x2,y2,'or',label = 'versicolor')
plt.plot(x3,y3,'x',label = 'virginica')
plt.legend()

fig = plt.figure()
x4=df.iloc[:50,2:3]
x5=df.iloc[50:100,2:3]
y4=df.iloc[:50,3:4]
y5=df.iloc[50:100,3:4]
x6=df.iloc[100:150,2:3]
y6=df.iloc[100:150,3:4]
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.plot(x4,y4,'ob',label = 'setosa')
plt.plot(x5,y5,'or',label = 'versicolor')
plt.plot(x6,y6,'xk',label = 'virginica')
plt.legend()
plt.show()