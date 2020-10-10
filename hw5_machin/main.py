from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score

X1 = np.random.randint(101, size=(100, 3))
X2 = np.random.randint(50,101, size=(100, 3))
Y1 = np.full((100,1),0)
Y2 = np.full((100,1),1)

X=np.concatenate([X1,X2],axis=0)
Y=np.concatenate([Y1,Y2],axis=0)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
model = LogisticRegression()
model.fit(X_train,Y_train)
pred= model.predict(X_test)
#모델 -> fit -> predict
print(Y_test.shape)
print("실제 값 :"+str(Y_test.ravel()))
print("예측 값 :"+str(pred))
print("스코어 값(정확도) :"+ str(accuracy_score(Y_test,pred)*100)+"%")
