import matplotlib.pylab as plt #그림을 그리기위함
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets #선형 모델 중 당뇨병 모델을 가져옴

from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes() # load data
#당뇨병 데이터 로드
#print(diabetes.keys()) #이것을 통해 이 모델이 어떤 특징들을 갖고있는지와 방향성을 알게됨.
#print(diabetes['data'])
#print(diabetes['target'])
#print(diabetes['DESCR'])
#print(diabetes['feature_names'])#'age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'

diabetes_X_train = diabetes.data[:-20,:]#슬라이싱 훈련용 데이터
print(diabetes_X_train.shape)
diabetes_X_test = diabetes.data[-20:,:] # 테스트용 데이터터print(diabetes_X_test)
diabetes_y_train = diabetes.target[:-20]#처음부터 뒤에서 20번째까지 슬라이싱 훈련용 데이터
#print(diabetes_y_train)
diabetes_y_test = diabetes.target[-20:]#뒤에서 20번째부터 끝까지 슬라이싱
print(diabetes.target.shape)



model = LinearRegression()
#선형 모델 생성
model.fit(diabetes_X_train, diabetes_y_train)
#diabetes_X_train과 diabetes_y_train의 최적의 선을 찾아 그리기
#==> 훈렬용 데이터들을 이용하여 최적의 선을 찾아 그려낸다.

y_pred = model.predict(diabetes_X_test)#20개의 테스트 데이터에 대한 출력 데이터 예측값
plt.plot(diabetes_y_test, y_pred, '.')#. 모양으로  예측점들을 점찍기
x = np.linspace(0, 330, 100)#0부터330까지 100개를 일정한 간격으로 만들기
y = x
plt.plot(x, y)#그래프 그리기
plt.show()#그래프 나타내기
#당뇨병환자 20명에 관한 당뇨병 발생 1년 후의 당뇨병 진행률을 나타낸 것 같다.
