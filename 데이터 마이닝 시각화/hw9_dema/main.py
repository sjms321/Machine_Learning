import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("nba.csv")
A=(df['Team']=='New York Knicks')|(df['Team']=='Golden State Warriors')
one=df[A][['Age','Salary','Team','Position']]
sns.set_theme(style="whitegrid")
g=sns.boxplot(x="Age",y="Salary",hue="Position",data=one)
sns.boxplot(x="Age",y="Salary",hue="Position",data=one)
plt.show()
'''
일정한 두께를 갖지 않고 있는 box들을 표본이 1개 뿐이 없어서 점처럼 찍혀있습니다. 
어느 정도의 두께를 갖고있는 box또한 표본 또한 3개이상 겹치는 표본이 존재하지 않아서 나이대 별 포지션 급여의 정보가 
유의미 하게 나타나 있다고 보기 힘들 것 같습니다.
32세의 SF포지션을 보면 높은 연봉을 받는것 같아 보이지만 이것 표본이 적기에 나이 때문에 높은 연봉이 측정 되는것인지 알 수가 없습니다.
표본이 적어 어떤 데이터를 예측?하고 확정 짓기는 힘들어 보입니다.
가장 높은 연봉을 받은 나이는 32세의 SF포지션 가장낮은 연봉을 받은 인원은 23세의 SF포지션 입니다.
'''

sns.jointplot(x="Age",y="Salary",data=one,kind='reg')
plt.show()

'''
이 그래프를 보면 나이가 오를수록 연봉이 오르는 것이 이상적인 선이라고 할 수 있으나 
비교적 높은 나이 33세에도 22세와 다를 것이 없는 연봉은 받는 사람도 있고, 반면에 26세의 나이임에도 불구하고
높은 연봉을 받는 인원도 있습니다 따라서 표본의 수가 너무 적어 데이터를 어떠어떠하다고 확정 짓기는 힘들어 보입니다.
이 그래프를 통해 알 수 있는 또 하나의 사실은 연봉을 높게 받는 인원은 적고 낮은 인원이 좀 많다는 것입니다.
또한 나이대가 고루(?) 분포되어 있습니다.
'''

sns.catplot(x="Age",data=one,aspect=5,kind="count",hue='Position')
plt.show()
'''
두번째 그래프에서와 같이 나이는 고루 분포 되어있고 표본의 개수가 적어 확정적인 판단을 내리기는 힘든것 같습니다
적은 표본 임에도 불구하고 분석을 해보자면
C 포지션 같은 경우는 20대 중후반에 집중적이고,
나머지 포지션 같은 경우에는 전 연령에 고루 분포 되어있는 것으로 보아나이와는 상관이 없는 포지션 같습니다.

'''
