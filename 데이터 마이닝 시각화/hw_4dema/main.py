import numpy as np



def one(Score):
    A=np.where(Score>90,'A',
               np.where(Score>80,'B',
               np.where(Score>70,'C','D')))
    print(A)

def two(score):
    ##각 학생의 과목별 점수에서 각 과목의 평균 값을 뺀 점수를 출력하시오
    A=score.copy()
    A = A.mean(axis=0)
    score = score-A
    print(score)
def three(score):
    ##맨 마지막에 총합 점수에 대한 열을 하나 만들어서 10 x 5의 행렬을 출력을 구하시오
    A=score.copy()
    A=A.sum(axis=1)
    A= A.reshape((10,1))
    score = np.concatenate([score,A],axis=1)
    print(score)

def four(score):
    ##3번 문제에 이어 맨 마지막 행에 국어, 영어, 수학 ,과학의 평균값을 구하여 11 x 6을 구하시오
    totalscore =score.copy()
    totalscore = totalscore.sum(axis=1)
    totalscore = totalscore.reshape((10,1))
    totalavg =score.copy()
    totalavg = totalavg.mean(axis=1)
    totalavg = totalavg.reshape((10,1))
    score = np.concatenate([score,totalscore],axis=1)
    score = np.concatenate([score,totalavg],axis=1)
    print(score)
    lastline = score.copy()
    lastline = score.mean(axis=0)
    lastline = lastline.reshape((1,6))
    score = np.concatenate([score,lastline],axis=0)
    print(score)
def main():
    Score = np.random.randint(100, size=(10, 4))
    print(Score)
    one(Score)
    two(Score)
    three(Score)
    four(Score)
main()