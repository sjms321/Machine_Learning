import numpy as np
def one(Score):
    A= Score[:,2:3]
    print(A)
def two(Score):
    A=[]
    for i in range(10):
        sum = Score[i,0]+Score[i,1]+Score[i,2]+Score[i,3]
        A.append(sum)
        print(sum)
def Three(Score):
    A=[True,False,False,False,True,False,True,False,True,True]
    print(Score[A])
def four(Score):
    A= Score.transpose((1,0))
    print(A)
def main():
    Score = np.random.randint(100, size=(10, 4))
    one(Score)
    two(Score)
    Three(Score)
    four(Score)
main()
