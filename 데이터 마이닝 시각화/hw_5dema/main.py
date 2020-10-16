import numpy as np
import  matplotlib.pyplot as plt

def main():
    Score = np.random.randint(100, size=(10, 4))
    one(Score)
    two(Score)
    three(Score)
    four(Score)

def one(Score):
    y=Score[:,:1]
    x=Score[:,1:2]
    fig =plt.figure()

    plt.xlabel("English")
    plt.ylabel("Korean")
    plt.scatter(x,y,marker='^')
    plt.show()
def two(Score):
    fig = plt.figure(figsize=(10, 10))

    grid = plt.GridSpec(2, 2, hspace=0.2, wspace=0.2)
    korean= Score[:,:1]
    english = Score[:,1:2]
    math = Score[:,2:3]
    science = Score[:,3:]
    k_hist = fig.add_subplot(grid[:1,:1])
    plt.xlabel("Korea")
    plt.ylabel("count")
    e_hist = fig.add_subplot(grid[:1,1:])
    plt.xlabel("english")
    plt.ylabel("count")
    m_hist = fig.add_subplot(grid[1:,:1])
    plt.xlabel("math")
    plt.ylabel("count")
    s_hist = fig.add_subplot(grid[1:,1:])
    plt.xlabel("science")
    plt.ylabel("count")
    k_hist.hist(korean)
    e_hist.hist(english)
    m_hist.hist(math)
    s_hist.hist(science)

    plt.show()

def three(Score):
    B=  np.logical_and(Score[:,:1]>=80,Score[:,:1]<90)
    C =np.logical_and(Score[:,:1]>=70,Score[:,:1]<80)
    arrlist = [np.count_nonzero(Score[:,:1]>=90),
               np.count_nonzero(B),
               np.count_nonzero(C),
               np.count_nonzero(Score[:,:1]<70)]
    label = ['A', 'B', 'C','D']
    print(np.count_nonzero(Score[:,:1]>=90))
    print(np.count_nonzero(B))
    print(np.count_nonzero(C))
    print(np.count_nonzero(Score[:,:1]<70))
    plt.pie(arrlist, labels=label)
    plt.show()
def four(Score):
    arrlist=np.array(Score)
    plt.boxplot(arrlist,labels=("Kor","Eng","Math","Science"))
    plt.show()
main()