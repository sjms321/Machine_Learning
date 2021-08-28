import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("nba (2).csv")
df.hist('Age')
plt.show()

a=df.groupby(['Salary'])
print(df.loc[df['Salary'].idxmax(),:])
print(df.groupby('Team')['Salary'].mean())
plt.bar(np.arange(len(df.groupby('Team')['Salary'].mean())),df.groupby('Team')['Salary'].mean())
plt.show()



print(df['College'].value_counts())
print("AVG AGE:",df['Age'].mean())
print("AVG Salary:",df['Salarys'].mean())

for name, group in df.groupby('Position')['Name']:
    print(name)
    print(group)