import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("nba.csv")

print(df.pivot_table(["Age","Salary"],index=["Position","Team"]))

max_salary= lambda x:x['Salary'].max()
df.groupby(df['Position']).apply(max_salary)
plt.bar(np.arange(len(df.groupby(df['Position']).apply(max_salary))),df.groupby(df['Position']).apply(max_salary))
plt.xticks(np.arange(len(df.groupby(df['Position']).apply(max_salary))),['C', 'PF', 'PG', 'SF', 'SG'])
plt.show()


df.groupby(df['Team']).apply(max_salary)
plt.bar(np.arange(len(df.groupby(df['Team']).apply(max_salary))),df.groupby(df['Team']).apply(max_salary))
a=df['Team'].drop_duplicates().dropna()
B=a.str.split(" ").str[0].str[0]+a.str.split(" ").str[1].str[0]
plt.xticks(np.arange(len(df.groupby(df['Team']).apply(max_salary))),sorted(B))

plt.show()

max_age= lambda x:x['Age'].max()
df.groupby(df['Team']).apply(max_age)
plt.bar(np.arange(len(df.groupby(df['Team']).apply(max_age))),df.groupby(df['Team']).apply(max_age))

plt.xticks(np.arange(len(df.groupby(df['Team']).apply(max_age))),sorted(B))
fig = plt.figure(figsize=(1,1))
plt.show()