import pandas as pd

df = pd.read_csv('us-mobility.csv')

#Purges any line that contains Total to isolate counties
df = df[df.A != "Total"]
print(df.head(10))
