import pandas as pd

cases = pd.read_csv('cases.csv')
mask = pd.read_csv('mask-use-by-county.csv')
mobil = pd.read_csv('mobility.csv')
population = pd.read_csv('population.csv')
pres = pd.read_csv('pres.csv')
risk = pd.read_csv('risk.csv')

df = cases.merge(mask)
df = df.merge(mobil)
df = df.merge(population)
df = df.merge(pres)
df = df.merge(risk)

print(df.head(10))

df.to_csv('dataset.csv')