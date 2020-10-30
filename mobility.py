import pandas as pd

df = pd.read_csv('us-mobility.csv')
fips = pd.read_csv('county_fips_master.csv',encoding = "ISO-8859-1")
fips = fips.iloc[:,0:3]
#Purges any line that contains Total to isolate counties
df = df[df != "Total"]


df = df.dropna()
fips = fips.rename(columns= {"county_name":"county"})
df = df[(df['date'] == '2020-04-08')]



r = df.merge(fips)
r = r.iloc[:,2:-1]
