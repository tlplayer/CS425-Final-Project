import csv
import pandas as pd

master = []

with open("census-population-landarea.csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')

    for row in reader:
        if row[0] != 'fips' and row[0] != '0' and int(row[0]) % 1000 != 0:
            master.append([row[0], row[4], row[6]])

df = pd.DataFrame(master)
df.to_csv(path_or_buf='population.csv',index=False)