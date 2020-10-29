import csv
import numpy as np
import pandas as pd

trump = []
clinton = []

with open("countypres_2000-2016.csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')


    for row in reader:
        if row[0] == '2016':
            if row[6] == 'Donald Trump':
                trump.append(row)
            elif row[6] == 'Hillary Clinton':
                clinton.append(row)

print(len(trump))
print(len(clinton))

outcome = []

# 1 for Trump wins
# 0 for Clinton wins

for i in range(len(trump)):
    if trump[i][8] == 'NA' or clinton[i][8] == 'NA':
        continue

    if int(trump[i][8]) - int(clinton[i][8]) > 0:
        outcome.append(tuple([trump[i][4], 1]))
    else:
        outcome.append(tuple([trump[i][4], 0]))


df = pd.DataFrame(outcome)
df.to_csv(path_or_buf='outcome.csv',index=False)