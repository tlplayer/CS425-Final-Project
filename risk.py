import csv
import pandas as pd

master = []

with open("Risk Levels Downloadable Data.csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')

    for row in reader:
        if row[1] == "COVID-19 Risk Level":
            # column 69 is April 8th, a Wednesday so not to get weekend influence
            master.append([row[0], row[69]])

# now get the master list of fips and make dictionaries out of it
fips = {}
with open("county_fips_master.csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')

    for row in reader:
        if row[0] != 'fips':
            name = row[1] + ", " + row[2]
            name = name.upper()
            fips[name] = row[0]

# change county name to FIPS
# change green and yellow to 0, for low risk
# change orange and red to 1, for high risk
for row in master:
    if row[0].upper() in fips.keys():
        row[0] = fips[row[0].upper()]
        if row[1] == 'Green' or row[1] == 'Yellow':
            row[1] = 0
        else:
            row[1] = 1

master_adjusted = []
for row in master:
    if row[0].isnumeric():
        master_adjusted.append(row)

df = pd.DataFrame(master_adjusted)
df.to_csv(path_or_buf='risk.csv',index=False)