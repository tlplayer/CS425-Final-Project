# April 8 stops at line 43200 in us-counties.csv
import csv
import pandas as pd

cases = {}
deaths = {}

with open("us-counties.csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')

    i = 0
    for row in reader:
        # first make sure it's a valid row not past April 8th
        formatted_date = row[0].split('-')
        formatted_date = ''.join(formatted_date)
        if formatted_date == 'date':
            continue
        if int(formatted_date) > 20200408:
            break
        if row[3] == '' or row[4] == '':
            continue

        if row[3] in cases:
            cases[int(row[3])] += int(row[4])
            deaths[int(row[3])] += int(row[5])
        else:
            cases[int(row[3])] = int(row[4])
            deaths[int(row[3])] = int(row[5])
        
# read in FIPS master list to add in counties that have 0 infections 0 deaths
fips = []
with open("county_fips_master.csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')

    for row in reader:
        if row[0] != 'fips':
            fips.append(int(row[0]))

for code in fips:
    if int(code) not in cases.keys():
        cases[int(code)] = 0
        deaths[int(code)] = 0

# master is in the form FIPS, INFECTIONS, DEATHS
master = []

for fips in cases.keys():
    master.append([fips, cases[fips], deaths[fips]])

df = pd.DataFrame(master)
df.to_csv(path_or_buf='feature_data/cases.csv',index=False)