from dtw import dtw
import  numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import operator

df = pd.read_csv('/Users/tony/Desktop/BDA/Wallmart-project/data/features.csv')

# Make list for each store
ts_store=[]
for i in range(1,46):
    ts_store.append(df[(df.Store==i)].Fuel_Price.tolist())


# Make list for each state
ts_state={}
mypath='/Users/tony/Desktop/BDA/Wallmart-project/data/ts_state/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print onlyfiles
# For each file in folder
for i in onlyfiles:

    if i.find('.csv') != -1:
        #Read csv
        df = pd.read_csv(mypath+i)
        # Convert in date
        df['Week of'] = pd.to_datetime(df['Week of'])
        # Slice and sort
        df = df[(df['Week of'] >= '2010-02-01') & (df['Week of'] <= '2013-07-31')].sort_values(['Week of'])

        # Make ts
        ts_state[i]=(df[df.columns[1]].tolist())

columns=[x.replace(".csv","") for x in ts_state.keys()]
columns.append("Area")
df = pd.DataFrame(columns=columns, index=[i for i in range(1,46)])
# Check similarity with dtw
for i in range (0,len(ts_store)):
    dist_row={}
    for key  in ts_state.keys():
        x = np.array(ts_store[i]).reshape(-1, 1)
        y = np.array(ts_state[key]).reshape(-1, 1)
        dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
        dist_row[key.replace(".csv","")]=dist

    dist_row['Area'] = min(dist_row.iteritems(), key=operator.itemgetter(1))[0]

    df.loc[i+1] = pd.Series(dist_row)
    print "Done Store %s" % str(i+1)

df.to_csv('dtw_matrix.csv')
