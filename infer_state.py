from dtw import dtw
import  numpy as np
import pandas as pd

import operator

# Load csv
df = pd.read_csv('/Users/tony/Desktop/BDA/Wallmart-project/data/walmart_ur_and_area.csv')
# Convert date in datetime
df['Date'] = pd.to_datetime(df['Date'])
# Filter only for right date
df = df[(df['Date'] >= '2010-02-01') & (df['Date'] <= '2012-10-26')].sort_values(['Date'])

# Put in list store_id by area
store_id_by_area={}
for i in df["Area"].unique().tolist():
    store_id_by_area[i]=df[(df.Area == i)].Store_id.unique().tolist()

# Group by for store_id, year, month and make avg
df_grouped = df.groupby(['Store_id', 'the_year', 'the_month'], as_index=False)['Unemployment'].mean()

# For each store generate timeseire
ts_store = []
for i in range(1, 46):
    ts_store.append(df_grouped[(df_grouped.Store_id == i)].Unemployment.tolist())

# Load csv
df = pd.read_csv('/Users/tony/Desktop/BDA/Wallmart-project/data/ts_unomployment_rate/unemployment_rate_complete.csv')
# Convert date in datetime
df['Date'] = pd.to_datetime(df['Date'])


ts_state={}
# For each area
for i in store_id_by_area.keys():

    dict_area={}
    for j in df[(df.Area == i)].State.unique().tolist():
        dict_area[j]=df[df.State==j].Unemployment_rate.tolist()
    ts_state[i]=dict_area


print ts_state
print ts_store
print store_id_by_area

columns=[x for x in df.State.unique().tolist()]
columns.append('Stato')
columns.append('Store_id')
columns.append('Area')
df = pd.DataFrame(columns=columns, index=[i for i in range(1,46)])
for i in store_id_by_area:
    for j in store_id_by_area[i]:
        dist_row = {}
        x = np.array(ts_store[j-1]).reshape(-1, 1)
        for z in ts_state[i]:
            y = np.array(ts_state[i][z]).reshape(-1, 1)
            dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
            dist_row[z] = dist

        dist_row['Stato'] = min(dist_row.iteritems(), key=operator.itemgetter(1))[0]
        dist_row['Store_id'] = j
        dist_row['Area'] = i
        df.loc[j] = pd.Series(dist_row)
        print "Done Store %s" % str(j)

df.to_csv('dtw_matrix_state.csv')
