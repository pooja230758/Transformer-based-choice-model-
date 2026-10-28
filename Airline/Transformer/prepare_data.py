import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

print("Loading data...")
df = pd.read_csv('../data_github/transaction_data.csv')

groups = df.groupby('orderid')

max_len = 30
num_orders = len(groups)

print(f"Num orders: {num_orders}, max_len: {max_len}")

cat_indices = [0, 1, 2, 3, 4, 11, 12]
conti_indices = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]

data_cat = np.zeros((num_orders, max_len, 7), dtype=np.int32)
data_conti = np.zeros((num_orders, max_len, 10), dtype=np.float32)
labels = np.zeros((num_orders, max_len), dtype=np.int32)
valid_len = np.zeros(num_orders, dtype=np.int32)

feature_cols = [c for c in df.columns if c not in ['orderid', 'label']]

df_conti = df[feature_cols].iloc[:, conti_indices]
scaler = MinMaxScaler()
scaler.fit(df_conti)

for i, (orderid, group) in enumerate(groups):
    v_len = len(group)
    if v_len > max_len:
        v_len = max_len
        group = group.iloc[:max_len]
        
    valid_len[i] = v_len
    
    cat_vals = group[feature_cols].iloc[:, cat_indices].values
    conti_vals = scaler.transform(group[feature_cols].iloc[:, conti_indices].values)
    
    data_cat[i, :v_len, :] = cat_vals
    data_conti[i, :v_len, :] = conti_vals
    
    for j, val in enumerate(group['label'].values):
        if val == 1:
            labels[i, j] = 1

os.makedirs('np_data', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

np.save('data_cat_trans30.npy', data_cat)
np.save('data_conti_trans30.npy', data_conti)
np.save('label_trans30.npy', labels)
np.save('valid_len_trans30.npy', valid_len)

print("Saved numpy files.")
