import sqlite3
import pandas as pd

conn = sqlite3.connect('results.db')
df = pd.read_sql_query("SELECT * FROM kernels", conn)

# 找 sphere 附近
sphere_idx = df[df['kernel_name'].str.contains('sphere', case=False)].index
for idx in sphere_idx:
    start = max(0, idx-5)
    end = min(len(df), idx+6)
    print(df.iloc[start:end][['kernel_name', 'duration_ns']])