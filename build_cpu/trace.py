import sqlite3
import pandas as pd

conn = sqlite3.connect('bench_physx_scene_trace.db')

# 读取所有数据
df = pd.read_sql_query(
    "SELECT _Index, Name, DurationNs, BeginNs, EndNs FROM HIPOPS_12_12_12_52_2491_1783478153 ORDER BY _Index",
    conn
)

print(f"Total kernels: {len(df)}")

# 查找 sphereNphase_Kernel（精确匹配）
sphere_mask = df['Name'] == 'sphereNphase_Kernel'
sphere_indices = df[sphere_mask].index.tolist()

print(f"Found {len(sphere_indices)} sphereNphase_Kernel calls")

# 如果没有找到，尝试模糊匹配
if len(sphere_indices) == 0:
    print("No exact match, trying fuzzy match...")
    sphere_mask = df['Name'].str.contains('sphere', case=False, na=False)
    sphere_indices = df[sphere_mask].index.tolist()
    print(f"Found {len(sphere_indices)} kernels containing 'sphere'")

# 分析每个 sphere 前后的 kernel
print(f"\n{'='*80}")
print("Analyzing sphereNphase_Kernel neighbors")
print(f"{'='*80}\n")

for idx in sphere_indices[:10]:  # 只看前10个，避免输出太长
    start = max(0, idx - 5)
    end = min(len(df), idx + 6)
    
    print(f"Sphere call at _Index: {df.loc[idx, '_Index']} (row {idx})")
    print(f"Duration: {df.loc[idx, 'DurationNs']:,} ns")
    print("-" * 80)
    
    for j in range(start, end):
        marker = ">>> SPHERE <<<" if j == idx else " " * 14
        idx_val = df.loc[j, '_Index']
        name = df.loc[j, 'Name']
        dur = df.loc[j, 'DurationNs']
        print(f"  {idx_val:>8} | {name:45} | {dur:>12,} ns | {marker}")
    
    print()