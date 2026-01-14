#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Read the log file
log_file = "output/grid_search_results_2025/grid_search.log"
if not os.path.exists(log_file):
    print(f"Log file {log_file} not found.")
    exit(1)

data = []
with open(log_file, 'r') as f:
    for line in f:
        if line.startswith("lookback_period="):
            # Parse line: lookback_period=20, hold_period=10, top_n=1, Total Return:  58.45%
            match = re.match(r"lookback_period=(\d+), hold_period=(\d+), top_n=(\d+), Total Return:\s*([\d.]+)%", line)
            if match:
                lp, hp, tn, ret = match.groups()
                data.append({
                    'lookback_period': int(lp),
                    'hold_period': int(hp),
                    'top_n': int(tn),
                    'total_return': float(ret)
                })

df = pd.DataFrame(data)

# Get unique values
top_ns = sorted(df['top_n'].unique())
lookback_periods = sorted(df['lookback_period'].unique())
hold_periods = sorted(df['hold_period'].unique())

# Create heatmaps for each top_n
fig, axes = plt.subplots(1, len(top_ns), figsize=(15, 5))
if len(top_ns) == 1:
    axes = [axes]

for i, tn in enumerate(top_ns):
    subset = df[df['top_n'] == tn]
    pivot = subset.pivot(index='lookback_period', columns='hold_period', values='total_return')
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[i])
    axes[i].set_title(f'Heatmap for top_n={tn}')
    axes[i].set_xlabel('Hold Period')
    axes[i].set_ylabel('Lookback Period')

plt.tight_layout()
save_path = log_file.rsplit('/', 1)[0]
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(f'{save_path}/hyperparameter_heatmap.png')
print(f"Heatmap saved to {save_path}/hyperparameter_heatmap.png")