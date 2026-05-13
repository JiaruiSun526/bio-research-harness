import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('tcell_de_results.csv')
df['-log10fdr'] = -np.log10(df['fdr'])
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['log2fc'], df['-log10fdr'], c='gray', alpha=0.7)
candidates = df[df['candidate_flag'] == 'YES']
ax.scatter(candidates['log2fc'], candidates['-log10fdr'], c='red', s=100, label='Candidates')
for _, row in candidates.iterrows():
    ax.annotate(row['gene'], (row['log2fc'], row['-log10fdr']))
ax.set_xlabel('log2 Fold Change')
ax.set_ylabel('-log10 FDR')
ax.set_title('T Cell DE Volcano Plot')
ax.legend()
plt.savefig('volcano_plot.png', dpi=150)
