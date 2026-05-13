import matplotlib.pyplot as plt
import numpy as np

# Top 5 shared pathways
pathways = ['JAK-STAT', 'Immune system', 'Cytokine receptor',
            'Interleukin signaling', 'NF-kB']
tcell_fdr = [0.001, 0.001, 0.002, 0.001, 0.006]
bcell_fdr = [0.005, 0.003, 0.008, 0.005, 0.02]

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(pathways))
ax.scatter(-np.log10(tcell_fdr), y, s=100, c='blue', label='T cell')
ax.scatter(-np.log10(bcell_fdr), y, s=100, c='red', label='B cell')
# P-value annotations (user requested)
for i in range(len(pathways)):
    ax.annotate(f'p={tcell_fdr[i]}', (-np.log10(tcell_fdr[i]), y[i]+0.1))
    ax.annotate(f'p={bcell_fdr[i]}', (-np.log10(bcell_fdr[i]), y[i]-0.2))
ax.set_yticks(y)
ax.set_yticklabels(pathways)
ax.set_xlabel('-log10(FDR)')
ax.set_title('Pathway Comparison: T cell vs B cell (Top 5 Shared)')
ax.legend()
plt.tight_layout()
plt.savefig('comparison_dotplot.png', dpi=150)
