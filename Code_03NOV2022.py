#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import anndata as ad
import numpy as np

# Read gene expression data in parquet format
df_url = "/Users/aitormc/adata_train.parquet"
df = pd.read_parquet(df_url)

from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

# Sample data with categorical labels
row_labels = df['obs_id'].values
col_labels = df['gene'].values
values = df['count']

# Create mappings from categorical labels to numerical indices
row_label_to_index = {label: index for index, label in enumerate(set(row_labels))}
col_label_to_index = {label: index for index, label in enumerate(set(col_labels))}

# Map categorical labels to numerical indices and create triplets
row_indices = [row_label_to_index[label] for label in row_labels]
col_indices = [col_label_to_index[label] for label in col_labels]

# Create a sparse matrix from the triplets
sparse_matrix = coo_matrix((values, (row_indices, col_indices)))

sparse_matrix = sparse_matrix.tocsr()

# Export to npz
from scipy.sparse import coo_matrix, save_npz

filename = "rna_seq_counts_sparse_matrix.npz"
save_npz(filename, sparse_matrix)

# Export observation id to csv
sr = pd.Series(row_label_to_index)
sr.to_csv('rna_seq_obs_id.csv')

# Export gene id to csv
sr = pd.Series(col_label_to_index)
sr.to_csv('rna_seq_gene_symbols.csv')

# Load sparse matrix
from scipy.sparse import load_npz
fn = '/Users/aitormc/rna_seq_counts_sparse_matrix.npz'

X = load_npz(fn)

# Create an AnnData object with the sparse matrix
import scanpy as sc

adata = sc.AnnData(X=X)

# Load metadata of obs id
import pandas as pd

fn = '/Users/aitormc/rna_seq_obs_id_symbol.csv'
df_obs = pd.read_csv(fn,index_col = 0)
df_obs.columns = ['index cell']
df_obs['obs_id'] = df_obs.index
print(df_obs)

# Load metadata of gene id
fn = '/Users/aitormc/rna_seq_gene_symbol.csv'
df_var = pd.read_csv(fn,index_col = 0)
print(df_var.shape)
df_var.columns = ['index gene']
df_var

# Load metadata of obs
fn = "/Users/aitormc/adata_obs_meta.csv"
adata_metadata = pd.read_csv(fn)
print(adata_metadata)

# Merge metadata with obs id
df_obs_ = df_obs.merge(adata_metadata, on='obs_id', how='left')

# Add obs and col metadata to AnnData 
adata.obs = df_obs_
adata.var = df_var

# Filtering of genes/cells and normalization counts
print(adata.shape)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
print(adata.shape)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata.X)

import pandas as pd
import matplotlib.pyplot as plt

occurrences = adata.obs.groupby(['cell_type', 'sm_name']).size().unstack().fillna(0)

# Calculate the number of rows and columns for the grid
num_rows = 3
num_columns = 3

# Get the list of unique cells
unique_cells = occurrences.index.unique()

# Create a grid of subplots with the desired number of rows and columns
fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 10))

# Iterate through cells and generate a bar plot for each one
for i, cell in enumerate(unique_cells):
    row = i // num_columns
    col = i % num_columns

    ax = axs[row, col]
    ax.bar(occurrences.columns, occurrences.loc[cell])
    ax.set_title(f'Cell type: {cell}')
    ax.set_xlabel('Compound')
    ax.set_ylabel('Absolute frequency')
    
    # Adjust the font size of the X-axis
    ax.tick_params(axis='x', labelsize=2)
    
    ax.set_xticklabels(occurrences.columns, rotation=45)
    ax.set_ylim(0, 6500)  # Adjust the y-axis range according to your data

# Remove unused subplots if there are fewer than 9 cells
for i in range(len(unique_cells), num_rows * num_columns):
    fig.delaxes(axs[i // num_columns, i % num_columns])

plt.tight_layout()
plt.show()

# Create training and validation datasets

# Validation
is_t_regulatory = adata.obs["cell_type"] == "T regulatory cells"
is_resminostat_or_cabozantinib = adata.obs["sm_name"].isin(["Resminostat", "Cabozantinib"])

validation = adata[is_t_regulatory & is_resminostat_or_cabozantinib]

# Training
# Remove T regulatory cells for compounds Resminostat and  
training = adata[~((adata.obs["cell_type"] != "T regulatory cells") &
                    (adata.obs["sm_name"] != "Resminostat") & (adata.obs["sm_name"] != "Cabozantinib"))]

