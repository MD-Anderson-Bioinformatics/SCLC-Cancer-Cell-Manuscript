"""
Cell Transport Potential for SC68 tumors
Author: Sarah Maddox Groves
2019/2020
"""

import scanpy as sc
import scvelo as scv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import os.path as op
import scipy as sp
from scipy.spatial import distance
from scvelo import settings
from scvelo import logging as logg
from scvelo.tools.utils import scale, groups_to_bool, strings_to_categoricals
from random import seed
from scipy.sparse import linalg, csr_matrix, issparse
from matplotlib.colors import ListedColormap, to_rgba
from scvelo.preprocessing.neighbors import get_connectivities
from scipy.sparse import csr_matrix

#############################################################
# additional functions
#############################################################

def normalize(adata, log = True, plot_hvg = False, min_mean = 0.0125, max_mean = 5, min_disp = 0.8):
    print("Normalizing...")
    sc.pp.normalize_per_cell(adata, copy=False)
    if log == True:
        sc.pp.log1p(adata)
    print("Finding highly variable genes (HVGs)... ")
    sc.pp.highly_variable_genes(adata, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp, flavor='cell_ranger')
    if plot_hvg == True:
        sc.pl.highly_variable_genes(adata)
        plt.show()
    return adata

def write_to_obs(adata, key, vals, cell_subset=None):
    if cell_subset is None:
        adata.obs[key] = vals
    else:
        vals_all = adata.obs[key].copy() if key in adata.obs.keys() else np.zeros(adata.n_obs)
        vals_all[cell_subset] = vals
        adata.obs[key] = vals_all


def ctrp(data, vkey='velocity', groupby=None, groups=None, self_transitions=True, basis='pca', plot = 'umap',
                    weight_diffusion=0, scale_diffusion=1, eps=1e-2, copy=False, n_neighbors = None, adata_dist = 'X_pca'):
    '''

    Args:
        data: AnnData object
        vkey: velocity is default-- where to grab the graph for computing the transition matrix.
        groupby: louvain or Batch_Names; this assumes that each group is completely disjoint and computes fate
        for each group individually
        groups:
        self_transitions: if True, the transition matrix is computed with self transitions possible.
        basis: if None, the transition matrix is computed on the full dataset.
        plot: if "umap" or "pca" plot the absorbing states on the dimensionality reduced graph
        weight_diffusion:
        scale_diffusion:
        eps:
        copy:
        n_neighbors: number of neighbors
        adata_dist: if not None, the distance between points is computed on the reduced dimensional dataset

    Returns:adds to adata:
        absorbing
        log1p_plasticity

    '''

    adata = data.copy() if copy else data
    logg.info('computing terminal states', r=True)

    groupby = 'cell_fate' if groupby is None and 'cell_fate' in adata.obs.keys() else groupby
    categories = adata.obs[groupby].cat.categories if groupby is not None and groups is None else [None]
    for c, cat in enumerate(categories):
        groups = cat if cat is not None else groups
        cell_subset = groups_to_bool(adata, groups=groups, groupby=groupby)
        _adata = adata if groups is None else adata[cell_subset]
        connectivities = get_connectivities(_adata, 'distances')

        T = scv.tl.transition_matrix(_adata, vkey=vkey, basis=basis, weight_diffusion=weight_diffusion,
                              scale_diffusion=scale_diffusion, self_transitions=self_transitions, backward=False)

        eigvecs_ends = eigs(T, eps=eps, perc=[2, 98])[1]
        ends = csr_matrix.dot(connectivities, eigvecs_ends).sum(1)
        # ends = eigvecs_ends.sum(1)

        ends = scale(np.clip(ends, 0, np.percentile(ends, 98)))
        write_to_obs(adata, 'end_points', ends, cell_subset)

        _adata.obs['col'] = ends

        n_ends = eigvecs_ends.shape[1]
        groups_str = ' (' + groups + ')' if isinstance(groups, str) else ''

        logg.info('    identified ' + str(n_ends) + ' end points' + groups_str)
        print("Dropping absorbing rows for fundamental matrix...")


        absorbing = np.where(ends >=1. - eps)[0]
        T = dropcols_coo(T,np.where(ends >=1. - eps)[0])
        if np.all(T.getnnz(1) == 0) == False:
            dropped = absorbing
        else:
            dropped = np.concatenate((absorbing, np.where(T.getnnz(1) == 0)[0]))
            print("Dropping rows with all zeroes...")
            T = dropcols_coo(T, np.where(T.getnnz(1) == 0)[0])

        I = np.eye(T.shape[0])
        print(T.shape)

        if np.abs(np.linalg.det(I-T)) == 0:
            print('Determinant equals 0. Solving for fate using pseudoinverse.')
            fate = np.linalg.pinv(I-T)
        else:
            print("Calculating fate...")
            fate = np.linalg.inv(I - T)  # fundamental matrix of markov chain
        if issparse(T): fate = fate.A
        fate = fate / fate.sum(axis=1)  # each row is a different starting cell

        if adata_dist is None:
            adataX = dropcols_coo(_adata.X, dropped)
            adataX = sp.sparse.csr_matrix.todense(np.expm1(adataX))

        elif adata_dist == 'raw':
            adataX = dropcols_coo(_adata.raw.X, dropped)

            adataX = sp.sparse.csr_matrix.todense(adataX)
        else:
            adataX = _adata.obsm[adata_dist]
            mask = np.ones(len(adataX), dtype=bool)
            mask[dropped] = False
            adataX = adataX[mask,:]
        print("Calculating distances...")
        dist = distance.cdist(adataX, adataX, 'euclidean')
        print("Calculating inner product...")
        ip = np.einsum('ij,ij->i', fate, dist)
        if cell_subset is not None:
            cs = cell_subset.copy()
            new_subset = pd.Series(cs, index = adata.obs_names.values)
            true_subset = np.where(cs == True)[0]
            for i in list(dropped):
                new_subset.iloc[true_subset[i]] = False
        else:
            new_subset = pd.Series([True]*len(adata.obs_names.values), index = adata.obs_names.values)
            for i in list(dropped):
                new_subset.iloc[i] = False
            if 'log1p_plasticity' in adata.obs.keys():
                adata.obs['log1p_plasticity'] = np.zeros(adata.n_obs)
        write_to_obs(adata, 'log1p_plasticity', np.log1p(ip), new_subset)

    adata.obs['absorbing'] = [i for i in adata.obs['end_points'] >=1. - eps]

    return adata if copy else None

def eigs(T, k=100, eps=1e-3, perc=None):
    try:
        eigvals, eigvecs = linalg.eigs(T.T, k=k, which='LR')  # find k eigs with largest real part

        p = np.argsort(eigvals)[::-1]                        # sort in descending order of eigenvalues
        eigvals = eigvals.real[p]
        eigvecs = eigvecs.real[:, p]

        idx = (eigvals >= 1 - eps)                           # select eigenvectors with eigenvalue of 1 within eps tolerance
        eigvals = eigvals[idx]
        eigvecs = np.absolute(eigvecs[:, idx])
        print("Eigenvalues: ", eigvals)
        print(eigvecs.shape)
        if perc is not None:
            lbs, ubs = np.percentile(eigvecs, perc, axis=0)
            eigvecs[eigvecs < lbs] = 0
            eigvecs = np.clip(eigvecs, 0, ubs)
            eigvecs /= eigvecs.max(0)

    except:
        eigvals, eigvecs = np.empty(0), np.zeros(shape=(T.shape[0], 0))
    return eigvals, eigvecs

def dropcols_coo(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
    C.row -= idx_to_drop.searchsorted(C.row)    # decrement column indices
    C._shape = (C.shape[0]- len(idx_to_drop), C.shape[1] - len(idx_to_drop))
    return C.tocsr()


#############################################################
# Read in loom files
# loom files were created as described in manuscript by velocyto pipeline
#############################################################

#untreated sc68 sample
sample1 = 'SC68.LB19002_GT19-03882_SI-GA-C4_S7_L007'
rep1 = scv.read(f'SC68/{sample1}.loom', cache=True)
rep1.var_names_make_unique()

sample2 = 'SC68cr.LB19003_GT19-05072_S1_L001'
rep2 = scv.read(f'SC68/{sample2}.loom', cache=True)
rep2.var_names_make_unique()

#concatenate datasets together
adata_full = rep1.concatenate(rep2, batch_key = 'treatment',batch_categories=['untreated', 'cisplatin'])

#############################################################
# Filter cells to be consistent with rest of manuscript
#############################################################

#use only filtered cells from previous analyses (barcodes listed in each csv below)
unt_names = pd.read_csv('./sc68/68_unt_filtered_cells.csv', header = 0, index_col=0)
unt_filtered = []
for i in unt_names['x']:
    n = i.split('.')[1].split('-')[0]
    unt_filtered.append(n)

cis_names = pd.read_csv('./sc68/68_cis_filtered_cells.csv', header = 0, index_col=0)
cis_filtered = []
for i in cis_names['x']:
    n = i.split('.')[1].split('-')[0]
    cis_filtered.append(n)

#generate filtered dataframe based on two csvs above
filter_obs = pd.DataFrame(index=adata_full.obs_names, columns=['filter', 'treatment'])
for i in adata_full[adata_full.obs['treatment']=='untreated'].obs_names:
    n = i.split('/')[9]
    s = (n.split('_')[5].split(':')[1].split('-')[0])
    filter_obs.loc[i]['filter'] =s
    filter_obs.loc[i]['treatment'] ='untreated'

for i in adata_full[adata_full.obs['treatment']=='cisplatin'].obs_names:
    n = i.split('/')[9]
    s = (n.split('_')[5].split(':')[1].split('-')[0])
    filter_obs.loc[i]['filter'] =s
    filter_obs.loc[i]['treatment'] ='cisplatin'

# finally, set "filtered" attribute of adata_full
unt_bool = []
for i,r in filter_obs.iterrows():
    if r['treatment'] == 'cisplatin':
        if r['filter'] in cis_filtered:
            unt_bool.append('True-cis')
        else: unt_bool.append(None)

    elif r['treatment'] == 'untreated':
        if r['filter'] in unt_filtered:
            unt_bool.append('True-unt')
        else: unt_bool.append(None)
filter_obs['kept'] = unt_bool

adata_full.obs['filtered'] = filter_obs['kept']

adata = adata_full[(adata_full.obs.filtered=='True-cis') | (adata_full.obs.filtered=='True-unt') ,:].copy()
adata = normalize(adata, log=True)
sc.pp.neighbors(adata)
scv.pp.moments(adata)
sc.tl.tsne(adata)
sc.tl.leiden(adata)
sc.pl.tsne(adata, color = ['treatment'], palette = ['blue','orange'])


#############################################################
# Compute cell death and cycle scores
#############################################################
#repeated for each set of death genes
apo = pd.read_csv("./GO_term_summary_20200826_165105.txt", index_col=0, header = 0, sep = '\t')
apo['MGI Gene/Marker ID'] = [i.upper() for i in apo['MGI Gene/Marker ID']]
sc.tl.score_genes(adata, apo['MGI Gene/Marker ID'], score_name='apo')
sc.pl.tsne(adata, color = 'apo')

scv.tl.score_genes_cell_cycle(adata)
scv.pl.scatter(adata, color_gradients=['S_score', 'G2M_score'],
    smooth=True, perc=[10, 90], save = 'cell-cycle', basis = 'tsne')

#############################################################
# Calculate velocity using scvelo
#############################################################
scv.tl.velocity(adata, groupby='treatment', mode = 'deterministic')
scv.tl.velocity_graph(adata)
scv.tl.velocity_confidence(adata)
keys = 'velocity_length', 'velocity_confidence'
scv.pl.scatter(adata, c=keys, cmap='coolwarm', perc=[5, 95], save = 'vel_confidence_length', basis = 'tsne')

#############################################################
# PAGA velocity graph-- from scVelo tutorial
#############################################################

# this is needed due to a current bug - bugfix is coming soon.
adata.uns['neighbors']['distances'] = adata.obsp['distances']
adata.uns['neighbors']['connectivities'] = adata.obsp['connectivities']
scv.tl.paga(adata, groups='leiden')

cis = adata[adata.obs['treatment']=='cisplatin']
unt = adata[adata.obs['treatment']=='untreated']

scv.tl.paga(cis, groups='leiden')
df = scv.get_df(cis, 'paga/transitions_confidence', precision=2).T
df.style.background_gradient(cmap='Blues').format('{:.2g}')


scv.tl.paga(unt, groups='leiden')
df = scv.get_df(unt, 'paga/transitions_confidence', precision=2).T
df.style.background_gradient(cmap='Blues').format('{:.2g}')

scv.pl.paga(adata, basis='tsne', size=50, alpha=.1, palette = 'tab20',
            min_edge_width=2, node_size_scale=1.5, save = 'paga_all_68')
scv.pl.paga(unt, basis='tsne', size=50, alpha=.1,palette = 'tab20',
            min_edge_width=2, node_size_scale=1.5,save = 'paga_unt_68')
scv.pl.paga(cis, basis='tsne', size=50, alpha=.1,palette = 'tab20',
            min_edge_width=2, node_size_scale=1.5,save = 'paga_cis_68')

#############################################################
# Calculate log1p_plasticity
#############################################################

ctrp(adata, groupby = 'treatment', adata_dist='X_pca', eps = 0.01, basis = 'pca', self_transitions=False)

#############################################################
#Plotting
#############################################################

_adata = adata[adata.obs['absorbing']==True,]
ax = scv.pl.scatter(adata, color='log1p_plasticity', show=False,color_map='jet', alpha = 0.5,basis='tsne',dpi = 350, zorder = 1)
scv.pl.scatter(_adata, color='black',  size=30, ax = ax, zorder = 2, alpha = 1, save = 'plasticity_all', basis = 'tsne')

plt.figure(figsize = (10,5))
color = sns.color_palette('muted')[0]
for r in adata.obs['treatment'].cat.categories:
    _adata = adata.copy()
    _adata = _adata[_adata.obs['treatment']==r,]
    sns.distplot(sorted(_adata.obs['log1p_plasticity']), bins = 100)
    plt.axvline(x = _adata.obs['log1p_plasticity'].median(), ymax=1.5, ymin = 0, linestyle = '--', color = color)
    color = 'orange'
#     plt.scatter(y = [x/len(_adata.obs_names) for x in range(len(_adata.obs_names))], x = sorted(_adata.obs['log1p_plasticity']))
    print(_adata.obs['log1p_plasticity'].mean())
    plt.show()
