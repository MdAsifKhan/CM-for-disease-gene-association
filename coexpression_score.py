import os
import numpy as np
import json
import pandas as pd

DATA_ROOT = '../CM-for-disease-gene-association/data/'

entrez_ensembl = pd.read_csv(DATA_ROOT + 'disgenet/entrez_to_ensemble.txt',sep='\t')
entrez  = entrez_ensembl['Entrez_ID'].values
ensembl = entrez_ensembl['Ensembl_Id'].values
entrez_ensembl_map  = {ent:ens for ent,ens in zip(entrez,ensembl)}


with open(DATA_ROOT + 'disgenet/gene_nodes.dict','r') as f:
	gene_nodes = json.load(f)

print('Building Pairwise Semantic Similarity Matrix \n')

# Coexpression Kernel, missing genes will have 0 Coexp Value
coexp_sim_matrix = np.zeros((len(gene_nodes), len(gene_nodes)), dtype='float32')

for file in enumerate(os.listdir(DATA_ROOT + 'similarity/')):
	if int(file[1]) in entrez_ensembl_map and entrez_ensembl_map[int(file[1])] in gene_nodes:
		idx = gene_nodes[entrez_ensembl_map[int(file[1])]]
		with open(DATA_ROOT + 'similarity/' + file[1], 'r') as f:
			for line in f:
				data = line.strip().split('\t')
				gene = data[0]
				coexp_score = data[2]
				if int(gene) in entrez_ensembl_map and entrez_ensembl_map[int(gene)] in gene_nodes:
					ensembl_id = entrez_ensembl_map[int(gene)]
					coexp_sim_matrix[idx,gene_nodes[ensembl_id]] = coexp_score

np.save(DATA_ROOT + 'Coexpression_score.npy',coexp_sim_matrix)
