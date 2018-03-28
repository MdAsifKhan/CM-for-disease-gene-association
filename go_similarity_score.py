import os
import numpy as np
import json
import pandas as pd

DATA_ROOT = '../CM-for-disease-gene-association/data/'

with open(DATA_ROOT + 'disgenet/gene_nodes.dict','r') as f:
	gene_nodes = json.load(f)


entrez_ensembl = pd.read_csv(DATA_ROOT + 'disgenet/entrez_to_ensemble.txt',sep='\t')
entrez  = entrez_ensembl['Entrez_ID'].values
ensembl = entrez_ensembl['Ensembl_Id'].values
entrez_ensembl_map  = {ent:ens for ent,ens in zip(entrez,ensembl)}


# Go-Similarity Kernel, Missing Genes (ones not in Graph) have zero score
semantic_file1 = 'ontologies1/go.txt'
go_sim = pd.read_csv(DATA_ROOT + semantic_file1 , header=None)

go_sim_matrix = np.zeros((len(gene_nodes),len(gene_nodes)), dtype='float32')
go_gene_nodes = [line.strip().split('\t')[0] for line in open(DATA_ROOT + 'ontologies1/gene_go_annotation.tab','r')]

c = 0
for id_ in go_gene_nodes:
	if int(id_) in entrez_ensembl_map and entrez_ensembl_map[int(id_)] in gene_nodes:
		vec_go = go_sim[c:c+18795].values
		vec_go = vec_go.flatten()
		vec_go_n = [(vec_go[i],entrez_ensembl_map[int(gene)]) for i,gene in enumerate(go_gene_nodes) \
				if (int(gene) in entrez_ensembl_map and entrez_ensembl_map[int(gene)] in gene_nodes)] 
		ind_ = gene_nodes[entrez_ensembl_map[int(id_)]]
		for inst in vec_go_n:
			col = gene_nodes[inst[1]]
			go_sim_matrix[ind_,col] = inst[0]
	c = c + 18795

np.save(DATA_ROOT + 'GO_similarity.npy', go_sim_matrix) 