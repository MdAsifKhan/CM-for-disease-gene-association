import networkx as nx 
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import json
import h5py
from sklearn.cross_validation import LeaveOneOut, KFold


def read_complete_graph(graph_file):
	prot_graph = pd.read_csv(DATA_ROOT +'Graph/' +graph_file ,sep=' ')
	graph = nx.Graph()
	protein_1 = prot_graph['protein1']
	protein_2 = prot_graph['protein2']
	for node1,node2 in zip(protein_1,protein_2):
		graph.add_node(node1)
		graph.add_node(node2)
		graph.add_edge(node1, node2)
	protein_adj_matrix = np.asarray(nx.adjacency_matrix(graph, nodelist=None).todense())
	protein_nodes = graph.nodes()

	return (protein_adj_matrix,protein_nodes)

def walk_kernel(adj_matrix,initial_prob,alpha):
	nm_nodes = len(adj_matrix)
	adj_matrix = preprocessing.normalize(adj_matrix, norm='l1', axis=0)
	I = np.eye(nm_nodes, dtype='float32')
	R = alpha*np.linalg.inv(I-(1-alpha)*adj_matrix)
	M = np.dot(R, initial_prob)
	
	return M

def random_walk(adj_matrix, initial_prob, alpha):	
	adj_matrix = preprocessing.normalize(adj_matrix, norm='l1', axis=0)
	prob_t = np.copy(initial_prob)
	while(1):
		prob_t_1 = (1-alpha)*np.dot(adj_matrix, prob_t) + alpha*initial_prob
		dist = np.linalg.norm((prob_t_1 - prob_t), ord=1)
		if dist<1e-6:
			break
		else:			
			prob_t = np.copy(prob_t_1)			
	return prob_t_1


DATA_ROOT = '../CM-for-disease-gene-association/data/'

# Gene Graph File, Nodes in graph are Ensembl IDs of genes
graph_file = '9606.protein.links.v10.5.txt'
print('Read Graph ')
gene_adj_matrix, gene_nodes = read_complete_graph(graph_file)

gene_nodes = {n:v for v,n in enumerate(gene_nodes)}
inv_gene_nodes = {v:n for n,v in gene_nodes.items()}

disgenet_data = pd.read_csv(DATA_ROOT + 'disgenet/curated_gene_disease_associations.tsv',sep='\t')
gene_id_disgenet = disgenet_data['geneId'].values
disease_id_disgenet = disgenet_data['diseaseId'].values
# Entrez to Ensemble ID
gene_ensembl = pd.read_csv(DATA_ROOT + 'disgenet/gene2ensembl',sep='\t')
tax_id = gene_ensembl['#tax_id'].values
ensembl_id = gene_ensembl['Ensembl_protein_identifier'].values
gene_entrez = gene_ensembl['GeneID'].values
if not os.path.exists(DATA_ROOT + 'disgenet/entrez_to_ensemble.txt'):
	print('Mapping Entrez to Ensemble ')
	f1 = open(DATA_ROOT + 'disgenet/entrez_to_ensemble.txt','w')
	f1.write('Entrez_ID'+ '\t' + 'Ensembl_Id'+ '\n')
	disease_ensembl_map = dict()
	for ens, tx_id, gene in zip(ensembl_id,tax_id,gene_entrez):
		if tx_id == 9606:
			ens_ = '9606.' + ens.split('.')[0]
			if ens_ in gene_nodes:
				print(ens_)
				f1.write(str(gene) + '\t' + ens_ + '\n') 

entrez_ensembl = pd.read_csv(DATA_ROOT + 'disgenet/entrez_to_ensemble.txt',sep='\t')
entrez  = entrez_ensembl['Entrez_ID'].values
ensembl = entrez_ensembl['Ensembl_Id'].values
entrez_ensembl_map  = {ent:ens for ent,ens in zip(entrez,ensembl)}

print('Creating Disease-Gene-Association:UMLS-ID-->Ensembl-ID')
#Mapping Disease (UMLS) to Gene Ids (Ensembl)
disease_gene_map = dict()
for disease, gene in zip(disease_id_disgenet, gene_id_disgenet):
	if gene in entrez_ensembl_map:
		if disease in disease_gene_map:
			disease_gene_map[disease].append(entrez_ensembl_map[gene])
		else:
			disease_gene_map[disease] = [entrez_ensembl_map[gene]]

# File with walk, coexpression and GO similarity kernel matrices
data_file = DATA_ROOT + 'random_walk_score.h5'

if not os.path.exists(data_file):

	# Random Walk Analysis with 3 fold seed nodes 
	nm_fold = 3
	train_disease_gene_map = {i:{} for i in range(nm_fold)}
	test_disease_gene_map = {i:{} for i in range(nm_fold)}
	score_matrix = {i:{} for i in range(nm_fold)}
	for disease in disease_gene_map:
		gene_assocn = disease_gene_map[disease]
		if len(gene_assocn)>3:
			kf = KFold(len(gene_assocn), nm_fold, shuffle=True, random_state=9)
			for j,(train, test) in enumerate(kf):
				train_gene = [gene_assocn[i] for i in train]
				test_gene = [gene_assocn[i] for i in test]
				train_disease_gene_map[j][disease] = train_gene
				test_disease_gene_map[j][disease] = test_gene

				initial_prob = np.zeros(len(gene_nodes), dtype='float32')
				ind = [gene_nodes[el] for el in train_disease_gene_map[j][disease]]
				initial_prob[ind] = 1.0/float(len(train_disease_gene_map[j][disease]))

				print('Walking fold {} for disease {}'.format(j,disease))
				disease_prob = random_walk(gene_adj_matrix,initial_prob,0.7)
				score_matrix[j][disease] = disease_prob
	
	disease_score_matrix = np.zeros(nm_fold, (len(score_matrix[0]), len(gene_nodes)), dtype='float32')
	disease2id = {i:{} for i in range(nm_fold)}

	for j in range(nm_fold):
		score_mat = score_matrix[j]
		for i,(disease, reprsn) in enumerate(score_mat.items()):
			disease_score_matrix[j,i,:] = reprsn
			disease2id[j][disease] = i 
	
	#Mapping of Node names (Ensembl ID) to row index in Adjacency Matrix
	with open(DATA_ROOT + 'disgenet/gene_nodes.dict','w') as f:
		j = json.dumps(gene_nodes)
		f.write(j)

	#Disease-gene association used for training
	with open(DATA_ROOT + 'disgenet/train_disease_gene.dict','w') as f:
		j = json.dumps(train_disease_gene_map)
		f.write(j)

	#Disease-gene association used for testing
	with open(DATA_ROOT + 'disgenet/test_disease_gene.dict','w') as f:
		j = json.dumps(test_disease_gene_map)
		f.write(j)

	# Mapping from disease name to stationary probability vector
	with open(DATA_ROOT + 'disgenet/disease2id.dict','w') as f:
		j = json.dumps(disease2id)
		f.write(j)

	with h5py.File(data_file,'w') as hf:
		hf.create_dataset('disease_probability', data = disease_score_matrix)
