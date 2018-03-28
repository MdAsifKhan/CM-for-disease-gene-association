import networkx as nx 
import pandas as pd
import numpy as np
import os
import json
import h5py
import multiprocessing
import pdb
from sklearn import preprocessing

# Oth Order Features
def feature_matrix_0(disease_row, fold, disease_analyze, coexp_sim_matrix, go_genes_normalize):
	disease_prob = disease_score_matrix[fold, disease_row[disease_analyze],:]
	feature_mat = np.zeros((len(gene_nodes),4),dtype='float32')

	feature_mat[:,0] = disease_prob
	for gene in range(len(disease_prob)):
		coexp_score = coexp_sim_matrix[gene]
		go_score = go_genes_normalize[gene]
		disease_rwr = np.delete(disease_prob, gene)
		coexp_score = np.delete(coexp_score,gene)
		go_score = np.delete(go_score, gene)
		feature_mat[gene,1] = max(disease_rwr*coexp_score)
		feature_mat[gene,2] = max(disease_rwr*go_score)
		feature_mat[gene,3] = max(disease_rwr*go_score*coexp_score)

	return feature_mat

# Higher Order Features
def feature_matrix_1(disease_row, fold, disease_analyze, go_coexp, coexp_go, go_2nd_order, coexp_2nd_order):
	disease_prob = disease_score_matrix[fold, disease_row[disease_analyze],:]
	feature_mat = list()
	for gene in range(len(disease_prob)):
		go_coexp_score = go_coexp[gene]
		coexp_go_score = coexp_go[gene]
		go_2nd_score = go_2nd_order[gene]
		coexp_2nd_score = coexp_2nd_order[gene]

		go_coexp_score = np.delete(go_coexp_score, gene)
		coexp_go_score = np.delete(coexp_go_score,gene)
		go_2nd_score = np.delete(go_2nd_score, gene)
		coexp_2nd_score = np.delete(coexp_2nd_score, gene)
		disease_rwr = np.delete(disease_prob, gene)

		feature_mat.append(np.array([max(disease_rwr*go_coexp_score),max(disease_rwr*coexp_go_score),max(disease_rwr*go_2nd_score),max(disease_rwr*coexp_2nd_score)],dtype='float32'))

	feature_mat = np.array(feature_mat, dtype='float32')
	return feature_mat

# 1st Order Matrix Multiplication
def power_matrix_features(coexp_sim_matrix, go_genes_normalize):
	
	coexp_2nd_order = np.linalg.matrix_power(coexp_sim_matrix,2)
	go_2nd_order = np.linalg.matrix_power(go_genes_normalize,2)
	coexp_go = np.dot(coexp_sim_matrix, go_genes_normalize)
	go_coexp = np.dot(go_genes_normalize, coexp_sim_matrix)

	return (go_coexp, coexp_go, go_2nd_order, coexp_2nd_order)



def parallel_disease(disease_row, disease, fold, coexp_matrix, go_genes_normalize, go_coexp, coexp_go, go_2nd_order, coexp_2nd_order):
	feature_mat = feature_matrix_0(disease_row, fold, disease, coexp_matrix, go_genes_normalize)
	feature_mat_2 = feature_matrix_1(disease_row, fold, disease, go_coexp, coexp_go, go_2nd_order, coexp_2nd_order)
	features = np.concatenate((feature_mat, feature_mat_2), axis=1)
	df = pd.DataFrame(features)
	df.to_csv(feat_directory + '/' + disease + '.txt', sep='\t', header=None)

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

DATA_ROOT = '../CM-for-disease-gene-association/data/'

with h5py.File(DATA_ROOT + 'random_walk_score.h5' ,'r') as hf:
	disease_score_matrix = hf.get('disease_probability')
	disease_score_matrix = np.array(disease_score_matrix, dtype='float32')

coexp_sim_matrix = np.load(DATA_ROOT + 'Coexpression_score.npy').astype('float32')
go_sim_matrix = np.load(DATA_ROOT + 'GO_similarity.npy').astype('float32')

# Scale Matrix to range:[0,1]
min_max_go_scaler = preprocessing.MinMaxScaler()
go_sim_scaled_matrix = min_max_scaler.fit_transform(go_sim_matrix)

min_max_co_scaler = preprocessing.MinMaxScaler()
coexp_sim_scaled_matrix = min_max_scaler.fit_transform(coexp_sim_matrix)

with open(DATA_ROOT + 'disgenet/train_disease_gene.dict','r') as f:
	train_disease_gene_map = json.load(f)
with open(DATA_ROOT + 'disgenet/test_disease_gene.dict','r') as f:
	test_disease_gene_map = json.load(f)
with open(DATA_ROOT + 'disgenet/gene_nodes.dict','r') as f:
	gene_nodes = json.load(f)
with open(DATA_ROOT + 'disgenet/disease2id.dict','r') as f:
	disease2id = json.load(f)


inv_gene_nodes = {v:n for n,v in gene_nodes.items()}


feature_directory  = DATA_ROOT + 'multi-modal-features/'
nm_folds = 3
print('Computing Higher Order Coexpression and GO Matrix')
go_coexp, coexp_go, go_2nd_order, coexp_2nd_order = power_matrix_features(coexp_sim_scaled_matrix, go_sim_scaled_matrix)

for fold in range(nm_folds):
	feat_directory = feature_directory + feat + '/fold'+ str(fold)
	if len(os.listdir(feat_directory))<len(disease2id[str(fold)]):
		print('Creating Features, type: {} fold {} '.format(feat, fold))
		disease_row = disease2id[str(fold)]
		processes = [multiprocessing.Process(target=parallel_disease, args=(disease_row, disease, fold, coexp_matrix, go_genes_normalize, go_coexp, coexp_go, go_2nd_order, coexp_2nd_order)) for disease in disease_row]
		for p in processes:
			p.start()
		for p in processes:
			p.join()