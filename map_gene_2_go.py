import numpy as np
import pandas as pd

DATA_ROOT = '../CVSM-for-Gene-Disease-Prioritization/data/'

def create_annotation(annotation_mapping, name_entrez):
	data = pd.read_csv(annotation_mapping, header=None, sep='\t', skiprows=34)
	gene_name = data[2].values
	gene_annot = data[4].values
	gene_annot_map = dict()
	for gene,annot in zip(gene_name, gene_annot):
		if gene in gene_annot_map:
			gene_annot_map[gene].append(annot)
		else:
			gene_annot_map[gene] = [annot]
	gene_info = pd.read_csv(name_entrez, sep='\t')
	gene_symbol = [gene.upper() for gene in np.asarray(gene_info['Symbol'].values, dtype='str')]
	entrez =  gene_info['GeneID'].values
	tax_id = gene_info['#tax_id'].values
	symbol_entrez_map = dict()
	for tx,ent,sym in zip(tax_id,entrez,gene_symbol):
		if tx==9606:
			symbol_entrez_map[sym] = ent
	with open(DATA_ROOT + 'ontologies1/gene_go_annotation.tab','w') as f:
		for gene,annot in gene_annot_map.items():
			if gene in symbol_entrez_map:
				gene = symbol_entrez_map[gene]
				annot_set = '\t'.join(el.replace(':','_') for el in annot)
				f.write(str(gene) + '\t' + annot_set + '\n')


annotation_mapping = DATA_ROOT + 'goa_human.gaf'
name_entrez = DATA_ROOT + 'gene_info'
create_annotation(annotation_mapping, name_entrez)
