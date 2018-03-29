# CM-for-disease-gene-association

This repository implements a multi-modal approach to address the problem of Disease-Gene-Association. It combines three modes of data: 
Random-Walk on PPI, GO similarity and CO expressions.

Evaluation is done against curated-DisGeNet.

## Running The Experiments
Follow following steps to generate features.
Step:1
Create Gene to Function Mapping 
`python map_gene_2_go.py`.
Step:2
Compute pairwise GO Similarity 
`groovy Sim_Pairwise_GO.groovy`
Step:3
Compute Stationary Probability vector for disease with 3 fold cross validation.
`python random_walk_disease_gene.py`
Step:4
Compute Coexpression Similarity Matrix from coexpression score
`python coexpression_score.py`
Step:5
Compute joint features from three modes.
`feature_computation.py` 
Step:6
Evaluation
`ToDo`

