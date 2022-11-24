## Transform

This stage transforms a dataset from its original format to the training input format.

Users need to specify the following parameters in ```transform.sh```:
- **dataset**: choose from ```T20H```, and ```T15S```;
- **model**: choose from ```HIN2Vec```, ```PTE```, ```metapath2vec-ESim```, ```TransE```, ```ConvE```, ```DistMult```, ```ComplEx```, ```TEDM-PU```, ```PUNE```, ```GCN```, ```HAN```, ```MAGNN```, ```R-GCN```, ```TED```;
- **attributed**: choose ```True``` for attributed training or ```False``` for unattributed training;
- **supervised**: choose ```True``` for semi-supervised training or ```False``` for unsupervised training.

*Note: Only PU-Learning and Message-Passing Methods (```TEDM-PU```, ```PUNE```, ```GCN```, ```HAN```, ```MAGNN```, ```R-GCN```, ```TED```) support attributed or semi-supervised training.* <br /> 

Run ```bash transform.sh``` to complete *Stage 2: Transform*.