## Evaluate

This stage evaluates the output embeddings based on specific tasks.

Users need to specify the following parameters in ```evaluate.sh```:
- **dataset**: choose from ```T20H```, ```T15S```, ```PubMed``` and ```DBLP```;
- **model**: choose from ```HIN2Vec```, ```PTE```, ```metapath2vec-ESim```, ```TransE```, ```ConvE```, ```DistMult```, ```ComplEx```, ```TEDM-PU```, ```PUNE```, ```GCN```, ```HAN```, ```MAGNN```, ```R-GCN```, ```TED```;
- **attributed**: choose ```True``` for attributed training or ```False``` for unattributed training;
- **supervised**: choose ```True``` for semi-supervised training or ```False``` for unsupervised training.
- **task**: We treat tax evasion detection as a classification task.

*Note: Only PU-Learning and Message-Passing Methods (```TEDM-PU```, ```PUNE```, ```GCN```, ```HAN```, ```MAGNN```, ```R-GCN```, ```TED```) support attributed or semi-supervised training.* <br /> 

**Node Classification**: <br /> 
We train a separate linear Support Vector Machine (LinearSVC) based on the learned embeddings on training set and predict on the test set.


Run ```bash evaluate.sh``` to complete *Stage 4: Evaluate*.

The evaluation results are stored in ```record.dat``` of the corresponding dataset. 