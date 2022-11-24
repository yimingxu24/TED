## Model: TED

### Input

*Stage 2: Transform* prepares 5 input files stored in ```data/{dataset}``` or ```data/{dataset}/{pattern}```:
- ```data/{dataset}/{pattern}```: Save all RPT group instances. The first line is the node type.
- ```node.dat```: For attributed training, each line is formatted as ```{node_id}\t{node_type}\t{node_attributes}``` where entries in ```{node_attributes}``` are separated by ```,```.
- ```link.dat```: Each line is formatted as ```{head_node_id}\t{tail_node_id}\t{link_type}```.
- ```config.dat```: The first line specifies the targeting node type. The second line specifies the targeting link type. The third line specifies the information related to each link type, e.g., ```{head_node_type},{tail_node_type},{link_type}\t```.
- ```label.dat```: This file is only needed for semi-supervised training. Each line is formatted as ```{node_id}\t{node_label}```.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```Att_Sup_*v*.emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.