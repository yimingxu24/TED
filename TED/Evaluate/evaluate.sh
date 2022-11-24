#!/bin/bash

dataset='PubMed'
model='TED'
attributed='True'
supervised='True'
# r='5v5'

python evaluate.py -dataset ${dataset} -model ${model} -attributed ${attributed} -supervised ${supervised}
