#!/bin/bash

dataset="PubMed"
folder="data/${dataset}/"
pattern_file="${folder}pattern/"
node_file="${folder}node.dat"
config_file="${folder}config.dat"
link_file="${folder}link.dat"
label_file="${folder}label.dat"
emb_file="${folder}emb.dat"


size=50
nhead="8"
dropout=0.4
neigh_por=0.6
lr=0.005
weight_decay=0.0005
batch_size=256
epochs=500
device="cuda"

attributed="True"
supervised="True"


if [ "${dataset}" = "T20H" ] || [ "${dataset}" = "T15S" ]
then
    ratio="5v5"
    label_file="${folder}label_TaxPayer${ratio}.dat"
    python3 src/T20HModel/main.py --node=${node_file} --link=${link_file} --config=${config_file} --label=${label_file} --output=${emb_file} --device=${device} --meta=${meta} --size=${size} --nhead=${nhead} --dropout=${dropout} --neigh-por=${neigh_por} --lr=${lr} --weight-decay=${weight_decay} --batch-size=${batch_size} --epochs=${epochs} --ratio=${ratio} --attributed=${attributed} --supervised=${supervised}
elif [ "${dataset}" = "PubMed" ]
then
    rpt="CDD,GDD,SDD,GDCD,SDCD,SDGD"
    nhead="6"
    meta="1,2,4,8"
    python src/PubMedModel/main.py --node=${node_file} --link=${link_file} --config=${config_file} --label=${label_file} --output=${emb_file} --device=${device} --rpt=${rpt} --meta=${meta} --size=${size} --nhead=${nhead} --dropout=${dropout} --neigh-por=${neigh_por} --lr=${lr} --weight-decay=${weight_decay} --batch-size=${batch_size} --epochs=${epochs} --attributed=${attributed} --supervised=${supervised}
elif [ "${dataset}" = "DBLP" ]
then
    rpt="VAA,PAA,YAA,PPAA"
    nhead="4"
    meta="1,2,3,4,5"
    python src/DBLPModel/main.py --node=${node_file} --link=${link_file} --config=${config_file} --label=${label_file} --output=${emb_file} --device=${device} --rpt=${rpt} --meta=${meta} --size=${size} --nhead=${nhead} --dropout=${dropout} --neigh-por=${neigh_por} --lr=${lr} --weight-decay=${weight_decay} --batch-size=${batch_size} --epochs=${epochs} --attributed=${attributed} --supervised=${supervised}    
fi


if [ "${dataset}" = "T20H" ] || [ "${dataset}" = "T15S" ]
then
    name="${folder}Att_Sup_${ratio}.emb.dat"
else
    name="${folder}Att_Sup.emb.dat"
fi

cp "${emb_file}" "${name}"
