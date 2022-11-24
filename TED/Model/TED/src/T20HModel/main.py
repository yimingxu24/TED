import time
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from TED_model import *
from utils import *

def parse_args():
    
    parser = argparse.ArgumentParser(description='MEGA')
    
    parser.add_argument('--node', type=str, default='../data/T20H/node.dat')
    parser.add_argument('--link', type=str, default='../data/T20H/link.dat')
    parser.add_argument('--config', type=str, default='../data/T20H/config.dat')
    parser.add_argument('--label', type=str, default='../data/T20H/label.dat')
    parser.add_argument('--output', type=str, default='../data/T20H/Att_Sup.emb.dat')
    parser.add_argument('--rpt', type=str, default="PCC,PCCC,PCIC,PCPC,PCPCC")
    parser.add_argument('--meta', type=str, default="0,2,3,4")
    
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument('--nhead', type=str, default='8')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.4)
    
    parser.add_argument('--neigh-por', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--ratio', type=str, default='5v5')
    
    parser.add_argument('--attributed', type=str, default="True")
    parser.add_argument('--supervised', type=str, default="True")
    
    return parser.parse_args()


def output(args, embeddings, id_name):
    with open(args.output, 'w') as file:
        file.write(f'size={args.size}, nhead={args.nhead}, dropout={args.dropout}, neigh-por={args.neigh_por}, lr={args.lr}, batch-size={args.batch_size}, epochs={args.epochs}, attributed={args.attributed}, supervised={args.supervised}\n')

        for nid, name in id_name.items():
            file.write('{}\t{}\n'.format(name, ' '.join(embeddings[nid].astype(str))))

    
def main():
    torch.cuda.set_device(0)
    args = parse_args()
    set_seed(args.seed, args.device)
    if args.supervised=='True':
        adjs, id_name, features, item_attr, person_attr = load_data_semisupervised(args, args.node, args.link, args.config, args.rpt.split(','), list(map(lambda x: int(x), args.meta.split(','))))
        train_pool, train_label, nlabel, multi = load_label(args.label, id_name)
    elif args.supervised=='False':
        return 0
  
    nhead = list(map(lambda x: int(x), args.nhead.split(',')))  
    nnode, nchannel, nlayer, nfeat, rpt, meta = len(id_name), len(adjs), len(nhead), features.shape[1], args.rpt.split(','), list(map(lambda x: int(x), args.meta.split(',')))
    model = TEDModel(rpt, meta, nchannel, nfeat if args.attributed == 'True' else args.size, args.size, nlabel, nlayer, nhead, args.neigh_por, args.dropout, args.alpha, args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.attributed=='True':
        embeddings, embeddings_item, embeddings_person = torch.from_numpy(features).to(args.device), torch.from_numpy(item_attr).to(args.device), torch.from_numpy(person_attr).to(args.device)
    elif args.supervised=='False':
        return 0

    if args.supervised=='True':
        train_label = torch.from_numpy(train_label.astype(np.float32)).to(args.device)

    model.train()
    losses = []
    for epoch in range(args.epochs):
        if args.supervised=='True':
            # curr_index = np.sort(np.random.choice(np.arange(len(train_pool)), args.batch_size, replace=False))     
            curr_index = np.sort(np.random.choice(np.arange(len(train_pool)), 1, replace=False))  # The subgraph data cannot support 256 batch size.
            curr_batch = train_pool[curr_index]
            updates, pred = model(embeddings, adjs, curr_batch, embeddings_item, embeddings_person)
            if multi:
                loss = F.binary_cross_entropy(torch.sigmoid(pred), train_label[curr_index])
            else:
                loss = F.nll_loss(F.log_softmax(pred, dim=1), train_label[curr_index].long())

        loss.backward()
        losses.append(loss.item())

        if (epoch+1)%10==0 or epoch+1==args.epochs:
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'finish epoch {epoch}, loss {np.mean(losses)}', flush=True)
            optimizer.step()
            optimizer.zero_grad()
            losses = []
        else:
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'finish epoch {epoch}', flush=True)
    

    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'output embedding', flush=True)
    model.eval()
    outbatch_size = 2*args.batch_size
    rounds = math.ceil(nnode/outbatch_size)
    outputs = np.zeros((nnode, args.size)).astype(np.float32)

    with torch.no_grad():
        for index, i in enumerate(range(rounds)):
            seed_nodes = np.arange(i*outbatch_size, min((i+1)*outbatch_size, nnode))
            embs, _ = model(embeddings, adjs, seed_nodes, embeddings_item, embeddings_person)
            outputs[seed_nodes] = embs.detach().cpu().numpy()
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + f'finish output batch {index} -> {rounds}', flush=True)
        output(args, outputs, id_name)
    

if __name__ == '__main__':
    main()
