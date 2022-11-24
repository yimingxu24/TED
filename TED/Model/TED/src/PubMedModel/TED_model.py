import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InstanceAggLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha, device):
        super(InstanceAggLayer, self).__init__()

        self.dropout = dropout
        self.device = device

        self.P_disease = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.P_disease.data, gain=1.414)
        self.P_gene = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.P_gene.data, gain=1.414)
        self.P_chemical = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.P_chemical.data, gain=1.414)
        self.P_species = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.P_species.data, gain=1.414)

        self.W_DD = nn.Parameter(torch.zeros(size=(out_dim * 2, out_dim)))
        nn.init.xavier_uniform_(self.W_DD.data, gain=1.414)
        self.W_CDD = nn.Parameter(torch.zeros(size=(out_dim * 3, out_dim)))
        nn.init.xavier_uniform_(self.W_CDD.data, gain=1.414)
        self.W_GDD = nn.Parameter(torch.zeros(size=(out_dim * 3, out_dim)))
        nn.init.xavier_uniform_(self.W_GDD.data, gain=1.414)
        self.W_SDD = nn.Parameter(torch.zeros(size=(out_dim * 3, out_dim)))
        nn.init.xavier_uniform_(self.W_SDD.data, gain=1.414)
        self.W_GDCD = nn.Parameter(torch.zeros(size=(out_dim * 4, out_dim)))
        nn.init.xavier_uniform_(self.W_GDCD.data, gain=1.414)
        self.W_SDCD = nn.Parameter(torch.zeros(size=(out_dim * 4, out_dim)))
        nn.init.xavier_uniform_(self.W_SDCD.data, gain=1.414)
        self.W_SDGD = nn.Parameter(torch.zeros(size=(out_dim * 4, out_dim)))
        nn.init.xavier_uniform_(self.W_SDGD.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, disease_feats, gene_feats, chemical_feats, species_feats, trans_adj_list, pattern_name):
        f_disease = torch.mm(disease_feats, self.P_disease)
        if gene_feats != None or chemical_feats != None or species_feats != None:
            f_gene = torch.mm(gene_feats, self.P_gene)
            f_chemical = torch.mm(chemical_feats, self.P_chemical)
            f_species = torch.mm(species_feats, self.P_species)

        if pattern_name == 'CDD':
            concat = torch.cat([f_disease[trans_adj_list[0]], f_disease[trans_adj_list[1]], f_chemical[trans_adj_list[2]]], dim=1)
            instance = self.leakyrelu(torch.matmul(concat, self.W_CDD))
        elif pattern_name == 'GDD':
            concat = torch.cat([f_disease[trans_adj_list[0]], f_disease[trans_adj_list[1]], f_gene[trans_adj_list[2]]], dim=1)
            instance = self.leakyrelu(torch.matmul(concat, self.W_GDD))
        elif pattern_name == 'SDD':
            concat = torch.cat([f_disease[trans_adj_list[0]], f_disease[trans_adj_list[1]], f_species[trans_adj_list[2]]], dim=1)
            instance = self.leakyrelu(torch.matmul(concat, self.W_SDD))
        elif pattern_name == 'GDCD':
            concat = torch.cat([f_disease[trans_adj_list[0]], f_disease[trans_adj_list[1]], f_gene[trans_adj_list[2]], f_chemical[trans_adj_list[3]]], dim=1)
            instance = self.leakyrelu(torch.matmul(concat, self.W_GDCD))
        elif pattern_name == 'SDCD':
            concat = torch.cat([f_disease[trans_adj_list[0]], f_disease[trans_adj_list[1]], f_species[trans_adj_list[2]], f_chemical[trans_adj_list[3]]], dim=1)
            instance = self.leakyrelu(torch.matmul(concat, self.W_SDCD))
        elif pattern_name == 'SDGD':
            concat = torch.cat([f_disease[trans_adj_list[0]], f_disease[trans_adj_list[1]], f_species[trans_adj_list[2]], f_gene[trans_adj_list[3]]], dim=1)
            instance = self.leakyrelu(torch.matmul(concat, self.W_SDGD))
        else:
            concat = torch.cat([f_disease[trans_adj_list[0]], f_disease[trans_adj_list[1]]], dim=1)
            instance = self.leakyrelu(torch.matmul(concat, self.W_DD))

        return instance


class InnerModel(nn.Module):

    def __init__(self, rpt, meta, nchannel, in_dim, out_dim, dropout, alpha, device, nheads, nlayer, neigh_por):
        super(InnerModel, self).__init__()
        self.rpt = rpt
        self.meta = meta
        self.pattern = rpt + meta
        self.nchannel = nchannel
        self.neigh_por = neigh_por
        self.nlayer = nlayer
        self.dropout = dropout
        self.device = device
        self.nheads = nheads
        self.out_dim = out_dim

        self.A = nn.Parameter(torch.zeros(size=(nchannel, out_dim * nheads[0]))).to(device)
        nn.init.xavier_uniform_(self.A.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(alpha)

        self.ins_aggs = []
        for i in range(nlayer):
            if i == 0:
                curr_in_dim = in_dim
            else:
                curr_in_dim = out_dim * nheads[i - 1]
            layer_ins_aggs = []

            for j in range(nheads[i]):
                layer_ins_aggs.append(InstanceAggLayer(curr_in_dim, out_dim, dropout, alpha, device).to(device))
                self.add_module('ins_aggs_layer{}_head{}'.format(i, j), layer_ins_aggs[j])
            self.ins_aggs.append(layer_ins_aggs)


    def sample(self, adj, samples, pattern_name):
        sample_list, adj_list = [samples], []
        for _ in range(self.nlayer):
            new_samples, new_adjs = set(sample_list[-1]), []
            gene_samples, chemical_samples, species_samples = set(), set(), set()
            if pattern_name in self.rpt:
                for sample in sample_list[-1]:
                    neighbor_size = adj[1][sample]
                    nneighbor = int(self.neigh_por * neighbor_size) + 1
                    start = adj[1][:sample].sum()
                    if neighbor_size <= nneighbor:
                        samples_instance_number = range(neighbor_size)
                    else:
                        samples_instance_number = random.sample(range(neighbor_size), nneighbor)
                        samples_instance_number.sort()
                    for i in samples_instance_number:
                        if pattern_name == 'CDD':
                            chemical_samples.add(adj[0][(start + i) * 2])
                            new_samples.add(adj[0][(start + i) * 2 + 1])
                            curr_new_adjs = np.stack(([sample], [adj[0][(start + i) * 2 + 1]], [adj[0][(start + i) * 2]]), axis=-1).tolist()
                            new_adjs.append(curr_new_adjs)
                        elif pattern_name == 'GDD':
                            gene_samples.add(adj[0][(start + i) * 2])
                            new_samples.add(adj[0][(start + i) * 2 + 1])
                            curr_new_adjs = np.stack(([sample], [adj[0][(start + i) * 2 + 1]], [adj[0][(start + i) * 2]]), axis=-1).tolist()
                            new_adjs.append(curr_new_adjs)
                        elif pattern_name == 'SDD':
                            species_samples.add(adj[0][(start + i) * 2])
                            new_samples.add(adj[0][(start + i) * 2 + 1])
                            curr_new_adjs = np.stack(([sample], [adj[0][(start + i) * 2 + 1]], [adj[0][(start + i) * 2]]), axis=-1).tolist()
                            new_adjs.append(curr_new_adjs)
                        elif pattern_name == 'GDCD':
                            new_samples.add(adj[0][(start + i) * 3])
                            gene_samples.add(adj[0][(start + i) * 3 + 1])
                            chemical_samples.add(adj[0][(start + i) * 3 + 2])
                            curr_new_adjs = np.stack(([sample], [adj[0][(start + i) * 3]], [adj[0][(start + i) * 3 + 1]], [adj[0][(start + i) * 3 + 2]]), axis=-1).tolist()
                            new_adjs.append(curr_new_adjs)
                        elif pattern_name == 'SDCD':
                            new_samples.add(adj[0][(start + i) * 3])
                            species_samples.add(adj[0][(start + i) * 3 + 1])
                            chemical_samples.add(adj[0][(start + i) * 3 + 2])
                            curr_new_adjs = np.stack(([sample], [adj[0][(start + i) * 3]], [adj[0][(start + i) * 3 + 1]], [adj[0][(start + i) * 3 + 2]]), axis=-1).tolist()
                            new_adjs.append(curr_new_adjs)
                        elif pattern_name == 'SDGD':
                            new_samples.add(adj[0][(start + i) * 3])
                            species_samples.add(adj[0][(start + i) * 3 + 1])
                            gene_samples.add(adj[0][(start + i) * 3 + 2])
                            curr_new_adjs = np.stack(([sample], [adj[0][(start + i) * 3]], [adj[0][(start + i) * 3 + 1]], [adj[0][(start + i) * 3 + 2]]), axis=-1).tolist()
                            new_adjs.append(curr_new_adjs)

                sample_list.append(np.array(list(new_samples)))
                sample_list.append(np.array(list(gene_samples)))
                sample_list.append(np.array(list(chemical_samples)))
                sample_list.append(np.array(list(species_samples)))
                adj_list.append(np.array([pair for chunk in new_adjs for pair in chunk]).T)
            else:
                for sample in sample_list[-1]:
                    neighbor_size = adj[1][sample]
                    nneighbor = int(self.neigh_por * neighbor_size) + 1
                    start = adj[1][:sample].sum()

                    if neighbor_size <= nneighbor:
                        curr_new_samples = adj[0][start:start + neighbor_size]
                    else:
                        curr_new_samples = random.sample(adj[0][start:start + neighbor_size].tolist(), nneighbor)
                    new_samples = new_samples.union(set(curr_new_samples))
                    curr_new_adjs = np.stack(([sample] * len(curr_new_samples), curr_new_samples), axis=-1).tolist()
                    curr_new_adjs.append([sample, sample])
                    new_adjs.append(curr_new_adjs)

                sample_list.append(np.array(list(new_samples)))
                adj_list.append(np.array([pair for chunk in new_adjs for pair in chunk]).T)

        return sample_list, adj_list


    def transform(self, sample_list, adj_list, pattern_name):
        trans_adj_list, target_index_outs = [], []
        base_index_dict = {k: v for v, k in enumerate(sample_list[0])}
        for i, adjs in enumerate(adj_list):
            if adjs.size != 0:
                if pattern_name in self.rpt:
                    # adjs.size != 0
                    disease_index_dict, gene_index_dict, chemical_index_dict, species_index_dict = {k: v for v, k in enumerate(sample_list[1])}, {k: v for v, k in enumerate(sample_list[2])}, {k: v for v, k in enumerate(sample_list[3])}, {k: v for v, k in enumerate(sample_list[4])}
                    target_index_outs.append([base_index_dict[k] for k in adjs[0]])

                    if pattern_name == 'CDD':
                        pattern_index1, pattern_index2, pattern_index3 = [disease_index_dict[k] for k in adjs[0]], [disease_index_dict[k] for k in adjs[1]], [chemical_index_dict[k] for k in adjs[2]]
                        trans_adj_list.append([pattern_index1, pattern_index2, pattern_index3])
                    elif pattern_name == 'GDD':
                        pattern_index1, pattern_index2, pattern_index3 = [disease_index_dict[k] for k in adjs[0]], [disease_index_dict[k] for k in adjs[1]], [gene_index_dict[k] for k in adjs[2]]
                        trans_adj_list.append([pattern_index1, pattern_index2, pattern_index3])
                    elif pattern_name == 'SDD':
                        pattern_index1, pattern_index2, pattern_index3 = [disease_index_dict[k] for k in adjs[0]], [disease_index_dict[k] for k in adjs[1]], [species_index_dict[k] for k in adjs[2]]
                        trans_adj_list.append([pattern_index1, pattern_index2, pattern_index3])
                    elif pattern_name == 'GDCD':
                        pattern_index1, pattern_index2, pattern_index3, pattern_index4 = [disease_index_dict[k] for k in adjs[0]], [disease_index_dict[k] for k in adjs[1]], [gene_index_dict[k] for k in adjs[2]], [chemical_index_dict[k] for k in adjs[3]]
                        trans_adj_list.append([pattern_index1, pattern_index2, pattern_index3, pattern_index4])
                    elif pattern_name == 'SDCD':
                        pattern_index1, pattern_index2, pattern_index3, pattern_index4 = [disease_index_dict[k] for k in adjs[0]], [disease_index_dict[k] for k in adjs[1]], [species_index_dict[k] for k in adjs[2]], [chemical_index_dict[k] for k in adjs[3]]
                        trans_adj_list.append([pattern_index1, pattern_index2, pattern_index3, pattern_index4])
                    elif pattern_name == 'SDGD':
                        pattern_index1, pattern_index2, pattern_index3, pattern_index4 = [disease_index_dict[k] for k in adjs[0]], [disease_index_dict[k] for k in adjs[1]], [species_index_dict[k] for k in adjs[2]], [gene_index_dict[k] for k in adjs[3]]
                        trans_adj_list.append([pattern_index1, pattern_index2, pattern_index3, pattern_index4])
                else:
                    target_index_outs.append([base_index_dict[k] for k in adjs[0]])
                    base_index_dict = {k: v for v, k in enumerate(sample_list[i + 1])}

                    neighbor_index_out, neighbor_index_in = [base_index_dict[k] for k in adjs[0]], [base_index_dict[k] for k in adjs[1]]
                    trans_adj_list.append([neighbor_index_out, neighbor_index_in])

        return target_index_outs, trans_adj_list


    def forward(self, feats, adj, samples, embeddings_gene, embeddings_chemical, embeddings_species, pattern_name):
        sample_list, adj_list = self.sample(adj, samples, pattern_name)
        target_index_outs, trans_adj_list = self.transform(sample_list, adj_list, pattern_name)
        disease_feats = feats[sample_list[1]]

        if len(sample_list) != 2:
            gene_feats = embeddings_gene[sample_list[2]]
            chemical_feats = embeddings_chemical[sample_list[3]]
            species_feats = embeddings_species[sample_list[4]]

        for i, layer_ins_aggs in enumerate(self.ins_aggs):
            disease_feats = F.dropout(disease_feats, self.dropout, training=self.training)
            if pattern_name in self.rpt:
                gene_feats = F.dropout(gene_feats, self.dropout, training=self.training)
                chemical_feats = F.dropout(chemical_feats, self.dropout, training=self.training)
                species_feats = F.dropout(species_feats, self.dropout, training=self.training)
                instance = torch.cat([agg(disease_feats, gene_feats, chemical_feats, species_feats, trans_adj_list[-i - 1], pattern_name) for agg in layer_ins_aggs], dim=1)
            else:
                instance = torch.cat([agg(disease_feats, None, None, None, trans_adj_list[-i - 1], pattern_name) for agg in layer_ins_aggs], dim=1)

        innerAtt = self.leaky_relu(torch.matmul(instance, self.A[self.pattern.index(pattern_name)].view(-1, 1)).squeeze(1))
        att = torch.full((len(samples), instance.shape[0]), -9e15).to(self.device)
        index = range(len(trans_adj_list[0][0]))
        att[target_index_outs[0], index] = innerAtt
        att = F.softmax(att, dim=1)
        att = F.dropout(att, self.dropout, training=self.training)
        inner_feats = torch.matmul(att, instance)

        return F.elu(inner_feats)


class CrossModel(nn.Module):

    def __init__(self, nchannel, in_dim, att_dim, device, alpha):
        super(CrossModel, self).__init__()
        self.nchannel = nchannel
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.device = device

        self.P_q = nn.Parameter(torch.zeros(size=(200, att_dim)))
        nn.init.xavier_uniform_(self.P_q.data, gain=1.414)
        self.cross_att = nn.Parameter(torch.zeros(size=(nchannel, att_dim * 2)))
        nn.init.xavier_uniform_(self.cross_att.data, gain=1.414)
        self.linear_block = nn.Sequential(nn.Linear(in_dim, att_dim), nn.Tanh())
        self.leaky_relu = nn.LeakyReLU(alpha)
    def forward(self, inner_pattern_out, samples_feats, nnode):
        new_inner = torch.cat([self.linear_block(inner_pattern_out[i]).view(1, nnode, -1) for i in range(self.nchannel)], dim=0)
        samples_feats = self.leaky_relu(torch.mm(samples_feats, self.P_q))

        crossAtt = []
        for i in range(self.nchannel):
            att = torch.cat([samples_feats, new_inner[i]], dim=1)
            crossAtt.append(torch.matmul(att, self.cross_att[i].view(-1, 1)))

        crossAtt = self.leaky_relu(torch.stack(crossAtt, dim=1).view(nnode, self.nchannel) / torch.tensor([math.sqrt(self.att_dim)], dtype=torch.float32).to(self.device))
        crossAtt = F.softmax(crossAtt, dim=1)

        aggre_hid = []
        for i in range(nnode):
            aggre_hid.append(torch.mm(crossAtt[i].view(1, -1), new_inner[:, i, :]))
        aggre_hid = torch.stack(aggre_hid, dim=0).view(nnode, self.att_dim)

        return aggre_hid


class TEDModel(nn.Module):

    def __init__(self, rpt, meta, nchannel, nfeat, nhid, nlabel, nlayer, nheads, neigh_por, dropout, alpha, device):
        super(TEDModel, self).__init__()
        self.rpt = rpt
        self.meta = meta
        self.pattern = rpt + meta
        self.InnerModels = [InnerModel(rpt, meta, nchannel, nfeat, nhid, dropout, alpha, device, nheads, nlayer, neigh_por) for i in range(nchannel)]
        self.CrossModels = CrossModel(nchannel, nhid * nheads[-1], nhid, device, alpha).to(device)

        for i, inner_att in enumerate(self.InnerModels):
            self.add_module('inner_att_{}'.format(i), inner_att)
        self.add_module('cross_att', self.CrossModels)

        self.supervised = False
        if nlabel != 0:
            self.supervised = True
            self.LinearLayer = torch.nn.Linear(nhid, nlabel).to(device)
            self.add_module('linear', self.LinearLayer)

    def forward(self, x, adjs, samples, embeddings_gene, embeddings_chemical, embeddings_species):
        inner_out = []
        for i, patt_agg in enumerate(self.InnerModels):
            inner_out.append(patt_agg(x, adjs[i], samples, embeddings_gene, embeddings_chemical, embeddings_species, self.pattern[i]))

        samples_feats = x[samples]
        inner_pattern_out = torch.stack(inner_out, dim=0)
        aggre_hid = self.CrossModels(inner_pattern_out, samples_feats, len(samples))

        if self.supervised:
            pred = self.LinearLayer(aggre_hid)
        else:
            pred = None

        return aggre_hid, pred