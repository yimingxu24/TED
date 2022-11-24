import argparse
from transform_model import *


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, type=str, help='Targeting dataset.', 
                        choices=['T20H', 'T15S', 'PubMed', 'DBLP'])
    parser.add_argument('-model', required=True, type=str, help='Targeting model.', 
                        choices=['HIN2Vec', 'PTE', 'metapath2vec-ESim', 'TransE', 'ConvE', 'DistMult', 'ComplEx', 'TEDM-PU', 'PUNE', 'GCN', 'HAN','MAGNN', 'R-GCN', 'TED'])
    parser.add_argument('-attributed', required=True, type=str, help='Only TEDM-PU, PUNE, GCN, R-GCN, HAN, MAGNN, and TED support attributed training.',
                        choices=['True','False'])
    parser.add_argument('-supervised', required=True, type=str, help='Only TEDM-PU, PUNE, GCN, R-GCN, HAN, MAGNN, and TED support semi-supervised training.', 
                        choices=['True','False'])
    
    return parser.parse_args()


def check(args):
    
    if args.attributed=='True':
        if args.model not in ['TEDM-PU', 'PUNE', 'GCN', 'HAN','MAGNN', 'R-GCN', 'TED']:
            print(f'{args.model} does not support attributed training!')
            print('Only TEDM-PU, PUNE, GCN, R-GCN, HAN, MAGNN, and TED support attributed training!')
            return False
        if args.dataset not in ['T20H', 'T15S', 'PubMed', 'DBLP']:
            print(f'{args.dataset} does not support attributed training!')
            print('Only T20H, T15S, PubMed and DBLP support attributed training!')
            return False
        
    if args.supervised=='True':
        if args.model not in ['TEDM-PU', 'PUNE', 'GCN', 'HAN','MAGNN', 'R-GCN', 'TED']:
            print(f'{args.model} does not support semi-supervised training!')
            print('Only TEDM-PU, PUNE, GCN, R-GCN, HAN, MAGNN, and TED support semi-supervised training!')
            return False
        
    return True


def main():
    
    args = parse_args()
    
    if not check(args):
        return
    
    print('Transforming {} to {} input format for {}, {} training!'.format(args.dataset, args.model, 'attributed' if args.attributed=='True' else 'unattributed', 'semi-supervised' if args.supervised=='True' else 'unsupervised'))
    
    if args.model=='HIN2Vec': hin2vec_convert(args.dataset)
    elif args.model=='PTE': pte_convert(args.dataset)
    elif args.model=='metapath2vec-ESim': metapath2vec_esim_convert(args.dataset)
    elif args.model=='TransE': transe_convert(args.dataset)
    elif args.model=='ConvE': conve_convert(args.dataset) 
    elif args.model=='DistMult': distmult_convert(args.dataset)
    elif args.model=='ComplEx': complex_convert(args.dataset)
    elif args.model=='HAN': han_convert(args.dataset, args.attributed, args.supervised)
    elif args.model=='MAGNN': magnn_convert(args.dataset, args.attributed, args.supervised)
    elif args.model=='R-GCN': rgcn_convert(args.dataset, args.attributed, args.supervised)
    elif args.model=='TED': ted_convert(args.dataset, args.attributed, args.supervised)   
 
    print('Data transformation finished!')
    
    return


if __name__=='__main__':
    main()
