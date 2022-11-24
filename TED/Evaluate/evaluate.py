import argparse
from node_classification import *


data_folder, model_folder = '../Data', '../Model'
emb_file, record_file = 'emb.dat', 'record.dat'


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
    parser.add_argument('-ratio', type=str, help='5v5, 4v6, 3v7, 2v8, 1v9')
    
    return parser.parse_args()


def load(emb_file_path):

    emb_dict = {}
    with open(emb_file_path,'r') as emb_file:        
        for i, line in enumerate(emb_file):
            if i == 0:
                train_para = line[:-1]
            else:
                index, emb = line[:-1].split('\t')
                emb_dict[index] = np.array(emb.split()).astype(np.float32)
        
    return train_para, emb_dict  


def record(args, all_tasks, train_para, all_scores):    
    
    with open(f'{data_folder}/{args.dataset}/{record_file}', 'a', encoding='utf-8') as file:
        for task, score in zip(all_tasks, all_scores):
            if args.dataset == 'T20H' or args.dataset == 'T15S':
                file.write(f'model={args.model}, task={task}, attributed={args.attributed}, supervised={args.supervised}, Train P:N ratio : {args.ratio}\n')
                file.write(f'{train_para}\n')
                file.write(f'f1={score[0]:.4f}, acc={score[1]:.4f}, pre={score[2]:.4f}, rec={score[3]:.4f}, auc={score[4]:.4f}\n')
            else:
                file.write(f'model={args.model}, task={task}, attributed={args.attributed}, supervised={args.supervised}\n')
                file.write(f'{train_para}\n')
                file.write(f'Macro-F1={score[0]:.4f}, Micro-F1={score[1]:.4f}\n')

            file.write('\n')
        
    return


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
    
    print('Load Embeddings!')
    
    label_file = 'label_TaxPayer{}.dat'.format(args.ratio) if args.dataset == 'T20H' or args.dataset == 'T15S' else 'label.dat'
    label_test_file = 'label_TaxPayer{}.dat.test'.format(args.ratio) if args.dataset == 'T20H' or args.dataset == 'T15S' else 'label.dat.test'

    # h = "变量1" if a>b else "变量2"
    emb_file = ''
    emb_file = emb_file + 'Att' if args.attributed == 'True' else emb_file + 'UnAtt'

    if args.supervised == 'True':
        emb_file = emb_file + '_Sup_{}'.format(args.ratio) if args.dataset == 'T20H' or args.dataset == 'T15S' else emb_file + '_Sup'
    else:
        emb_file += '_UnSup'
    emb_file += '.emb.dat'

    emb_file_path = f'{model_folder}/{args.model}/data/{args.dataset}/{emb_file}'
    train_para, emb_dict = load(emb_file_path)
    
    print('Start Evaluation!')
    all_tasks, all_scores = [], []

    print(f'Evaluate Classification Performance for Model {args.model} on Dataset {args.dataset}!')

    label_file_path = f'{data_folder}/{args.dataset}/{label_file}'
    label_test_path = f'{data_folder}/{args.dataset}/{label_test_file}'

    scores = classification_evaluate(args.dataset, args.supervised, label_file_path, label_test_path, emb_dict)
    all_tasks.append('nc')
    all_scores.append(scores)
    
    print('Record Results!')
    record(args, all_tasks, train_para, all_scores)
        
    return


if __name__=='__main__':
    main()