import warnings
import numpy as np
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import random

seed = 1
max_iter = 3000
np.random.seed(seed)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def classification_evaluate(dataset, supervised, label_file_path, label_test_path, emb_dict):    
    train_labels, train_embeddings = [], []
    with open(label_file_path,'r') as label_file:
        for line in label_file:
            index, _, _, label = line[:-1].split('\t')
            train_labels.append(label)
            train_embeddings.append(emb_dict[index])    
    train_labels, train_embeddings = np.array(train_labels).astype(int), np.array(train_embeddings)  
    
    test_labels, test_embeddings = [], []
    with open(label_test_path,'r') as label_file:
        for line in label_file:
            index, _, _, label = line[:-1].split('\t')
            test_labels.append(label)
            test_embeddings.append(emb_dict[index])    
    test_labels, test_embeddings = np.array(test_labels).astype(int), np.array(test_embeddings)  
    
    clf = LinearSVC(random_state=seed, max_iter=max_iter)
    clf.fit(train_embeddings, train_labels)
    preds = clf.predict(test_embeddings)

    if dataset == 'T20H' or dataset == 'T15S':
        f1 = f1_score(test_labels, preds)
        acc = accuracy_score(test_labels, preds)
        pre = precision_score(test_labels, preds)
        rec = recall_score(test_labels, preds)
        auc = roc_auc_score(test_labels, preds)
            
        print(format(f1, '.4f'), format(acc, '.4f'), format(pre, '.4f'), format(rec, '.4f'), format(auc, '.4f'))
        return f1, acc, pre, rec, auc
    else:
        macro = f1_score(test_labels, preds, average='macro')
        micro = f1_score(test_labels, preds, average='micro')
        print(format(macro, '.4f'), format(micro, '.4f'))
        return macro, micro
