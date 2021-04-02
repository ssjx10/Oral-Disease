import os
import random
import numpy as np
import torch
import cv2
import pandas as pd

def t():
    print('test!')

def train_val_split(datadir, train_ratio=0.8):
    data = os.listdir(datadir)

    label_0=[]
    label_1=[]
    for i in range(len(data)):
        label = data[i].split('_')[-1].split('.')[0]
        if label == 'N' or label == 'B':
                label_0.append(data[i])
        elif label == 'C':
                label_1.append(data[i])
    
    random.shuffle(label_0)
    random.shuffle(label_1)
    
    len_train_0 = int(len(label_0)*train_ratio)
    #print(len_train_0)
    train_0 ,val_0 = label_0[0:len_train_0], label_0[len_train_0:]
    print('label_0 : ', len(label_0), 'train_0 : ',len(train_0), 'val_0 : ', len(val_0))

    len_train_1 = int(len(label_1)*train_ratio)
    train_1, val_1 = label_1[0:len_train_1], label_1[len_train_1:]
    print('label_1 : ', len(label_1), 'train_1 : ',len(train_1), 'val_1 : ', len(val_1))

    train_data = train_0 + train_1
    val_data = val_0 + val_1

    random.shuffle(train_data)
    random.shuffle(val_data)
    
    print('len_train : ', len(train_data), 'len_val : ', len(val_data))
    return train_data, val_data

    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    #print(random.random())
    if torch.cuda.is_available():
        print(f'seed : {seed_value}')
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def eval_metrics (pred, label):
    metrics = get_metrics (pred, label)
        
    return np.round(metrics['f1'], 5)

def _confusion_matrix(pred, label, positive_class=1):  
    '''
    (pred, label)
    TN (not p_class, not p_class) / FN (not p_class, p_class) / FP (p_class, not p_class) / TP (p_class, p_class)
    ex)
    TN (0,0) / FN (0,1)/ FP (1,0) / TP (1,1)
    '''
    TN, FN, FP, TP = 0, 0, 0, 0

    for y_hat, y in zip(pred, label):
        if y_hat != positive_class:
            if y != positive_class:
                    TN = TN + 1
            else:
                    FN = FN + 1
        elif y_hat == positive_class:
            if y != positive_class:
                FP = FP + 1
            else:
                TP = TP + 1
    return TN, FN, FP, TP

def get_metrics (pred, label, eps=1e-5):
    '''
    label : 0,1
    pred : sigmoid
    '''
    #pred = pred > 0.5
    metrics = dict()

    num_P, num_N = np.sum(label==1), np.sum(label==0)
    TN, FN, FP, TP = _confusion_matrix(pred, label)
    metrics['prec'] = TP / (TP + FP + eps) ## ppv
    metrics['recall'] = TP / (TP + FN + eps) ## sensitivive
    metrics['f1'] = 2*(metrics['prec'] * metrics['recall'])/(metrics['prec'] + metrics['recall'] + eps)

    return metrics

def auc_score(prob, label, eps=1e-5):
    data = pd.DataFrame({'label' : label, 'prob' : prob})
    data = data.sort_values('prob', ascending=False)
    FPR = []
    TPR = []
    P = len(data[data['label'] == 1])
    N = len(data[data['label'] == 0])
    for i in data['prob']:
        tmp_p = data[data['prob'] >= i]
        TP = len(tmp_p[tmp_p['label'] == 1])
        tmp_TPR = TP/(P+eps)
        tmp_n = data[data['prob'] >= i]
        FP = len(tmp_n[tmp_n['label'] == 0])
        tmp_FPR = FP/(N+eps)
        TPR.append(tmp_TPR)
        FPR.append(tmp_FPR)
    
    AUC_TPR = [0] + TPR
    AUC_FPR = [0] + FPR

    AUC = 0
    for i in range(1, len(AUC_TPR)):
        tmp_AUC = (AUC_TPR[i - 1] + AUC_TPR[i]) * (AUC_FPR[i] - AUC_FPR[i - 1]) / 2
        AUC += tmp_AUC
    return AUC
