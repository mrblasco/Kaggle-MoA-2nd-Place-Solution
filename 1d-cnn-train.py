#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

# import warnings
from time import time
import datetime, random
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _WeightedLoss

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import os
import copy
from copy import deepcopy as dp
import argparse
import json
import utils
import logging

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight,
                                                  pos_weight = pos_weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class TrainDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
        }
        return dct

class TestDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss

def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data['x'].to(device)
        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds

class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        cha_1 = 256
        cha_2 = 512
        cha_3 = 512

        cha_1_reshape = int(hidden_size/cha_1)
        cha_po_1 = int(hidden_size/cha_1/2)
        cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):

        x = self.batch_norm1(x) if x.size()[0] > 1 else x 
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0],self.cha_1,
                      self.cha_1_reshape)

        x = self.batch_norm_c1(x) if x.size()[0] > 1 else x 
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x) if x.size()[0] > 1 else x 
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x) if x.size()[0] > 1 else x 
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x) if x.size()[0] > 1 else x 
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x =  x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x) if x.size()[0] > 1 else x 
        x = self.dropout3(x)
        x = self.dense3(x)

        return x

def run_training(fold, seed):

    seed_everything(seed)

    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index

    train_df = train[train['kfold'] != fold].reset_index(drop=True).copy()
    valid_df = train[train['kfold'] == fold].reset_index(drop=True).copy()

    x_train, y_train,y_train_ns = train_df[feature_cols], train_df[target_cols].values,train_df[target_nonsc_cols2].values
    x_valid, y_valid,y_valid_ns  =  valid_df[feature_cols], valid_df[target_cols].values,valid_df[target_nonsc_cols2].values
    x_test = test_[feature_cols]

    #------------ norm --------------
    col_num = list(set(feat_dic['gene'] + feat_dic['cell']) & set(feature_cols))
    col_num.sort()
    x_train[col_num],ss = norm_fit(x_train[col_num],True,'quan')
    x_valid[col_num]    = norm_tra(x_valid[col_num],ss)
    x_test[col_num]     = norm_tra(x_test[col_num],ss)

    #------------ pca --------------
    def pca_pre(tr,va,te,
                n_comp,feat_raw,feat_new):
        pca = PCA(n_components=n_comp, random_state=42)
        tr2 = pd.DataFrame(pca.fit_transform(tr[feat_raw]),columns=feat_new)
        va2 = pd.DataFrame(pca.transform(va[feat_raw]),columns=feat_new)
        te2 = pd.DataFrame(pca.transform(te[feat_raw]),columns=feat_new)
        return(tr2,va2,te2)

    pca_feat_g = [f'pca_G-{i}' for i in range(n_comp1)]
    feat_dic['pca_g'] = pca_feat_g
    x_tr_g_pca,x_va_g_pca,x_te_g_pca = pca_pre(x_train,x_valid,x_test,
                                               n_comp1,feat_dic['gene'],pca_feat_g)
    x_train = pd.concat([x_train,x_tr_g_pca],axis = 1)
    x_valid = pd.concat([x_valid,x_va_g_pca],axis = 1)
    x_test  = pd.concat([x_test,x_te_g_pca],axis = 1)

    pca_feat_g = [f'pca_C-{i}' for i in range(n_comp2)]
    feat_dic['pca_c'] = pca_feat_g
    x_tr_c_pca,x_va_c_pca,x_te_c_pca = pca_pre(x_train,x_valid,x_test,
                                               n_comp2,feat_dic['cell'],pca_feat_g)
    x_train = pd.concat([x_train,x_tr_c_pca],axis = 1)
    x_valid = pd.concat([x_valid,x_va_c_pca],axis = 1)
    x_test  = pd.concat([x_test,x_te_c_pca], axis = 1)

    x_train,x_valid,x_test = x_train.values,x_valid.values,x_test.values

    train_dataset = TrainDataset(x_train, y_train_ns)
    valid_dataset = TrainDataset(x_valid, y_valid_ns)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE
                                              , shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE
                                              , shuffle = False)

    model = Model(
        num_features=num_features,
        num_targets=num_targets_0,
        hidden_size=hidden_size,
    )

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5, 
                                              max_lr=0.0001, epochs=EPOCHS, steps_per_epoch=len(trainloader))

    loss_tr = nn.BCEWithLogitsLoss()   #SmoothBCEwLogits(smoothing = 0.001)
    loss_va = nn.BCEWithLogitsLoss()    

    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0

    for epoch in range(1):
        train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, DEVICE)
        valid_loss, valid_preds = valid_fn(model, loss_va, validloader, DEVICE)
        logging.info(f"FOLD: {fold}, EPOCH: {epoch},train_loss: {train_loss}, valid_loss: {valid_loss}")

    model.dense3 = nn.utils.weight_norm(nn.Linear(model.cha_po_2, num_targets))
    model.to(DEVICE)

    train_dataset = TrainDataset(x_train, y_train)
    valid_dataset = TrainDataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

    loss_tr = SmoothBCEwLogits(smoothing = 0.001)
    loss_va = nn.BCEWithLogitsLoss()    

    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0

    oof = np.zeros((len(train), len(target_cols)))
    best_loss = np.inf

    mod_name = f"FOLD_mod11_{seed}_{fold}_.pth"
    
    for epoch in range(EPOCHS):

        train_loss = train_fn(model, optimizer,scheduler, loss_tr, trainloader, DEVICE)
        valid_loss, valid_preds = valid_fn(model, loss_va, validloader, DEVICE)
        logging.info(f"SEED: {seed}, FOLD: {fold}, EPOCH: {epoch},train_loss: {train_loss}, valid_loss: {valid_loss}")

        if valid_loss < best_loss:

            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), os.path.join(args.model_dir, mod_name))

        elif(EARLY_STOP == True):

            early_step += 1
            if (early_step >= early_stopping_steps):
                break

    #--------------------- PREDICTION---------------------
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )

    model.load_state_dict(torch.load(os.path.join(args.model_dir,mod_name)))
    model.to(DEVICE)

    predictions = np.zeros((len(test_), len(target_cols)))
    predictions = inference_fn(model, testloader, DEVICE)
    return oof, predictions

def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    for fold in range(NFOLDS):
        logging.info("Fold {} of {}".format(fold+1, params.num_folds))
        oof_, pred_ = run_training(fold, seed)

        predictions += pred_ / NFOLDS
        oof += oof_

    return oof, predictions

def norm_fit(df_1,saveM = True, sc_name = 'zsco'):   
    from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,Normalizer,QuantileTransformer,PowerTransformer
    ss_1_dic = {'zsco':StandardScaler(),
                'mima':MinMaxScaler(),
                'maxb':MaxAbsScaler(), 
                'robu':RobustScaler(),
                'norm':Normalizer(), 
                'quan':QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal"),
                'powe':PowerTransformer()}
    ss_1 = ss_1_dic[sc_name]
    df_2 = pd.DataFrame(ss_1.fit_transform(df_1),index = df_1.index,columns = df_1.columns)
    if saveM == False:
        return(df_2)
    else:
        return(df_2,ss_1)

def norm_tra(df_1,ss_x):
    df_2 = pd.DataFrame(ss_x.transform(df_1),index = df_1.index,columns = df_1.columns)
    return(df_2)

def g_table(list1):
    table_dic = {}
    for i in list1:
        if i not in table_dic.keys():
            table_dic[i] = 1
        else:
            table_dic[i] += 1
    return(table_dic)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def Parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input_dir', default='./data/from_kaggle'
                    , help='Directory containing dataset')
    args.add_argument('--model_dir', default='./experiments/base_model'
                      , help='Directory containing params.json')
    args = args.parse_args()
    return args

# MAIN -------------------------------------------------------

args = Parse_args()

# Set logger
utils.set_logger(os.path.join(args.model_dir, 'train.log'))

# Load parameters
logging.info("Loading params.json from {}".format(args.model_dir))
json_path = os.path.join(args.model_dir, 'params.json')   
assert os.path.isfile(json_path), "No json file found at {}".format(json_path) 
params = utils.Params(json_path)

seed_everything(seed=42)
SEED = [0, 1, 2, 3 ,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# load data 
logging.info("Loading the datasets from {}".format(args.input_dir))  
sc_dic = {}
train_features          = pd.read_csv(os.path.join(args.input_dir, 'train_features.csv'))
train_targets_scored    = pd.read_csv(os.path.join(args.input_dir, 'train_targets_scored.csv'))
train_targets_nonscored = pd.read_csv(os.path.join(args.input_dir, 'train_targets_nonscored.csv'))
test_features           = pd.read_csv(os.path.join(args.input_dir, 'test_features_calibr.csv'))
#sample_submission       = pd.read_csv(os.path.join(args.input_dir, 'sample_submission.csv'))
train_drug              = pd.read_csv(os.path.join(args.input_dir, 'train_drug.csv'))

# Target names 
target_cols = train_targets_scored.drop('sig_id', axis=1).columns.values.tolist()
target_nonsc_cols = train_targets_nonscored.drop('sig_id', axis=1).columns.values.tolist()

######## non-score ########
nonctr_id = train_features.loc[train_features['cp_type']!='ctl_vehicle','sig_id'].tolist()
tmp_con1 = [i in nonctr_id for i in train_targets_scored['sig_id']]
mat_cor = pd.DataFrame(np.corrcoef(train_targets_scored.drop('sig_id',axis = 1)[tmp_con1].T,
                      train_targets_nonscored.drop('sig_id',axis = 1)[tmp_con1].T))
mat_cor2 = mat_cor.iloc[(train_targets_scored.shape[1]-1):,0:train_targets_scored.shape[1]-1]
mat_cor2.index = target_nonsc_cols
mat_cor2.columns = target_cols
mat_cor2 = mat_cor2.dropna()
mat_cor2_max = mat_cor2.abs().max(axis = 1)

q_n_cut = 0.9
target_nonsc_cols2 = mat_cor2_max[mat_cor2_max > np.quantile(mat_cor2_max,q_n_cut)].index.tolist()
logging.info("Keep {} selected non-scored targets".format(len(target_nonsc_cols2)))

# Dictionary for features 
feat_dic = {}
GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
feat_dic['gene'] = GENES
feat_dic['cell'] = CELLS

# sample norm 
logging.info("Quantile normalization...")
q2 = train_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = train_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.75).copy()
qmean = (q2+q7)/2
train_features[feat_dic['gene']] = (train_features[feat_dic['gene']].T - qmean.values).T
q2 = test_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = test_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.75).copy()
qmean = (q2+q7)/2
test_features[feat_dic['gene']] = (test_features[feat_dic['gene']].T - qmean.values).T

q2 = train_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = train_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.72).copy()
qmean = (q2+q7)/2
train_features[feat_dic['cell']] = (train_features[feat_dic['cell']].T - qmean.values).T
qmean2 = train_features[feat_dic['cell']].abs().apply(np.quantile,axis = 1,q = 0.75).copy()+4
train_features[feat_dic['cell']] = (train_features[feat_dic['cell']].T / qmean2.values).T.copy()

q2 = test_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = test_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.72).copy()
qmean = (q2+q7)/2
test_features[feat_dic['cell']] = (test_features[feat_dic['cell']].T - qmean.values).T
qmean2 = test_features[feat_dic['cell']].abs().apply(np.quantile,axis = 1,q = 0.75).copy()+4
test_features[feat_dic['cell']] = (test_features[feat_dic['cell']].T / qmean2.values).T.copy()

# remove ctl
logging.info("Remove controls...")
train = train_features.merge(train_targets_scored, on='sig_id')
train = train.merge(train_targets_nonscored[['sig_id']+target_nonsc_cols2], on='sig_id')

train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

target = train[['sig_id']+target_cols]
target_ns = train[['sig_id']+target_nonsc_cols2]

train0 = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)

target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

# drug ids
tar_sig = target['sig_id'].tolist()
train_drug = train_drug.loc[[i in tar_sig for i in train_drug['sig_id']]]
target = target.merge(train_drug, on='sig_id', how='left') 

# LOCATE DRUGS
vc = train_drug.drug_id.value_counts()
vc1 = vc.loc[vc <= 19].index
vc2 = vc.loc[vc > 19].index

feature_cols = []
for key_i in feat_dic.keys():
    value_i = feat_dic[key_i]
    print(key_i,len(value_i))
    feature_cols += value_i

logging.info("Final num. of features {}".format(len(feature_cols)))
feature_cols0 = dp(feature_cols)
    
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

# Averaging on multiple SEEDS
for seed in SEED:
    logging.info("Seed {} out of {}".format(SEED.index(seed)+1, len(SEED)))

    seed_everything(seed=seed)
    folds = train0.copy()
    feature_cols = dp(feature_cols0)
    
    # kfold - leave drug out
    target2 = target.copy()
    dct1 = {}; dct2 = {}
    skf = MultilabelStratifiedKFold(n_splits = 5) # , shuffle = True, random_state = seed
    tmp = target2.groupby('drug_id')[target_cols].mean().loc[vc1]
    tmp_idx = tmp.index.tolist()
    tmp_idx.sort()
    tmp_idx2 = random.sample(tmp_idx,len(tmp_idx))
    tmp = tmp.loc[tmp_idx2]
    for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[target_cols])):
        dd = {k:fold for k in tmp.index[idxV].values}
        dct1.update(dd)

    # STRATIFY DRUGS MORE THAN 19X
    skf = MultilabelStratifiedKFold(n_splits = 5) # , shuffle = True, random_state = seed
    tmp = target2.loc[target2.drug_id.isin(vc2)].reset_index(drop = True)
    tmp_idx = tmp.index.tolist()
    tmp_idx.sort()
    tmp_idx2 = random.sample(tmp_idx,len(tmp_idx))
    tmp = tmp.loc[tmp_idx2]
    for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[target_cols])):
        dd = {k:fold for k in tmp.sig_id[idxV].values}
        dct2.update(dd)

    target2['kfold'] = target2.drug_id.map(dct1)
    target2.loc[target2.kfold.isna(),'kfold'] = target2.loc[target2.kfold.isna(),'sig_id'].map(dct2)
    target2.kfold = target2.kfold.astype(int)

    folds['kfold'] = target2['kfold'].copy()

    train = folds.copy()
    test_ = test.copy()

    # HyperParameters
    DEVICE              = ('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS              = params.num_epochs
    BATCH_SIZE          = params.batch_size
    LEARNING_RATE       = params.learning_rate
    WEIGHT_DECAY        = params.weight_decay
    NFOLDS              = params.num_folds
    EARLY_STOPPING_STEPS = params.early_stopping_steps
    EARLY_STOP          = params.early_stop == "True"
    n_comp1             = params.ncompo_genes # 50
    n_comp2             = params.ncompo_genes # 15

    num_features=len(feature_cols) + n_comp1 + n_comp2
    num_targets=len(target_cols)
    num_targets_0=len(target_nonsc_cols2)
    hidden_size=4096

    tar_freq = np.array([np.min(list(g_table(train[target_cols].iloc[:,i]).values())) for i in range(len(target_cols))])
    tar_weight0 = np.array([np.log(i+100) for i in tar_freq])
    tar_weight0_min = dp(np.min(tar_weight0))
    tar_weight = tar_weight0_min/tar_weight0
    pos_weight = torch.tensor(tar_weight).to(DEVICE)

    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)
    
    oof_tmp = dp(oof)
    oof_tmp = oof_tmp * len(SEED) / (SEED.index(seed)+1)
    sc_dic[seed] = np.mean([log_loss(train[target_cols].iloc[:,i],oof_tmp[:,i]) for i in range(len(target_cols))])


logging.info(np.mean([log_loss(train[target_cols].iloc[:,i],oof[:,i]) for i in range(len(target_cols))]))

train0[target_cols] = oof
test[target_cols] = predictions

### for blend test ###
train0.to_csv(os.path.join(args.model_dir, 'train_pred.csv'), index=False)
### for blend test ###

# sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
test.to_csv(os.path.join(args.model_dir, 'submission.csv'), index=False)

logging.info(pd.DataFrame(sc_dic,index=['sc']).T)
logging.info("done!")


