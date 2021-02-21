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

# DNN 
from model.dnn import SmoothBCEwLogits
from model.dnn import TrainDataset
from model.dnn import TestDataset
from model.dnn import Model
from model.dnn import FineTuneScheduler

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

def run_training(fold, seed):

    seed_everything(seed)

    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index

    train_df = train[train['kfold'] != fold].reset_index(drop=True).copy()
    valid_df = train[train['kfold'] == fold].reset_index(drop=True).copy()

    x_train, y_train, y_train_all  = train_df[feature_cols], train_df[target_cols].values, train_df[all_target_cols].values
    x_valid, y_valid, y_valid_all =  valid_df[feature_cols], valid_df[target_cols].values, valid_df[all_target_cols].values
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

    #------------ var --------------
    from sklearn.feature_selection import VarianceThreshold
    var_thresh = VarianceThreshold(0.8)
    var_thresh.fit(x_train)
    x_train = x_train.loc[:,var_thresh.variances_ > 0.8]
    x_valid = x_valid.loc[:,var_thresh.variances_ > 0.8]
    x_test  = x_test.loc[:,var_thresh.variances_  > 0.8]

    num_features = x_train.shape[1]

    x_train,x_valid,x_test = x_train.values,x_valid.values,x_test.values

    def train_model(model, tag_name, target_cols_now, fine_tune_scheduler=None):
        if tag_name == 'ALL_TARGETS':
            train_dataset = TrainDataset(x_train, y_train_all)
            valid_dataset = TrainDataset(x_valid, y_valid_all)
        else:
            train_dataset = TrainDataset(x_train, y_train)
            valid_dataset = TrainDataset(x_valid, y_valid)

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY[tag_name])
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                  steps_per_epoch=len(trainloader),
                                                  pct_start=PCT_START,
                                                  div_factor=DIV_FACTOR[tag_name], 
                                                  max_lr=MAX_LR[tag_name],
                                                  epochs=EPOCHS)

        if tag_name == 'ALL_TARGETS':
            loss_tr = SmoothBCEwLogits(smoothing=0.001,pos_weight = pos_weight_all)
        else:
            loss_tr = SmoothBCEwLogits(smoothing=0.001,pos_weight = pos_weight)
        loss_fn = nn.BCEWithLogitsLoss()

        oof = np.zeros((len(train), len(target_cols_now)))
        best_loss = np.inf

        mod_name = f"mod31_{tag_name}_{seed}_{fold}.pth"
        for epoch in range(EPOCHS):
            if fine_tune_scheduler is not None:
                fine_tune_scheduler.step(epoch, model)
            train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, DEVICE)
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
            logging.info(f"SEED: {seed}, FOLD: {fold}, {tag_name}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}")
            if np.isnan(valid_loss):
                break
            if valid_loss < best_loss:
                best_loss = valid_loss
                oof[val_idx] = valid_preds
                torch.save(model.state_dict(), os.path.join(args.model_dir, mod_name))
        return oof

    fine_tune_scheduler = FineTuneScheduler(EPOCHS)

    pretrained_model = Model(num_features, num_all_targets)
    pretrained_model.to(DEVICE)

    # Train on scored + nonscored targets
    train_model(pretrained_model, 'ALL_TARGETS', all_target_cols)

    tag_name = 'ALL_TARGETS'
    mod_name = f"mod31_{tag_name}_{seed}_{fold}.pth"
    # Load the pretrained model with the best loss
    pretrained_model = Model(num_features, num_all_targets)
    pretrained_model.load_state_dict(torch.load(os.path.join(args.model_dir, mod_name)))
    pretrained_model.to(DEVICE)

    # Copy model without the top layer
    final_model = fine_tune_scheduler.copy_without_top(pretrained_model, num_features, num_all_targets, num_targets)

    # Fine-tune the model on scored targets only
    oof = train_model(final_model, 'SCORED_ONLY', target_cols, fine_tune_scheduler)

    tag_name = 'SCORED_ONLY'
    mod_name = f"mod31_{tag_name}_{seed}_{fold}.pth"
    # Load the fine-tuned model with the best loss
    model = Model(num_features, num_targets)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, mod_name)))
    model.to(DEVICE)

    #--------------------- PREDICTION---------------------
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

    predictions = np.zeros((len(test_), num_targets))
    predictions = inference_fn(model, testloader, DEVICE)
    return oof, predictions

def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    for fold in range(NFOLDS):
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
utils.set_logger(os.path.join(args.model_dir, 'train_dnn.log'))

# Load parameters
logging.info("Loading params.json from {}".format(args.model_dir))
json_path = os.path.join(args.model_dir, 'params.json')   
assert os.path.isfile(json_path), "No json file found at {}".format(json_path) 
params = utils.Params(json_path)

seed_everything(seed=42)

# SEED = [200, 201, 202, 203 ,204, 205, 206, 207, 208, 209]
SEED = [0, 1, 2, 3 ,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

NFOLDS = params.num_folds_dnn

# load data 
logging.info("Loading the datasets from {}".format(args.input_dir))  
sc_dic = {}
train_features          = pd.read_csv(os.path.join(args.input_dir, 'train_features.csv'))
train_targets_scored    = pd.read_csv(os.path.join(args.input_dir, 'train_targets_scored.csv'))
train_targets_nonscored = pd.read_csv(os.path.join(args.input_dir, 'train_targets_nonscored.csv'))
test_features           = pd.read_csv(os.path.join(args.input_dir, 'test_features.csv'))
sample_submission       = pd.read_csv(os.path.join(args.input_dir, 'sample_submission.csv'))
train_drug              = pd.read_csv(os.path.join(args.input_dir, 'train_drug.csv'))

target_cols = train_targets_scored.drop('sig_id', axis=1).columns.values.tolist()
aux_target_cols = train_targets_nonscored.drop('sig_id', axis=1).columns.values.tolist()
all_target_cols = target_cols + aux_target_cols

# Dictionary features 
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
logging.info("Drop control samples...")
train = train_features.merge(train_targets_scored, on='sig_id')
train = train.merge(train_targets_nonscored, on='sig_id')
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

target = train[train_targets_scored.columns]

train0 = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)

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
len(feature_cols)
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
    skf = MultilabelStratifiedKFold(n_splits = NFOLDS) # , shuffle = True, random_state = seed
    tmp = target2.groupby('drug_id')[target_cols].mean().loc[vc1]
    tmp_idx = tmp.index.tolist()
    tmp_idx.sort()
    tmp_idx2 = random.sample(tmp_idx,len(tmp_idx))
    tmp = tmp.loc[tmp_idx2]
    for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[target_cols])):
        dd = {k:fold for k in tmp.index[idxV].values}
        dct1.update(dd)

    # STRATIFY DRUGS MORE THAN 18X
    skf = MultilabelStratifiedKFold(n_splits = NFOLDS) # , shuffle = True, random_state = seed
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
    EPOCHS              = params.num_epochs # 24
    BATCH_SIZE          = params.batch_size # 128

    # HyperParameters
    #WEIGHT_DECAY = {'ALL_TARGETS': 1e-5, 'SCORED_ONLY': 3e-6}
    WEIGHT_DECAY = {'ALL_TARGETS': params.weight_decay_all
                  , 'SCORED_ONLY': params.weight_decay_scored}
    MAX_LR      = {'ALL_TARGETS': 1e-2, 'SCORED_ONLY': 3e-3}
    DIV_FACTOR  = {'ALL_TARGETS': 1e3, 'SCORED_ONLY': 1e2}
    PCT_START   = 0.1

    n_comp1 = params.ncompo_genes_dnn  # 600
    n_comp2 = params.ncompo_cells_dnn # 50 

    num_targets = len(target_cols)
    num_aux_targets = len(aux_target_cols)
    num_all_targets = len(all_target_cols)
    hidden_size=4096

    tar_freq = np.array([np.min(list(g_table(train[target_cols].iloc[:,i]).values())) for i in range(len(target_cols))])
    tar_weight0 = np.array([np.log(i+100) for i in tar_freq])
    tar_weight0_min = dp(np.min(tar_weight0))
    tar_weight = tar_weight0_min/tar_weight0
    pos_weight = torch.tensor(tar_weight).to(DEVICE)

    tar_freq = np.array([np.min(list(g_table(train[all_target_cols].iloc[:,i]).values())) for i in range(len(all_target_cols))])
    tar_weight0 = np.array([np.log(i+100) for i in tar_freq])
    tar_weight0_min = dp(np.min(tar_weight0))
    pos_weight_all = tar_weight0_min/tar_weight0
    pos_weight_all = torch.tensor(pos_weight_all).to(DEVICE)

    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)
    
    oof_tmp = dp(oof)
    oof_tmp = oof_tmp * len(SEED) / (SEED.index(seed)+1)
    sc_dic[seed] = np.mean([log_loss(train[target_cols].iloc[:,i],oof_tmp[:,i]) for i in range(len(target_cols))])
    
logging.info(np.mean([log_loss(train[target_cols].iloc[:,i],oof[:,i]) for i in range(len(target_cols))]))

train0[target_cols] = oof
test[target_cols] = predictions

# save predictions and metrics
train0.to_csv(os.path.join(args.model_dir, 'dnn_train.csv'), index=False)
test.to_csv(os.path.join(args.model_dir, 'dnn_test.csv'), index=False)
pd.DataFrame(sc_dic,index=['sc']).to_csv(os.path.join(args.model_dir, 'dnn_sc_dic.csv'))

logging.info("done!")
