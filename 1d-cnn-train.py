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
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import os
import copy
from copy import deepcopy as dp
import argparse
import json
import utils
import logging

# CNN
import model.cnn 
from model.cnn import SmoothBCEwLogits
from model.cnn import TrainDataset
from model.cnn import TestDataset
from model.cnn import Model

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
    """Train the model for one fold, seed.

    Steps for training are: 

    Args:
        fold: (numeric) fold index for `kfold` 
        seed: (numeric) seed
        train: (df) dataset from training 
        test_: (df) dataset from testing
        feat_dic: (dict) genes and cells columns
        params: (Params) hyper-params for the model 
        num_features, num_targets_0: (number) length-type vars
        pos_weight: (weights) 
    """    
    seed_everything(seed)
 
    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index

    train_df = train[train['kfold'] != fold].reset_index(drop = True).copy()
    valid_df = train[train['kfold'] == fold].reset_index(drop = True).copy()

    x_train, y_train, y_train_ns = train_df[feature_cols], train_df[target_cols].values, train_df[target_nonsc_cols2].values
    x_valid, y_valid, y_valid_ns  =  valid_df[feature_cols], valid_df[target_cols].values, valid_df[target_nonsc_cols2].values
    x_test = test_[feature_cols]

    #------------ norm --------------
    col_num = list(set(feat_dic['gene'] + feat_dic['cell']) & set(feature_cols))
    col_num.sort()
    x_train[col_num], ss = norm_fit(x_train[col_num], True, 'quan')
    x_valid[col_num]    = norm_tra(x_valid[col_num], ss)
    x_test[col_num]     = norm_tra(x_test[col_num], ss)

    #------------ pca 1 --------------
    pca_feat_g = [f'pca_G-{i}' for i in range(params.ncompo_genes)]
    feat_dic['pca_g'] = pca_feat_g
    x_tr_g_pca, x_va_g_pca, x_te_g_pca = pca_pre(x_train, x_valid, x_test, 
                                               params.ncompo_genes, feat_dic['gene'], pca_feat_g)
    x_train = pd.concat([x_train, x_tr_g_pca], axis = 1)
    x_valid = pd.concat([x_valid, x_va_g_pca], axis = 1)
    x_test  = pd.concat([x_test, x_te_g_pca], axis = 1)
    
    #------------ pca 2 --------------
    pca_feat_g = [f'pca_C-{i}' for i in range(params.ncompo_cells)]
    feat_dic['pca_c'] = pca_feat_g
    x_tr_c_pca, x_va_c_pca, x_te_c_pca = pca_pre(x_train, x_valid, x_test, 
                                               params.ncompo_cells, feat_dic['cell'], pca_feat_g)
    x_train = pd.concat([x_train, x_tr_c_pca], axis = 1)
    x_valid = pd.concat([x_valid, x_va_c_pca], axis = 1)
    x_test  = pd.concat([x_test, x_te_c_pca], axis = 1)

    x_train, x_valid, x_test = x_train.values, x_valid.values, x_test.values

    # Train model 
    params.num_targets = num_targets 
    model = Model(params)    
    model.to(DEVICE)
    
    # Pre-train w/ non-scored targets -------------------------------------------------------
    if (1): 
      train_dataset = TrainDataset(x_train, y_train_ns)
      valid_dataset = TrainDataset(x_valid, y_valid_ns)
      trainloader = DataLoader(train_dataset, batch_size = params.batch_size, shuffle = True)
      validloader = DataLoader(valid_dataset, batch_size = params.batch_size, shuffle = False)
      
      # Change num targets to non-scored targets!
      params.num_targets = num_targets_0
      model = Model(params)    
      model.to(DEVICE)
      optimizer = torch.optim.Adam(model.parameters()
                                  , lr = params.learning_rate
                                  , weight_decay = params.weight_decay
                                  )
      scheduler = optim.lr_scheduler.OneCycleLR(optimizer = optimizer
                                                , pct_start = 0.1
                                                , div_factor = 1e5
                                                , max_lr = 0.0001
                                                , epochs = params.num_epochs
                                                , steps_per_epoch = len(trainloader)
                                                )
      loss_tr = nn.BCEWithLogitsLoss()   #SmoothBCEwLogits(smoothing = params.smoothing)
      loss_va = nn.BCEWithLogitsLoss()    
      for epoch in range(1):
          train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, DEVICE)
          valid_loss, valid_preds = valid_fn(model, loss_va, validloader, DEVICE)
          logging.info(f"NS/FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")
          
      # Predict scored targets 
      params.num_targets = num_targets 
      model.dense3 = nn.utils.weight_norm(nn.Linear(model.cha_po_2, params.num_targets))
      model.to(DEVICE)

    # Scored targets 
    train_dataset = TrainDataset(x_train, y_train)
    valid_dataset = TrainDataset(x_valid, y_valid)
    trainloader = DataLoader(train_dataset, batch_size = params.batch_size, shuffle = True)
    validloader = DataLoader(valid_dataset, batch_size = params.batch_size, shuffle = False)

    optimizer = torch.optim.Adam(model.parameters()
                              , lr = params.learning_rate
                              , weight_decay = params.weight_decay
                              )
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer = optimizer
                                            , pct_start = 0.1   #  default = 0.3
                                            , div_factor = 1e3  # default = 25
                                            , max_lr = 1e-2
                                            , epochs = params.num_epochs
                                            , steps_per_epoch = len(trainloader)
                                            )

    loss_tr = SmoothBCEwLogits(smoothing = params.smoothing, pos_weight = pos_weight)
    loss_va = nn.BCEWithLogitsLoss()
    oof = np.zeros((len(train), len(target_cols)))
    best_loss = np.inf

    mod_name = f"FOLD_mod11_{seed}_{fold}_.pth"
    early_step = 0    
    for epoch in range(params.num_epochs):

        train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, DEVICE)
        valid_loss, valid_preds = valid_fn(model, loss_va, validloader, DEVICE)
        #logging.info(f"SEED: {seed}, FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")
        logging.info(f" {seed}, {fold}, {epoch}, {train_loss}, {valid_loss}")
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), os.path.join(args.model_dir, mod_name))

        else:
            early_step += 1
            if (early_step >= params.early_stopping_steps):
                break

    #--------------------- PREDICTION---------------------
    testdataset = TestDataset(x_test)
    testloader = DataLoader(testdataset, batch_size = params.batch_size, shuffle = False)

    model = Model(params)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, mod_name)))
    model.to(DEVICE)

    predictions = np.zeros((len(test_), len(target_cols)))
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

def norm_fit(df_1, saveM = True, sc_name = 'zsco'):   
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
    ss_1_dic = {'zsco':StandardScaler(), 
                'mima':MinMaxScaler(), 
                'maxb':MaxAbsScaler(), 
                'robu':RobustScaler(), 
                'norm':Normalizer(), 
                'quan':QuantileTransformer(n_quantiles = 100, random_state = 0, output_distribution = "normal"), 
                'powe':PowerTransformer()}
    ss_1 = ss_1_dic[sc_name]
    df_2 = pd.DataFrame(ss_1.fit_transform(df_1), index = df_1.index, columns = df_1.columns)
    if saveM == False:
        return(df_2)
    else:
        return(df_2, ss_1)

def norm_tra(df_1, ss_x):
    df_2 = pd.DataFrame(ss_x.transform(df_1), index = df_1.index, columns = df_1.columns)
    return(df_2)

def qnorm(train_f, test_f):
  """
  Quantile normalization 
  """
  # train = gene
  q2 = train_f[feat_dic['gene']].apply(np.quantile, axis = 1, q = 0.25).copy()
  q7 = train_f[feat_dic['gene']].apply(np.quantile, axis = 1, q = 0.75).copy()
  qmean = (q2+q7)/2
  train_f[feat_dic['gene']] = (train_f[feat_dic['gene']].T - qmean.values).T
  
  # test = gene
  q2 = test_f[feat_dic['gene']].apply(np.quantile, axis = 1, q = 0.25).copy()
  q7 = test_f[feat_dic['gene']].apply(np.quantile, axis = 1, q = 0.75).copy()
  qmean = (q2+q7)/2
  test_f[feat_dic['gene']] = (test_f[feat_dic['gene']].T - qmean.values).T

  # train = cell 
  q2 = train_f[feat_dic['cell']].apply(np.quantile, axis = 1, q = 0.25).copy()
  q7 = train_f[feat_dic['cell']].apply(np.quantile, axis = 1, q = 0.72).copy()
  qmean = (q2+q7)/2
  train_f[feat_dic['cell']] = (train_f[feat_dic['cell']].T - qmean.values).T
  qmean2 = train_f[feat_dic['cell']].abs().apply(np.quantile, axis = 1, q = 0.75).copy()+4
  train_f[feat_dic['cell']] = (train_f[feat_dic['cell']].T / qmean2.values).T.copy()

  # test = cell 
  q2 = test_f[feat_dic['cell']].apply(np.quantile, axis = 1, q = 0.25).copy()
  q7 = test_f[feat_dic['cell']].apply(np.quantile, axis = 1, q = 0.72).copy()
  qmean = (q2+q7)/2
  test_f[feat_dic['cell']] = (test_f[feat_dic['cell']].T - qmean.values).T
  qmean2 = test_f[feat_dic['cell']].abs().apply(np.quantile, axis = 1, q = 0.75).copy()+4
  test_f[feat_dic['cell']] = (test_f[feat_dic['cell']].T / qmean2.values).T.copy()

  return train_f, test_f

def g_table(list1):
    table_dic = {}
    for i in list1:
        if i not in table_dic.keys():
            table_dic[i] = 1
        else:
            table_dic[i] += 1
    return(table_dic)

def seed_everything(seed = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def pca_pre(tr, va, te, n_comp, feat_raw, feat_new):
    pca = PCA(n_components = n_comp, random_state = 42)
    tr2 = pd.DataFrame(pca.fit_transform(tr[feat_raw]), columns = feat_new)
    va2 = pd.DataFrame(pca.transform(va[feat_raw]), columns = feat_new)
    te2 = pd.DataFrame(pca.transform(te[feat_raw]), columns = feat_new)
    return(tr2, va2, te2)

def select_ns_targets (q_n_cut = 0.9): 
  nonctr_id = train_features.loc[train_features['cp_type']!= 'ctl_vehicle', 'sig_id'].tolist()
  tmp_con1 = [i in nonctr_id for i in train_targets_scored['sig_id']]
  mat_cor = pd.DataFrame(np.corrcoef(train_targets_scored.drop('sig_id', axis = 1)[tmp_con1].T
                        , train_targets_nonscored.drop('sig_id', axis = 1)[tmp_con1].T))
  mat_cor2 = mat_cor.iloc[(train_targets_scored.shape[1] - 1):, 0:train_targets_scored.shape[1]-1]
  mat_cor2.index = target_nonsc_cols
  mat_cor2.columns = target_cols
  mat_cor2 = mat_cor2.dropna()
  mat_cor2_max = mat_cor2.abs().max(axis = 1)
  out = mat_cor2_max[mat_cor2_max > np.quantile(mat_cor2_max, q_n_cut)].index.tolist()
  return out


args = argparse.ArgumentParser()
args.add_argument('--input_dir', default = './data/from_kaggle', help = 'Directory containing dataset')
args.add_argument('--model_dir', default = './experiments/base_model', help = 'Directory containing params.json')

# MAIN -------------------------------------------------------
args = args.parse_args()

# Set logger
utils.set_logger(os.path.join(args.model_dir, 'train.log'))

# Load parameters
logging.info("Loading params.json from {}".format(args.model_dir))
json_path = os.path.join(args.model_dir, 'params.json')   
assert os.path.isfile(json_path), "No json file found at {}".format(json_path) 
params = utils.Params(json_path)

seed_everything(seed = 42)
SEED = range(params.num_seeds)

# Scoring dictionary
sc_dic = {}

logging.info("Loading datasets from {}".format(args.input_dir))  

# load data 
train_features          = pd.read_csv(os.path.join(args.input_dir, 'train_features.csv'))
train_targets_scored    = pd.read_csv(os.path.join(args.input_dir, 'train_targets_scored.csv'))
train_targets_nonscored = pd.read_csv(os.path.join(args.input_dir, 'train_targets_nonscored.csv'))
test_features           = pd.read_csv(os.path.join(args.input_dir, 'test_features.csv'))
#sample_submission       = pd.read_csv(os.path.join(args.input_dir, 'sample_submission.csv'))
train_drug              = pd.read_csv(os.path.join(args.input_dir, 'train_drug.csv'))

# Target names 
target_cols = train_targets_scored.drop('sig_id', axis = 1).columns.values.tolist()
target_nonsc_cols = train_targets_nonscored.drop('sig_id', axis = 1).columns.values.tolist()

# Select non-scored targets
target_nonsc_cols2 = select_ns_targets()
logging.info("Keep {} selected non-scored targets".format(len(target_nonsc_cols2)))

# Dictionary for features 
GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
feat_dic = {}
feat_dic['gene'] = GENES
feat_dic['cell'] = CELLS

# Quantile normalization 
train_features, test_features = qnorm(train_features, test_features)

# Remove control signatures
train = train_features.merge(train_targets_scored, on = 'sig_id')
train = train.merge(train_targets_nonscored[['sig_id']+target_nonsc_cols2], on = 'sig_id')
train = train[train['cp_type']!= 'ctl_vehicle'].reset_index(drop = True)
test = test_features[test_features['cp_type']!= 'ctl_vehicle'].reset_index(drop = True)

target = train[['sig_id']+target_cols]
target_ns = train[['sig_id']+target_nonsc_cols2]

train0 = train.drop('cp_type', axis = 1)
test = test.drop('cp_type', axis = 1)

target_cols = target.drop('sig_id', axis = 1).columns.values.tolist()

# drug ids
tar_sig = target['sig_id'].tolist()
train_drug = train_drug.loc[[i in tar_sig for i in train_drug['sig_id']]]
target = target.merge(train_drug, on = 'sig_id', how = 'left') 

# LOCATE DRUGS
vc = train_drug.drug_id.value_counts()
vc1 = vc.loc[vc <= 19].index
vc2 = vc.loc[vc > 19].index

feature_cols = []
for key_i in feat_dic.keys():
    value_i = feat_dic[key_i]
    print(key_i, len(value_i))
    feature_cols += value_i

feature_cols0 = dp(feature_cols)
    
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

# Averaging on multiple SEEDS
for seed in range(params.num_seeds):

    seed_everything(seed = seed)
    folds = train0.copy()
    feature_cols = dp(feature_cols0)
    
    # kfold - leave drug out
    target2 = target.copy()
    dct1 = {}; dct2 = {}
    skf = MultilabelStratifiedKFold(n_splits = params.num_folds)
    tmp = target2.groupby('drug_id')[target_cols].mean().loc[vc1]
    tmp_idx = tmp.index.tolist()
    tmp_idx.sort()
    tmp_idx2 = random.sample(tmp_idx, len(tmp_idx))
    tmp = tmp.loc[tmp_idx2]
    for fold, (idxT, idxV) in enumerate(skf.split(tmp, tmp[target_cols])):
        dd = {k:fold for k in tmp.index[idxV].values}
        dct1.update(dd)

    # STRATIFY DRUGS MORE THAN 19X
    skf = MultilabelStratifiedKFold(n_splits = params.num_folds) 
    tmp = target2.loc[target2.drug_id.isin(vc2)].reset_index(drop = True)
    tmp_idx = tmp.index.tolist()
    tmp_idx.sort()
    tmp_idx2 = random.sample(tmp_idx, len(tmp_idx))
    tmp = tmp.loc[tmp_idx2]
    for fold, (idxT, idxV) in enumerate(skf.split(tmp, tmp[target_cols])):
        dd = {k:fold for k in tmp.sig_id[idxV].values}
        dct2.update(dd)

    target2['kfold'] = target2.drug_id.map(dct1)
    target2.loc[target2.kfold.isna(), 'kfold'] = target2.loc[target2.kfold.isna(), 'sig_id'].map(dct2)
    target2.kfold = target2.kfold.astype(int)

    folds['kfold'] = target2['kfold'].copy()

    train = folds.copy()
    test_ = test.copy()

    # Set parameters 
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    params.num_features  = len(feature_cols) + params.ncompo_genes + params.ncompo_cells
    num_targets   = len(target_cols)
    num_targets_0 = len(target_nonsc_cols2)
    
    # Weights 
    tar_freq = np.array([np.min(list(g_table(train[target_cols].iloc[:, i]).values())) 
                        for i in range(len(target_cols))])
    tar_weight0 = np.array([np.log(i+100) for i in tar_freq])
    tar_weight0_min = dp(np.min(tar_weight0))
    tar_weight = tar_weight0_min/tar_weight0
    pos_weight = torch.tensor(tar_weight).to(DEVICE)

    # Run k-fold 
    oof_, predictions_ = run_k_fold(params.num_folds, seed)
    oof += oof_ / params.num_seeds
    predictions += predictions_ / params.num_seeds
    
    # Compute metrics 
    oof_tmp = dp(oof)
    oof_tmp = oof_tmp * params.num_seeds / (seed+1)
    sc_dic[seed] = np.mean([log_loss(train[target_cols].iloc[:, i], oof_tmp[:, i]) 
                          for i in range(len(target_cols))])

# Return metrics
sc_dic['final'] = np.mean([log_loss(train[target_cols].iloc[:, i], oof[:, i]) 
                          for i in range(len(target_cols))])
logging.info(sc_dic['final'])

train0[target_cols] = oof
test[target_cols] = predictions

# save predictions and metrics
train0.to_csv(os.path.join(args.model_dir, 'cnn_train.csv'), index = False)
test.to_csv(os.path.join(args.model_dir, 'cnn_test.csv'), index = False)
pd.DataFrame(sc_dic, index = ['sc']).to_csv(os.path.join(args.model_dir, 'cnn_sc_dic.csv'))

logging.info("done!")


