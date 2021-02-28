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

# Tabnet 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor

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
        
def run_training(fold, seed):

    seed_everything(seed)

    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index

    train_df = train[train['kfold'] != fold].reset_index(drop=True).copy()
    valid_df = train[train['kfold'] == fold].reset_index(drop=True).copy()

    x_train, y_train  = train_df[feature_cols], train_df[target_cols].values
    x_valid, y_valid =  valid_df[feature_cols], valid_df[target_cols].values
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
    x_test  = x_test.loc[:,var_thresh.variances_ > 0.8]

    x_train,x_valid,x_test = x_train.values,x_valid.values,x_test.values

    class LogitsLogLoss(Metric):
        """
        LogLoss with sigmoid applied
        """
        def __init__(self):
            self._name = "logits_ll"
            self._maximize = False

        def __call__(self, y_true, y_pred):
            """
            Compute LogLoss of predictions.

            Parameters
            ----------
            y_true: np.ndarray
                Target matrix or vector
            y_score: np.ndarray
                Score matrix or vector

            Returns
            -------
                float
                LogLoss of predictions vs targets.
            """
            logits = 1 / (1 + np.exp(-y_pred))
            aux = (1 - y_true) * np.log(1 - logits + 1e-15) + y_true * np.log(logits + 1e-15)
            return np.mean(-aux)

    MAX_EPOCH = 120
    tabnet_params = dict(
        n_d = 64,
        n_a = 128,
        n_steps = 1,
        gamma = 1.3,
        lambda_sparse = 0,
        n_independent = 2,
        n_shared = 1,
        optimizer_fn = optim.Adam,
        optimizer_params = dict(lr = 2e-2, weight_decay = 1e-5),
        mask_type = "entmax",
        scheduler_params = dict(
            mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
        scheduler_fn = ReduceLROnPlateau,
        seed = seed,
        verbose = 10
    )

    ### Fit ###
    model = TabNetRegressor(**tabnet_params)
    model.fit(
        X_train = x_train,
        y_train = y_train,
        eval_set = [(x_valid, y_valid)],
        eval_name = ["val"],
        eval_metric = ["logits_ll"],
        max_epochs = MAX_EPOCH,
        patience = 40,
        batch_size = 1024, 
        virtual_batch_size = 32,
        num_workers = 1,
        drop_last = False,
        loss_fn = SmoothBCEwLogits(smoothing=1e-4) # wgt_bce
    )

    oof = np.zeros((len(train), len(target_cols)))
    valid_preds = 1 / (1 + np.exp(-model.predict(x_valid)))
    oof[val_idx] = valid_preds
    predictions = 1 / (1 + np.exp(-model.predict(x_test)))
    
    mod_name = f"mod21_{seed}_{fold}_.pth"
    mod_path = model.save_model(os.path.join(args.model_dir, mod_name))

    return oof, predictions

def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    for fold in range(NFOLDS):
        logging.info("Fold {} out of {}".format(fold+1,NFOLDS))
        oof_, pred_ = run_training(fold, seed)

        predictions += pred_ / NFOLDS
        oof += oof_

    return oof, predictions

def fe_stats(train, test):
    features_g = GENES
    features_c = CELLS

    feat_raw = train.columns
    for df in train, test:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)

        df['c52_c42'] = df['c-52'] * df['c-42']
        df['c13_c73'] = df['c-13'] * df['c-73']
        df['c26_c13'] = df['c-23'] * df['c-13']
        df['c33_c6'] = df['c-33'] * df['c-6']
        df['c11_c55'] = df['c-11'] * df['c-55']
        df['c38_c63'] = df['c-38'] * df['c-63']
        df['c38_c94'] = df['c-38'] * df['c-94']
        df['c13_c94'] = df['c-13'] * df['c-94']
        df['c4_c52'] = df['c-4'] * df['c-52']
        df['c4_c42'] = df['c-4'] * df['c-42']
        df['c13_c38'] = df['c-13'] * df['c-38']
        df['c55_c2'] = df['c-55'] * df['c-2']
        df['c55_c4'] = df['c-55'] * df['c-4']
        df['c4_c13'] = df['c-4'] * df['c-13']
        df['c82_c42'] = df['c-82'] * df['c-42']
        df['c66_c42'] = df['c-66'] * df['c-42']
        df['c6_c38'] = df['c-6'] * df['c-38']
        df['c2_c13'] = df['c-2'] * df['c-13']
        df['c62_c42'] = df['c-62'] * df['c-42']
        df['c90_c55'] = df['c-90'] * df['c-55']      

    feat_new = train.columns
    feat_stat = list(set(feat_new) - set(feat_raw))
    feat_stat.sort()
    return train, test, feat_stat

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

# -------------------------------------------------------
# MAIN -------------------------------------------------------
# -------------------------------------------------------

if __name__ == '__main__':

  args = Parse_args()

  # Set logger
  utils.set_logger(os.path.join(args.model_dir, 'train_tabnet.log'))

  # Load parameters
  logging.info("Loading params.json from {}".format(args.model_dir))
  json_path = os.path.join(args.model_dir, 'params.json')   
  assert os.path.isfile(json_path), "No json file found at {}".format(json_path) 
  params = utils.Params(json_path)

  seed_everything(seed=42)

  SEEDS = params.num_seeds 

  # load data 
  logging.info("Loading the datasets from {}".format(args.input_dir))  
  sc_dic = {}
  train_features          = pd.read_csv(os.path.join(args.input_dir, 'train_features.csv'))
  train_targets_scored    = pd.read_csv(os.path.join(args.input_dir, 'train_targets_scored.csv'))
  train_targets_nonscored = pd.read_csv(os.path.join(args.input_dir, 'train_targets_nonscored.csv'))
  test_features           = pd.read_csv(os.path.join(args.input_dir, 'test_features.csv'))
  sample_submission       = pd.read_csv(os.path.join(args.input_dir, 'sample_submission.csv'))
  train_drug              = pd.read_csv(os.path.join(args.input_dir, 'train_drug.csv'))

  # features 
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

  logging.info("Add feature stats...")
  train_features, test_features, feat_stat = fe_stats(train_features,test_features)
  feat_dic['stat'] = feat_stat

  # remove ctl
  logging.info("Drop control samples...")
  train = train_features.merge(train_targets_scored, on='sig_id')
  train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
  test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

  target = train[train_targets_scored.columns]

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
  len(feature_cols)
  feature_cols0 = dp(feature_cols)

  oof = np.zeros((len(train), len(target_cols)))
  predictions = np.zeros((len(test), len(target_cols)))

  # Averaging on multiple SEEDS
  for seed in range(SEEDS):
      logging.info("Seed {} out of {}".format(seed+1, SEEDS))
      seed_everything(seed=seed)
      folds = train0.copy()
      feature_cols = dp(feature_cols0)

      # HyperParameters
      DEVICE  = ('cuda' if torch.cuda.is_available() else 'cpu')
      NFOLDS  = params.num_folds_tabnet # 5    
      n_comp1 = params.ncompo_genes_tabnet #600
      n_comp2 = params.ncompo_cells_tabnet #50
    
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
 
      tar_freq = np.array([np.min(list(g_table(train[target_cols].iloc[:,i]).values())) for i in range(len(target_cols))])
      tar_weight0 = np.array([np.log(i+100) for i in tar_freq])
      tar_weight0_min = dp(np.min(tar_weight0))
      tar_weight = tar_weight0_min/tar_weight0
      pos_weight = torch.tensor(tar_weight).to(DEVICE)
    
      wgt_bce = dp(F.binary_cross_entropy_with_logits)
      wgt_bce.__defaults__ = (None, None, None, 'mean', pos_weight)
    
      oof_, predictions_ = run_k_fold(NFOLDS, seed)
      oof += oof_ / SEEDS
      predictions += predictions_ / SEEDS

      oof_tmp = dp(oof)
      oof_tmp = oof_tmp * SEEDS / (seed+1)
      sc_dic[seed] = np.mean([log_loss(train[target_cols].iloc[:,i],oof_tmp[:,i]) for i in range(len(target_cols))])

  logging.info(np.mean([log_loss(train[target_cols].iloc[:,i], oof[:,i]) for i in range(len(target_cols))]))
  
  train0[target_cols] = oof
  test[target_cols] = predictions

  # save predictions and metrics
  train0.to_csv(os.path.join(args.model_dir, 'tabnet_train.csv'), index=False)
  test.to_csv(os.path.join(args.model_dir, 'tabnet_test.csv'), index=False)
  pd.DataFrame(sc_dic,index=['sc']).to_csv(os.path.join(args.model_dir, 'tabnet_sc_dic.csv'))

  logging.info("done!")



