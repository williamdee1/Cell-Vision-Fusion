'''
Functions to remove columns and/or rows based on multiple criteria
File source: https://github.com/carpenter-singh-lab/2023_Arevalo_BatchCorrection/blob/main/cleaning.py
'''
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
# from collections.abc import Iterable
# import anndata as ad


logger = logging.getLogger(__name__)


def drop_outlier_feats(dframe: pd.DataFrame, threshold: float):
    '''Remove columns with 1 percentile of absolute values larger than threshold'''
    feat_cols = find_feat_cols(dframe.columns)
    large_feat = dframe[feat_cols].abs().quantile(0.99) > threshold
    large_feat = set(large_feat[large_feat].index)
    keep_cols = [c for c in dframe.columns if c not in large_feat]
    num_ignored = dframe.shape[1] - len(keep_cols)
    logger.info(f'{num_ignored} ignored columns due to large values')
    dframe = dframe[keep_cols]
    return dframe, num_ignored


def drop_outlier_samples(dframe, threshold):
    '''Remove rows with any absolute value larger than threshold'''
    feat_cols = find_feat_cols(dframe.columns)
    large_samples_idx = False
    for col in feat_cols:
        large_samples_idx = (dframe[col].abs() > threshold) | large_samples_idx
    num_ignored = large_samples_idx.sum()
    logger.info(f'{num_ignored} ignored rows due to large values')
    dframe = dframe[~large_samples_idx]
    return dframe


def drop_na_inf(dframe: pd.DataFrame, axis: int = 0):
    '''Drop NaN and Inf values in the features'''
    dframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_cols = find_feat_cols(dframe)
    if axis == 0:
        ignored = False
        for col in feat_cols:
            ignored = dframe[col].isna() | ignored
        num_ignored = ignored.sum()
        if num_ignored == 0:
            return dframe
        dframe = dframe[~ignored]
    else:
        ignored = []
        for col in feat_cols:
            if dframe[col].isna().any():
                ignored.append(col)
        num_ignored = len(ignored)
        if num_ignored == 0:
            return dframe
        dframe = dframe[dframe.columns.difference(ignored)]

    dim = 'rows' if axis == 0 else 'cols'
    logger.info(f'{num_ignored} deleted {dim} with NaN')
    return dframe


def isolation_removal(dframe: pd.DataFrame):
    '''
    Remove outliers with IsolationForest method
    '''
    model = IsolationForest(n_estimators=100, n_jobs=-1, random_state=0)
    feat_cols = find_feat_cols(dframe)
    model.fit(dframe[feat_cols])
    is_ok = model.predict(dframe[feat_cols])
    num_ignored = (is_ok != 1).sum()
    logger.info(f'{num_ignored} deleted rows by isolation')
    dframe = dframe[is_ok == 1]
    return dframe


def find_feat_cols(cols):
    ''' Find column names for features '''
    with open('data/ibp/feature_set.txt', 'r', encoding='utf8') as f_in:
        feature_set = f_in.read().splitlines()
    feat_cols = [c for c in cols if c in feature_set]
    return feat_cols


def find_meta_cols(cols):
    ''' Find column names for metadata '''
    with open('data/ibp/feature_set.txt', 'r', encoding='utf8') as f_in:
        feature_set = f_in.read().splitlines()
    meta_cols = [c for c in cols if c not in feature_set]
    return meta_cols


# def to_anndata(dframe: pd.DataFrame):
#     '''Convert pandas dataframe to AnnData object'''
#     dframe.reset_index(drop=True, inplace=True)
#     meta = dframe[find_meta_cols(dframe)]
#     feat_cols = find_feat_cols(dframe)
#     data = dframe[feat_cols].values.astype(np.float32)
#
#     adata = ad.AnnData(data)
#     adata.var_names = feat_cols
#     adata.obs = meta
#     return adata


def rem_pref(d, prefix):
    """ Remove string prefix from a dictionary key """
    return {k.replace(prefix, ''): v for k, v in d.items()}


def rr_prefix(d, prefix):
    "Recursively removing prefixes for all dictionary levels "

    # Change cv-fold keys:
    d = {int(key[2:]): value for key, value in d.items()}

    if isinstance(d, dict):
        return {k: rem_pref(v, prefix) for k, v in d.items()}
    else:
        return d

