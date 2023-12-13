import logging
import pickle
import numpy as np
import pandas as pd
from modules.utils import get_dataset_files, return_fnames_targets, norm_data
from modules.cleaning import rr_prefix


def load_data(model_type, cv_data, top_feats, args):
    """
    Loads the data for model training, dependent on whether you're training CVF fusion or SwinV2 model arch.
    A fusion model requires the image, IBP and compound structural data. Swinv2 just uses the image data.
    model_type: ['swin', 'enet', 'fusion']
    cv_data: cross-validation data splits
    top_feats: Number of top-ranked CellProfiler features to select (if any)
    args: model arguments
    """

    # Return image filenames for train, val and test datasets:
    logging.info("[------ Returning image filenames -------]")
    train_df = cv_data[args.cv_fold]['train_meta']
    train_fnames = get_dataset_files(train_df)
    val_df = cv_data[args.cv_fold]['val_meta']
    val_fnames = get_dataset_files(val_df)
    test_df = cv_data[args.cv_fold]['test_meta']
    test_fnames = get_dataset_files(test_df)

    if (model_type == 'swin') or (model_type == 'enet'):
        # Load image filenames and target values for each datapoint:
        X_train, y_train, _, _, _, _ = return_fnames_targets(args.img_loc, train_fnames,
                                                             cv_data[args.cv_fold]['y_train'],
                                                             train_df.Metadata_InChIKey.to_list())
        X_val, y_val, _, _, _, _ = return_fnames_targets(args.img_loc, val_fnames, cv_data[args.cv_fold]['y_val'],
                                                         val_df.Metadata_InChIKey.to_list())

        # Load test data:
        X_test, y_test, test_meta, _, _, _ = return_fnames_targets(args.img_loc, test_fnames,
                                                                   cv_data[args.cv_fold]['y_test'],
                                                                   test_df.Metadata_InChIKey.to_list())

        if args.full_train:
            # Combine training and validation data:
            X_train = X_train + X_val
            y_train = np.concatenate((y_train, y_val))
            return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'test_meta': test_meta}
        else:
            return {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val, 'X_test': X_test,
                    'y_test': y_test, 'test_meta': test_meta}

    elif model_type == 'fusion':
        # Load the three different data modalities:
        # -----------------------
        #  Norm. and Standardize
        # -----------------------
        # Choices are based on various norm/spherization methods applied:
        if args.tab_norm == 'minmax':
            # Load dataset csv files and perform feature selection:
            tab_X_train = cv_data[args.cv_fold]['X_train'].astype(np.float32)[top_feats].reset_index(drop=True)
            tab_y_train = cv_data[args.cv_fold]['y_train'].astype(np.float32)
            tab_X_val = cv_data[args.cv_fold]['X_val'].astype(np.float32)[top_feats].reset_index(drop=True)
            tab_y_val = cv_data[args.cv_fold]['y_val'].astype(np.float32)
            # Concat train and validation data:
            tab_X_train = pd.concat([tab_X_train, tab_X_val], axis=0).reset_index(drop=True)
            tab_y_train = np.concatenate((tab_y_train, tab_y_val))
            tab_X_test = cv_data[args.cv_fold]['X_test'].astype(np.float32)[top_feats].reset_index(drop=True)
            tab_y_test = cv_data[args.cv_fold]['y_test'].astype(np.float32)

            # Return metadata:
            tab_train_meta = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
            tab_test_meta = test_df.reset_index(drop=True)

            # Normalize tabular IBP data:
            tab_X_train, tab_X_test = norm_data('minmax', tab_X_train, tab_X_test)

            # Combined metadata:
            train_fnames = train_fnames + val_fnames
            train_cpnds = train_df.Metadata_InChIKey.to_list() + val_df.Metadata_InChIKey.to_list()
            test_cpnds = test_df.Metadata_InChIKey.to_list()

        else:
            if args.tab_norm == 'mads_shap':
                tn_loc = 'data/stnd/Corrected/Sph_Shap_Cor.pkl'          # MAD_Sphere/MADS_ShapFS.pkl'
                prefix_to_remove = 'sph_'
            elif args.tab_norm == 'mads_pycy':
                tn_loc = 'data/stnd/MAD_Sphere/MADS_PyCyFS.pkl'
                prefix_to_remove = 'sph_'
            elif args.tab_norm == 'madh_shap':
                tn_loc = 'data/stnd/MAD_Harmony/MAD_Harmony_ShapFS.pkl'
                prefix_to_remove = 'har_'
            elif args.tab_norm == 'madh_pycy':
                tn_loc = 'data/stnd/MAD_Harmony/MAD_Harmony_PyCyFS.pkl'
                prefix_to_remove = 'har_'
            else:
                prefix_to_remove = '_'
                tn_loc = None

            with open(tn_loc, 'rb') as file:
                tab_cv_data = pickle.load(file)
            tab_cv_data = rr_prefix(tab_cv_data, prefix_to_remove)

            # Training Data:
            tab_X_train = tab_cv_data[args.cv_fold]['X_train'].astype(np.float32).reset_index(drop=True)
            tab_y_train = tab_cv_data[args.cv_fold]['y_train'].astype(np.float32).reset_index(drop=True)
            tab_train_meta = tab_cv_data[args.cv_fold]['train_meta'].reset_index(drop=True)
            train_fnames = get_dataset_files(tab_train_meta)
            train_cpnds = tab_train_meta.Metadata_InChIKey.to_list()

            # Test Data:
            tab_X_test = tab_cv_data[args.cv_fold]['X_test'].astype(np.float32).reset_index(drop=True)
            tab_y_test = tab_cv_data[args.cv_fold]['y_test'].astype(np.float32).reset_index(drop=True)
            tab_test_meta = tab_cv_data[args.cv_fold]['test_meta'].reset_index(drop=True)
            test_fnames = get_dataset_files(tab_test_meta)
            test_cpnds = tab_test_meta.Metadata_InChIKey.to_list()

        # -----------------------
        #  Load Data
        # -----------------------
        X_img_train, y_train, _, X_tab_train, X_str_train, _ = return_fnames_targets(args.img_loc, train_fnames,
                                                                                     tab_y_train,
                                                                                     train_cpnds, tab_X_train,
                                                                                     tab_train_meta)

        X_test, y_test, test_meta, X_tab_test, X_str_test, _ = return_fnames_targets(args.img_loc, test_fnames,
                                                                                     tab_y_test,
                                                                                     test_cpnds, tab_X_test,
                                                                                     tab_test_meta)

        return {'X_img_train': X_img_train, 'y_train': y_train, 'X_tab_train': X_tab_train,
                'X_str_train': X_str_train, 'X_test': X_test, 'X_tab_test': X_tab_test, 'X_str_test': X_str_test,
                'y_test': y_test, 'test_meta': test_meta}


