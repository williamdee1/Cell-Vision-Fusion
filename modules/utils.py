# This work is made available under the Creative Commons Corporation CC BY 4.0 Legal Code.
# To view a copy of this license, visit
# https://github.com/williamdee1/Cell-Vision-Fusion/LICENSE.txt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import os
import re
import glob
import json
from sklearn.utils import class_weight
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

SEED = 42
ARR_FORMAT = '{Metadata_Source}~{Metadata_Batch}~{Metadata_Plate}~{Metadata_Well}.npy'


def prepare_data(dataset, test_size, moa_dict, seed, data_type, cross_val=None, cv_splits=4, norm=None):
    # Replace any NaN values with zeros:
    dataset = dataset.fillna(0)

    # Replace infinite values:
    dataset = dataset.replace([np.inf, -np.inf], 0)

    # Get list of unique compound ids:
    unique_ids = pd.DataFrame(dataset.groupby(['Metadata_JCP2022', 'moa'])['moa'].count()).rename(
        columns={'moa': 'replicates'}).reset_index()

    # Use dataset unique ids for stratified splitting of the dataset into train/test:
    y_uniq = unique_ids['moa'].values
    X_uniq = unique_ids['Metadata_JCP2022'].values

    # Split data by id:
    X_U_train, X_U_test, y_U_train, y_U_test = train_test_split(X_uniq, y_uniq, test_size=test_size,
                                                                random_state=seed, stratify=y_uniq)

    skf = StratifiedKFold(n_splits=cv_splits)

    # Nested k-fold Cross Validation (to be used for Hyperparameter Tuning):
    if cross_val == 'nested':
        # Return nested train and validation set k-fold data:
        cv_data = []
        for train_index, val_index in skf.split(X_U_train, y_U_train):
            X_train, X_val, y_train, y_val, train_meta, val_meta = split_data(dataset, moa_dict, X_U_train[train_index],
                                                              X_U_train[val_index], data_type, norm=norm)
            cv_data.append((X_train, X_val, y_train, y_val, train_meta, val_meta))
            print("X_train: ", len(X_train), " X_val: ", len(X_val),
                  " y_train: ", len(y_train), " y_val: ", len(y_val))
        print("Data split into %s stratified nested folds." % len(cv_data))

        # Return external test data:
        _, X_test, _, y_test, _, test_meta = split_data(dataset, moa_dict, X_U_train, X_U_test, data_type, norm=norm)
        test_data = [X_test, y_test, test_meta]
        print("Set-aside Test Set ---> X_test: ", len(X_test), " y_test: ", len(y_test))
        return cv_data, test_data

    elif cross_val == 'k-fold':
        # Return train and validation set k-fold data:
        cv_data = []
        for train_index, val_index in skf.split(X_uniq, y_uniq):
            X_train, X_val, y_train, y_val, train_meta, val_meta = split_data(dataset, moa_dict,
                                                                               X_uniq[train_index], X_uniq[val_index],
                                                                               data_type, norm=norm)
            cv_data.append((X_train, X_val, y_train, y_val, train_meta, val_meta))
            print("X_train: ", len(X_train), " X_val: ", len(X_val),
                  " y_train: ", len(y_train), " y_val: ", len(y_val))
        print("Data split into %s stratified folds." % len(cv_data))
        return cv_data

    # Otherwise, just return train and test data:
    else:
        X_train, X_test, y_train, y_test, _, _ = split_data(dataset, moa_dict, X_U_train, X_U_test,
                                                            data_type, norm=norm)
        print("X_train: ", len(X_train), " X_test: ", len(X_test),
              " y_train: ", len(y_train), " y_test: ", len(y_test))

        return X_train, X_test, y_train, y_test


def split_data(data, moa_dict, train_ids, test_ids, data_type, norm=None):
    # Filter dataset according to Metadata ids and shuffle:
    train_df = data[data['Metadata_JCP2022'].isin(train_ids)].sample(frac=1)
    test_df = data[data['Metadata_JCP2022'].isin(test_ids)].sample(frac=1)

    # Extract target:
    y_train = train_df.replace({'moa': moa_dict})['moa'].values
    y_test = test_df.replace({'moa': moa_dict})['moa'].values

    if data_type == 'tabular':
        # Metadata features to drop for feature data:
        meta_cols = ['Metadata_Source', 'Metadata_Plate', 'Metadata_Well', 'Metadata_Batch', 'Metadata_JCP2022', 'Metadata_InChI',
                     'Metadata_InChIKey', 'Metadata_PlateType', 'moa', 'target', 'smiles', 'clinical_phase', 'moa_src',
                     'blur_score', 'sat_score', 'focus_score', 'comp_score']
        # Filter meta_cols to include only columns present in train_df
        meta_cols_present = [col for col in meta_cols if col in train_df.columns]
        # Drop Columns from data:
        X_train = train_df.drop(meta_cols_present, axis=1)
        X_test = test_df.drop(meta_cols_present, axis=1)
        # Retain train/test set meta data for later interpretability:
        train_meta = train_df[meta_cols_present].reset_index(drop=True)
        test_meta = test_df[meta_cols_present].reset_index(drop=True)

        # Normalize data:
        if norm is not None:
            X_train, X_test = norm_data(norm, X_train, X_test)

        return X_train, X_test, y_train, y_test, train_meta, test_meta

    elif data_type == 'image':
        X_train = train_df.reset_index(drop=True)
        X_test = test_df.reset_index(drop=True)

        return X_train, X_test, y_train, y_test


def load_cv(cv_path, cv_splits, ibp_data, moa_dict, norm):
    # Load cv splits from json files:
    cv_data = []
    for cv in range(cv_splits):
        file_path = os.path.join(cv_path, f'cv_{cv}.json')
        with open(file_path, 'r') as json_file:
            loaded_data_dict = json.load(json_file)

        # Extract X, y and meta-data for each cv split dictionary:
        cv_split_dict = split_data_dicts(loaded_data_dict, ibp_data, moa_dict, norm)
        cv_data.append(cv_split_dict)

    return cv_data


def split_data_dicts(data_dict, ibp_data, moa_dict, norm):
    # Filter dataset according to Metadata ids and shuffle:
    train_df = ibp_data[ibp_data['Metadata_JCP2022'].isin(data_dict['train'])
                        ].sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = ibp_data[ibp_data['Metadata_JCP2022'].isin(data_dict['val'])
                        ].sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = ibp_data[ibp_data['Metadata_JCP2022'].isin(data_dict['test'])
                        ].sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Extract target labels:
    y_train = train_df.replace({'moa': moa_dict})['moa'].values
    y_val = val_df.replace({'moa': moa_dict})['moa'].values
    y_test = test_df.replace({'moa': moa_dict})['moa'].values

    # Metadata features to drop for feature data:
    meta_cols = ['Metadata_Source', 'Metadata_Plate', 'Metadata_Well', 'Metadata_Batch', 'Metadata_JCP2022',
                 'Metadata_InChI', 'Metadata_InChIKey', 'Metadata_PlateType', 'moa', 'target', 'smiles',
                 'clinical_phase', 'moa_src', 'Metadata_Treatment']
    # Filter meta_cols to include only columns present in train_df
    meta_cols_present = [col for col in meta_cols if col in train_df.columns]

    # Drop Columns from data:
    X_train = train_df.drop(meta_cols_present, axis=1)
    X_val = val_df.drop(meta_cols_present, axis=1)
    X_test = test_df.drop(meta_cols_present, axis=1)
    # Retain train/test set meta data for later interpretability:
    train_meta = train_df[meta_cols_present].reset_index(drop=True)
    val_meta = val_df[meta_cols_present].reset_index(drop=True)
    test_meta = test_df[meta_cols_present].reset_index(drop=True)

    # Normalize data:
    if norm is not None:
        X_train, X_val, X_test = norm_data(norm, X_train, X_val, X_test)

    return {'X_train': X_train, 'y_train': y_train, 'train_meta': train_meta,
            'X_val': X_val, 'y_val': y_val, 'val_meta': val_meta,
            'X_test': X_test, 'y_test': y_test, 'test_meta': test_meta}


def norm_data(norm, X_train, X_test):
    if norm == 'minmax':
        scaler = MinMaxScaler()

    elif norm == 'robust':
        scaler = RobustScaler()

    else:
        print("Please select a scaling method from: 'minmax', 'robust'.")

    # Fit scaler on training data:
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns.tolist())

    # Applying scaler to test set:
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns.tolist())

    return X_train, X_test


def get_dataset_files(data):
    """Converts rows of selected dataframe into .npy filenames which can be loaded."""
    # Create list of image files that are in a given dataset:
    img_files = []

    for _, row in data.iterrows():
        arr_path = ARR_FORMAT.format(**row.to_dict())
        img_files.append(arr_path)

    return img_files


def crop_centre(img, cropx, cropy, fields):
    """Crops a numpy image array to a specified shape around the centre."""
    f, c, y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2

    return img[:fields, :, starty:starty + cropy, startx:startx + cropx]


def adjust_img_arrays(img_dir, save_dir):
    """Given the location of images saved as 4D np arrays, loads each one and converts it to a 3D
    array of shape channels * width * height. Then saves those arrays with a unique field id."""

    # Return a list of files in folder:
    f_paths = glob.glob(img_dir + '*.npy')
    # Extract the names of each file from the file path list:
    img_fnames = [x.split('/')[-1] for x in f_paths]

    failed_paths = []

    for f, file_path in zip(img_fnames, f_paths):
        # Load numpy array:
        try:
            im_arr = np.load(file_path)
        except ValueError:
            print("---> NP Load Failed.")
            failed_paths.append(file_path)
            pass
        # Crop to desired model input shape:
        im_fields = im_arr.shape[0]
        crop_arr = crop_centre(im_arr, 970, 970, im_fields)  # Can alter to change no. fields

        # Convert to 3D array (channels * width * height):
        for i in range(im_fields):
            cwh = crop_arr[i]
            # Generate new name for file based on field and save:
            new_f = f.split('.')[0] + '~f%s.npy' % i
            np.save(save_dir + new_f, cwh)

    print("%s image loads failed." % len(failed_paths))
    print("Failed image paths: ", failed_paths)


def load_cv_data(cv_loc, cross_vals):
    """ Loads data from cross_vals number of folds from cv_loc."""
    cv_data = []

    for cv in range(cross_vals):
        X_train = pd.read_csv(cv_loc + 'X_train_CV_%s.csv' % cv)
        X_val = pd.read_csv(cv_loc + 'X_val_CV_%s.csv' % cv)
        y_train = np.load(cv_loc + 'y_train_CV_%s.npy' % cv)
        y_val = np.load(cv_loc + 'y_val_CV_%s.npy' % cv)

        cv_data.append([X_train, X_val, y_train, y_val])

    return cv_data


def generate_id(args):
    """
    Generates a unique directory for the query results.
    Source:  https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/train.py
    """
    prev_run_dirs = []
    if os.path.isdir(args.out_dir):
        prev_run_dirs = [x for x in os.listdir(args.out_dir) if os.path.isdir(os.path.join(args.out_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1

    if args.dist_type == 'ddp':
        proc_rank = os.environ['RANK']
        run_dir = os.path.join(args.out_dir, f'{cur_run_id:03d}-{args.run_id}-Proc_{proc_rank}')
    else:
        run_dir = os.path.join(args.out_dir, f'{cur_run_id:03d}-{args.run_id}')

    # Creates the directory if it doesn't already exist:
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    return run_dir


def extract_fields(X_images, y_data):
    """
    Converts 4D numpy arrays of shape (fields * channels * image width * height) into 3D arrays
    of shape (channels * width * height), splitting fields out as separate datapoints.
    :param X_images: A list of 4D np arrays
    :param y_data: A 1D numpy array of classes for the X_image data
    :return: A list of 3D np arrays and associated class labels as a 1D np array
    """
    full_X_data = []
    full_y_data = []

    for image, y_t in zip(X_images, y_data):
        fields = image.shape[0]

        for f in range(fields):
            cwh = image[f]
            full_X_data.append(cwh)
            full_y_data.append(y_t)

    return full_X_data, np.array(full_y_data)


def return_fnames_targets(img_loc, fnames, y_data, cpnds, tab_data=None, struct_data=None, umap_label=None):
    """
    Returns the full field file names of all images in img_loc which match the names given by X_data.
    Creates an updated y_data by extending the y values across the field images.
    """
    full_fnames = []
    new_ydata = []
    cpnd_ids = []
    tab_list = []
    prints_list = []
    umap_lbls = []

    if struct_data is not None:
        fprints = convert_fprints(struct_data.smiles.to_list(), 'morgan')
    else:
        fprints = [0] * len(fnames)

    for i, (f, y, c, p) in enumerate(zip(fnames, y_data, cpnds, fprints)):
        # For each file, find all np arrays which match the filenames specified in X_data:
        matching = glob.glob(os.path.join(img_loc, '%s*' % f.split('.')[0]))
        # Append all to image file list:
        for field_img in matching:
            full_fnames.append(field_img)
            new_ydata.append(y)
            cpnd_ids.append(c)
            # Include tabular and structural data if using a fusion architecture
            if tab_data is not None:
                tab_list.append(tab_data.loc[i])
            if struct_data is not None:
                prints_list.append(p)
            if umap_label is not None:
                umap_lbls.append(umap_label.loc[i])

    tab_df = pd.DataFrame(tab_list).reset_index(drop=True)

    return full_fnames, np.array(new_ydata), cpnd_ids, tab_df, np.array(prints_list), umap_lbls


def get_class_weights(y_train):
    """
    Takes the y values and returns a balanced class weight dictionary to be passed to the loss function.
    :param y_train: class labels within y_train
    :return: tensor of class weights
    """
    # List of unique y_values:
    y_unique = np.unique(np.array(y_train))

    # Computing class weights based on data:
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=y_unique,
                                                      y=np.array(y_train))

    # Create a dictionary of weights:
    class_weights_dict = dict(enumerate(class_weights))

    # Convert weights dict to tensor:
    class_weights_tens = torch.tensor(list(class_weights_dict.values()), dtype=torch.float32)

    return class_weights_tens


def image_splits(image, crop_size):
    """
    Create a list of images cropped from an original image.
    Overlaps the last vertical/horizontal images dependent on image size.
    """
    # Get image width and height from shape:
    img_h, img_w = image.shape

    # Calculate number of height/width crops:
    w_crops = round(img_w / crop_size)
    h_crops = round(img_h / crop_size)

    w_list = []
    h_list = []

    for w in range(w_crops):
        w_start = crop_size * w
        if w_start + crop_size > img_w:
            w_start = img_w - crop_size
        w_list.append(w_start)

    for h in range(h_crops):
        h_start = crop_size * h
        if h_start + crop_size > img_h:
            h_start = img_h - crop_size
        h_list.append(h_start)

    # Combine the crop locations into one list of splits:
    split_list = []

    for i in h_list:
        for j in w_list:
            split = image[i:i + crop_size, j:j + crop_size]
            split_list.append(split)

    return split_list


def morgan_fprints(smiles):
    molecules = Chem.MolFromSmiles(smiles)
    fingerprints = AllChem.GetMorganFingerprintAsBitVect(molecules, 2)
    x_array = []
    arrays = np.zeros(0, )
    DataStructs.ConvertToNumpyArray(fingerprints, arrays)
    x_array.append(arrays)
    x_array = np.asarray(x_array)
    x_array = ((np.squeeze(x_array)).astype(int))

    return x_array


def convert_fprints(smiles, fprint_type):
    # Convert to cannonical smiles
    can_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), True) for smi in smiles]

    # Convert canonical smiles to fingerprints:
    fprints = np.zeros((len(can_smiles), 2048), dtype=np.float32)

    for f in range(fprints.shape[0]):
        fprints[f] = morgan_fprints(can_smiles[f])

    return fprints


class Config:
    """
    Converts a yaml config dictionary to argparse format.
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)


