import argparse
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import torch
import logging
import json
import glob
import os
import yaml
from modules.utils import generate_id, load_cv, Config
from modules.model_train import train_nn_model
from modules.fusion_train import train_fusion_model
from modules.model_utils import remove_prefix, model_init
from modules.lr_scheduler import build_scheduler
from modules.eval_funcs import nn_model_eval
from modules.data_load import load_data
import modules.dist as dist
MOA_DICT = {'PI3K': 0, 'p38 MAPK': 1, 'RAF': 2, 'AURK': 3, 'CDK': 4, 'EGFR': 5, 'ROCK': 6,
            'MEK': 7, 'GSK': 8, 'mTOR': 9}


def run_model(args):
    # ----------------
    #  Logging
    # ----------------
    run_dir = generate_id(args)  # Create a unique dir for output
    log_path = "%s/train_log.txt" % run_dir  # Save logs to run_dir
    logging.basicConfig(filename=log_path, level=logging.INFO)  # Initialize logging
    with open('%s/args_vars.txt' % run_dir, 'w') as file:  # Record argparse parameters
        for key, value in vars(args).items():
            file.write(f"{key}: {value}\n")

    # ----------------
    #  Data Load
    # ----------------
    logging.info("[------ Loading CV Data -------]")
    ki_ibp = pd.read_csv(args.data_loc)  # Kinase inhibitor IBP data
    cv_data = load_cv(args.cv_loc, args.cv_folds, ki_ibp, MOA_DICT, norm=None)  # Cross-val data
    cv_results = pd.read_csv('data/shap_fs_res.csv')  # Shapley feature ranking
    top_feats = cv_results['features'][0:args.n_features].tolist()  # Extract n top features

    # Return loaded data dictionary:
    data_dict = load_data(args.model_type, cv_data, top_feats, args)

    # Load the required information based on model_type:
    if (args.model_type == 'swin') or (args.model_type == 'enet'):
        logging.info(f"[------ Loading Raw Image Data ({args.model_type}) -------]")
        X_train, y_train, X_test, y_test, test_meta = data_dict['X_train'], data_dict['y_train'], \
                                                         data_dict['X_test'], data_dict['y_test'], data_dict['test_meta']

    elif args.model_type == 'fusion':
        logging.info("[------ Loading Fusion Data -------]")
        X_img_train, y_train, X_tab_train, X_str_train = data_dict['X_img_train'], data_dict['y_train'], \
                                                         data_dict['X_tab_train'], data_dict['X_str_train'],
        X_img_test, y_test, test_meta, X_tab_test, X_str_test = data_dict['X_test'], data_dict['y_test'], \
                                                                data_dict['test_meta'], data_dict['X_tab_test'], \
                                                                data_dict['X_str_test']

    logging.info("[-----> %s Training Images -------]" % len(y_train))
    logging.info("[-----> %s Test Images -------]" % len(y_test))

    # ----------------------
    #  Data Standardization
    # ----------------------
    std_file = open(args.norm_loc)
    std_data = json.load(std_file)

    # ----------------
    #  Initialization
    # ----------------
    logging.info("[-----> Initializing Model... -------]")
    model = model_init(args)  # Initialize model
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),  # Optimizer
                            eps=1e-8, weight_decay=args.opt_wd)
    cuda = torch.cuda.is_available()  # Assign model to GPU if present
    if cuda:
        model, criterion, device = dist.gpu_init_dp(model, criterion)
    else:
        device = "cpu"

    logging.info("[------ Initialization Complete -------]")

    # ---------------------
    #  Schedule Init.
    # ---------------------
    if args.sched_type == 'none':
        lr_scheduler = None
    else:
        lr_scheduler = build_scheduler(args, optimizer)

    # ----------------
    #  Model Training
    # ----------------
    logging.info("[------ Beginning Model Training -------]")

    if (args.model_type == 'swin') or (args.model_type == 'enet'):
        train_nn_model(X_train, X_test, y_train, y_test, 'image', std_data, MOA_DICT,
                       model, optimizer, criterion, device, run_dir, args, scheduler=lr_scheduler)

    elif args.model_type == 'fusion':
        train_fusion_model(X_img_train, X_img_test, X_tab_train, X_tab_test, X_str_train, X_str_test,
                           y_train, y_test, std_data, model, optimizer, criterion,
                           device, run_dir, args, scheduler=lr_scheduler)

    # ------------------
    #  Model Evaluation
    # ------------------
    logging.info("[------ Beginning Model Evaluation -------]")
    for modpath in glob.glob(os.path.join(run_dir, '*.pth')):
        logging.info("Evaluating model: %s" % modpath)
        # Create a model to load weights into:
        eval_model = model_init(args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load saved weights:
        state_dict = torch.load(modpath, map_location=device)
        state_dict = remove_prefix(state_dict)
        # Load weights into model arch:
        eval_model.load_state_dict(state_dict)
        eval_model.to(device)
        # Run evaluation of saved model
        if (args.model_type == 'swin') or (args.model_type == 'enet'):
            nn_model_eval(args, eval_model, modpath, device, X_test, y_test, test_meta, std_data, run_dir)
        elif args.model_type == 'fusion':
            nn_model_eval(args, eval_model, modpath, device, X_img_test, y_test, test_meta, std_data, run_dir,
                          X_tab_test, X_str_test)
        # Remove model after evaluation:
        # os.remove(modpath)

    logging.info("[------ Eval. Complete -------]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', required=True)
    parse_args = parser.parse_args()
    with open(parse_args.config, 'r') as f:  # Load config file
        config = yaml.load(f, Loader=yaml.SafeLoader)
    args = Config(**config)  # Convert config dictionary to args parse format

    run_model(args)
