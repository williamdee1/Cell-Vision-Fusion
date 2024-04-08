# This work is made available under the Creative Commons Corporation CC BY 4.0 Legal Code.
# To view a copy of this license, visit
# https://github.com/williamdee1/Cell-Vision-Fusion/LICENSE.txt

import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, average_precision_score
import torch
import pandas as pd
import copy
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from modules.model_utils import CwpDataset
import os
import pickle
from matplotlib import pyplot as plt
import seaborn as sns


def pred_probs(torch_preds):
    class_probs = F.softmax(torch_preds, dim=1)

    _, class_preds = torch.max(class_probs, dim=1)

    return class_probs, class_preds


def amalgamate_preds(res_df, id_type):
    if 'actual' in res_df.columns:
        res_df.rename(columns={'actual': 'actuals'}, inplace=True)

    # Group by the ids and average across predictions made for that id (well or compound):
    df_mean_preds = res_df.groupby(id_type)['proba'].apply(
        lambda x: np.mean(x.tolist(), axis=0)).reset_index()

    # Create new column for the actual prediction at that level:
    df_mean_preds['preds'] = df_mean_preds['proba'].apply(lambda x: np.argmax(x))

    # Add column for prediction certainty (i.e. 88% certain of being class 'X'):
    df_mean_preds['cert'] = df_mean_preds['proba'].apply(lambda x: round(max(x) * 100, 2))

    # Add back in the actual values (using mode as all wells/compounds with same id have same label):
    df_mean_preds['actuals'] = res_df.groupby(id_type)['actuals'].apply(
        lambda x: x.mode().iloc[0]).reset_index().actuals.values

    return df_mean_preds


def summary_stats(df, pred_level, eval_dir):
    acc = accuracy_score(df.actuals, df.preds)
    f1 = f1_score(df.actuals, df.preds, average='macro')
    precision = precision_score(df.actuals, df.preds, average='macro', zero_division=0)
    recall = recall_score(df.actuals, df.preds, average='macro')

    # Create a prediction array from probs:
    pred_arr = np.array(df['proba'].tolist())
    roc_auc = roc_auc_score(df.actuals, pred_arr, average='macro', multi_class='ovr')

    # Calculate AUPR for each class
    aupr_scores = [average_precision_score(df.actuals == class_index, pred_arr[:, class_index]
                                           ) for class_index in range(pred_arr.shape[1])]
    mean_aupr = np.mean(aupr_scores)

    # Save Results:
    with open('{}/{}_{:.2f}_Acc_{:.2f}_F1.txt'.format(eval_dir, pred_level, acc*100, f1*100), 'w') as file:
        file.write('Accuracy: {:.2f}%\n'.format(acc*100))
        file.write('F1 Score: {:.2f}%\n'.format(f1*100))
        file.write('Precision: {:.2f}%\n'.format(precision * 100))
        file.write('Recall: {:.2f}%\n'.format(recall * 100))
        file.write('ROC AUC: {:.2f}%\n'.format(roc_auc * 100))
        file.write('AUPR: {:.2f}%\n'.format(mean_aupr * 100))

    return acc, f1


def model_eval(model, X_test, y_test, test_meta, cv, verbose=True, save_path=None):
    # Make predictions on the test/validation set:
    y_pred = model.predict(X_test)
    preds = [round(value) for value in y_pred]
    y_pred_proba = model.predict_proba(X_test)

    # Calculate well-level accuracy of predictions:
    well_acc = accuracy_score(y_test, preds)
    well_f1 = f1_score(y_test, preds, average='macro')

    # Calculate compound-level accuracy:
    pred_df = pd.DataFrame({'Metadata_JCP2022': test_meta, 'Actuals': y_test, 'Predictions': preds})
    grouped_df = pred_df.groupby('Metadata_JCP2022').agg(lambda x: x.mode().iloc[0]).reset_index()
    # Mean Method:
    pred_prob_df = pd.DataFrame({'Metadata_JCP2022': test_meta, 'Pred_Probs': y_pred_proba.tolist()})
    df_mean_predictions = pred_prob_df.groupby('Metadata_JCP2022')['Pred_Probs'].apply(lambda x: np.mean(
        x.tolist(), axis=0)).reset_index()
    df_mean_predictions['Max_Index'] = df_mean_predictions['Pred_Probs'].apply(lambda x: np.argmax(x))
    cpnd_acc = accuracy_score(grouped_df.Actuals.values, df_mean_predictions.Max_Index.values)
    cpnd_f1 = f1_score(grouped_df.Actuals.values, df_mean_predictions.Max_Index.values, average='macro')

    if verbose:
        print("Well-level Results: %.2f%% Accuracy | %.2f%% F1 Score" % (well_acc * 100.0, well_f1 * 100.0))
        print("Compound-level Results: %.2f%% Accuracy | %.2f%% F1 Score" % (cpnd_acc * 100.0, cpnd_f1 * 100.0))

    if save_path is not None:
        # Save Confusion Matrix:
        plot_dir = os.path.join(save_path, '/plots/')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        cm_save_loc = os.path.join(plot_dir, 'CV{}_{:.2f}Acc_{:.2f}F1_TEST_CM.png'.format(
            cv, cpnd_acc, cpnd_f1))
        conf_mat(np.array(y_test), np.array(preds), 'ML', cm_save_loc)

    return {'well_acc': well_acc, 'well_f1': well_f1, 'cpnd_acc': cpnd_acc, 'cpnd_f1': cpnd_f1,
            'preds': df_mean_predictions.Max_Index.values, 'actuals': grouped_df.Actuals.values,
            'proba': df_mean_predictions.Pred_Probs}


def cv_fold_eval(model, cv_data, feat_sel=None, verbose=True, save_path=None):
    cv_well_acc = []
    cv_well_f1 = []
    cv_cpnd_acc = []
    cv_cpnd_f1 = []
    fold_preds = []
    fold_actuals = []
    fold_probas = []

    for cv in range(len(cv_data)):
        # If a validation set is present, concatenate training and validation data:
        if cv_data[cv].get('X_val') is not None:
            X_train = pd.concat([cv_data[cv]['X_train'], cv_data[cv]['X_val']]).reset_index(drop=True)
            y_train = np.concatenate((cv_data[cv]['y_train'], cv_data[cv]['y_val']))
        else:
            X_train = cv_data[cv]['X_train']
            y_train = cv_data[cv]['y_train']
        X_test = cv_data[cv]['X_test']

        # Drop columns if using feature selection:
        if feat_sel is not None:
            X_train = X_train[feat_sel].reset_index(drop=True)
            X_test = X_test[feat_sel].reset_index(drop=True)

        # Make a deep copy of the model and fit it to the training data:
        cv_mod = copy.deepcopy(model)
        cv_mod.fit(X_train, y_train)

        # Evaluate Fitted Model:
        test_meta = cv_data[cv]['test_meta'].Metadata_JCP2022.to_list()
        res_dict = model_eval(cv_mod, X_test, cv_data[cv]['y_test'], test_meta, cv, verbose, save_path)

        # Save Model:
        if save_path is not None:
            # Save model:
            cv_mod_path = os.path.join(save_path, 'CV{}_{:.2f}_Acc.pth'.format(cv, res_dict['cpnd_acc']))
            with open(cv_mod_path, 'wb') as file:
                pickle.dump(cv_mod, file)

        cv_well_acc.append(res_dict['well_acc'])
        cv_well_f1.append(res_dict['well_f1'])
        cv_cpnd_acc.append(res_dict['cpnd_acc'])
        cv_cpnd_f1.append(res_dict['cpnd_f1'])
        fold_preds.append(res_dict['preds'])
        fold_actuals.append(res_dict['actuals'])
        fold_probas.append(res_dict['proba'])

    # Calculate Metrics Across Folds:
    mean_well_acc = np.mean(cv_well_acc)
    std_well_acc = np.std(cv_well_acc)
    mean_well_f1 = np.mean(cv_well_f1)
    std_well_f1 = np.std(cv_well_f1)
    mean_cpnd_acc = np.mean(cv_cpnd_acc)
    std_cpnd_acc = np.std(cv_cpnd_acc)
    mean_cpnd_f1 = np.mean(cv_cpnd_f1)
    std_cpnd_f1 = np.std(cv_cpnd_f1)

    # Print Results:
    print("---------- Cross-validated Mean Metrics ----------")
    print("Well-Level: %.2f%% Accuracy (+/- %.2f%%) | %.2f%% F1 Score (+/- %.2f%%) " % (
            mean_well_acc * 100, std_well_acc * 100, mean_well_f1 * 100, std_well_f1 * 100))
    print("Compound-Level: %.2f%% Accuracy (+/- %.2f%%) | %.2f%% F1 Score (+/- %.2f%%) " % (
        mean_cpnd_acc * 100, std_cpnd_acc * 100, mean_cpnd_f1 * 100, std_cpnd_f1 * 100))

    # Concatenate the arrays within the results lists:
    pred_arr = np.concatenate(fold_preds)
    act_arr = np.concatenate(fold_actuals)
    proba_arr = np.concatenate(fold_probas)

    return {'pred_arr': pred_arr, 'act_arr': act_arr, 'proba_arr': proba_arr}


def conf_mat(y_test, y_pred, model_type, save_loc):
    full_dict = {'PI3K': 0, 'p38 MAPK': 1, 'RAF': 2, 'AURK': 3, 'CDK': 4, 'EGFR': 5, 'ROCK': 6,
                'MEK': 7, 'GSK': 8, 'mTOR': 9}

    moa_dict = {v: k for k, v in full_dict.items()}

    confusion_matrix1 = pd.crosstab(y_test,
                                    y_pred,
                                    rownames=['Actual'], colnames=['Predicted'])
    confusion_matrix2 = pd.crosstab(y_test,
                                    y_pred,
                                    rownames=['Actual'], colnames=['Predicted'], normalize='index')

    # Extracting counts and percentages from the matrices above:
    counts = ["{0:,}".format(value) for value in
              confusion_matrix1.to_numpy().flatten()]

    percentages = ["{0:.2%}".format(value) for value in
                   confusion_matrix2.to_numpy().flatten()]

    # Combining counts and percentages as one label:
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(percentages, counts)]

    # Convert class labels:
    confusion_matrix2.index = confusion_matrix2.index.map(moa_dict)   # Convert row names
    confusion_matrix2.rename(columns=moa_dict, inplace=True)          # Convert col names

    # Number of different classes predicted by the model:
    classes = len(set(y_test))
    uniq_pred_clss = len(set(y_pred))

    # Reshaping the labels to fit the array:
    labels = np.asarray(labels).reshape(classes, uniq_pred_clss)

    # Plotting the heatmap:
    plt.figure(figsize=(16, 12), dpi=100)
    sns.heatmap(confusion_matrix2, annot=labels, fmt='', cmap='BuPu', vmin=0, vmax=1)
    plt.title("%s Confusion Matrix\n" % model_type, fontweight='bold')
    plt.savefig(save_loc, bbox_inches='tight')
    plt.close('all')


def nn_model_eval(args, model, modpath, device, X_test, y_test, test_meta, std_data, run_dir,
                  tab_data=None, str_data=None):
    # Create folder for evaluation plots:
    model_id = os.path.split(modpath)[-1].split('.')[0]
    eval_dir = os.path.join(run_dir, '{}_eval/'.format(model_id))
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Convert y_test data format:
    y_test_res = y_test.astype('float32').reshape((-1, 1))
    y_test_tens = torch.tensor(y_test_res, dtype=torch.long)

    # Create dataset and loader:
    tform = torch.nn.Sequential(
        transforms.CenterCrop(args.crop_size),
        transforms.Resize((args.resize, args.resize)),
    )

    if tab_data is not None:
        tab_X_train_tens = torch.tensor(tab_data.values, dtype=torch.float32)
        str_X_train_tens = torch.tensor(str_data, dtype=torch.float32)
        test_dataset = CwpDataset(X_test, y_test_tens, std_data, args, aug=False, transform=tform,
                                  tab_data=tab_X_train_tens, str_data=str_X_train_tens)
    else:
        test_dataset = CwpDataset(X_test, y_test_tens, std_data, args, aug=False, transform=tform)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers,
                             shuffle=False, drop_last=False)

    test_preds = []
    test_proba = []
    well_ids = []

    with torch.no_grad():
        model.eval()

        for i, batch in enumerate(test_loader):
            if tab_data is not None:
                X_img_batch, X_tab_batch, X_str_batch, y_batch, batch_ids = batch['pixel_vals'].to(device), \
                                                                            batch['tab_vals'].to(device), \
                                                                            batch['str_vals'].to(device), \
                                                                            batch['labels'].to(device), batch['well_id']
                test_pred_batch = model(X_img_batch, X_tab_batch, X_str_batch)

            else:
                X_batch, y_batch, batch_ids = batch['pixel_vals'].to(device), batch['labels'].to(device), \
                                              batch['well_id']
                test_pred_batch = model(X_batch)

            proba, class_preds = pred_probs(test_pred_batch)

            # Append items to lists outside of loop
            [test_preds.append(c.item()) for c in class_preds]
            [test_proba.append(p.detach().cpu().numpy()) for p in proba]
            [well_ids.append(x) for x in batch_ids]

    # Create dataframe of results:
    res_df = pd.DataFrame({'preds': test_preds, 'proba': test_proba, 'actuals': y_test,
                           'well_ids': well_ids, 'cpnd_ids': test_meta})

    # ----------------
    #  Log Stats
    # ----------------
    # Amalgamate predictions at a well and compound level and print results:
    well_df = amalgamate_preds(res_df, 'well_ids')
    well_acc, well_f1 = summary_stats(well_df, 'Well', eval_dir)

    cpnd_df = amalgamate_preds(res_df, 'cpnd_ids')
    cpnd_acc, cpnd_f1 = summary_stats(cpnd_df, 'Cpnd', eval_dir)

    # Convert proba from a list of numbers, to strings and save:
    for df in [res_df, well_df, cpnd_df]:
        df['proba'] = df['proba'].apply(lambda x: ', '.join(map(str, x)))
    res_df.to_csv(os.path.join(eval_dir, 'results_df.csv'), index=False)
    well_df.to_csv(os.path.join(eval_dir, 'well_preds.csv'), index=False)
    cpnd_df.to_csv(os.path.join(eval_dir, 'cpnd_preds.csv'), index=False)

    # Save confusion matrices:
    well_cm_save = 'well_CM_{:.2f}_Acc.png'.format(well_acc)
    conf_mat(well_df['actuals'], well_df['preds'], 'DL', os.path.join(eval_dir, well_cm_save))
    cpnd_cm_save = 'cpnd_CM_{:.2f}_Acc.png'.format(cpnd_acc)
    conf_mat(cpnd_df['actuals'], cpnd_df['preds'], 'DL', os.path.join(eval_dir, cpnd_cm_save))

