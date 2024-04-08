# This work is made available under the Creative Commons Corporation CC BY 4.0 Legal Code.
# To view a copy of this license, visit
# https://github.com/williamdee1/Cell-Vision-Fusion/LICENSE.txt

import os
import glob
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
import seaborn_image as isns
isns.set_context("notebook")
import seaborn as sns
from modules.viz import plot_learning_rate


def train_IBP_model(cv_fold, X_train, X_val, y_train, y_val, val_meta,
                    model, opt, criterion, device, run_dir, args,
                    scheduler, early_stopper=None):

    # Create folder for plots:
    plot_dir = os.path.join(run_dir, 'plots/')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Create folder for saving models:
    mod_dir = os.path.join(run_dir, 'models/')
    if not os.path.exists(mod_dir):
        os.makedirs(mod_dir)

    # Converting to tensors:
    # Target data:
    y_train_res = y_train.astype('float32').reshape((-1, 1))
    y_val_res = y_val.astype('float32').reshape((-1, 1))
    y_train_tens = torch.tensor(y_train_res, dtype=torch.long)
    y_val_tens = torch.tensor(y_val_res, dtype=torch.long)

    # Feature Data:
    X_train_tens = torch.tensor(X_train.values, dtype=torch.float32)
    X_val_tens = torch.tensor(X_val.values, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tens, y_train_tens)
    val_dataset = TensorDataset(X_val_tens, y_val_tens)

    # Create Dataloader Objects:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers,
                            shuffle=False, drop_last=False)

    # To track the min/max validation loss/ accuracy through epochs, set initially to infinity/zero:
    min_val_loss = np.Inf
    max_val_acc = 0

    # Best models throughout training:
    best_vl_model = 0
    best_va_model = 0
    mdl_list = []

    # Lists to hold accuracies and losses through epochs:
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    cpnd_acc_list = []

    # List to track learning rate through epochs:
    lr_tracker = []

    for epoch in range(args.n_epochs):
        # Setting initial losses/accuracy to zero, to be updated within batch loops
        train_loss_epoch = 0
        train_acc_epoch = 0
        val_loss_epoch = 0
        val_acc_epoch = 0
        cpnd_acc_epoch = 0
        batch_start = time.process_time()

        # Lists to hold well vals and predictions through batches:
        val_vals = []
        val_preds = []
        val_pred_proba = []

        # ---- Set model to TRAINING mode ---- #
        model.train()

        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Set gradients of model parameters to zero:
            opt.zero_grad()

            # Make model predictions on the batch data:
            y_pred_batch = model(X_batch)

            # Apply loss function to batch preds:
            train_batch_loss = criterion(y_pred_batch, y_batch.flatten())

            # Classification accuracy within batch, to use to calculate overall accuracy:
            train_batch_acc, ypred_vals = calc_acc(y_pred_batch, y_batch.flatten())

            # Calculate the sum of gradients using backward:
            train_batch_loss.backward()

            # Update parameters by performing a single optimization:
            opt.step()

            # Update the running loss/accuracy totals:
            train_loss_epoch += train_batch_loss.item()
            train_acc_epoch += train_batch_acc.item()

        # Disabling gradient calculation to speed up process, as backward will not be called during validation:
        with torch.no_grad():
            # ---- Set model to EVALUATION mode ---- #
            model.eval()

            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # ---- WELL-LEVEL VALIDATION ---- #
                val_pred_batch = model(X_batch)

                # Calculate loss and accuracy on val set:
                val_batch_loss = criterion(val_pred_batch, y_batch.flatten())
                val_batch_acc, ypred_vals = calc_acc(val_pred_batch, y_batch.flatten())

                # Update the running loss/accuracy totals:
                val_loss_epoch += val_batch_loss.item()
                val_acc_epoch += val_batch_acc.item()

                # Update val and preds lists:
                [val_vals.append(v.item()) for v in y_batch]
                [val_preds.append(p.item()) for p in ypred_vals]
                [val_pred_proba.append(b.numpy()) for b in val_pred_batch]

        # ---------------------
        #  Log Results
        # ---------------------
        # Calculate metrics across epoch:
        avg_train_loss = train_loss_epoch / len(train_loader)
        avg_train_acc = train_acc_epoch / len(train_loader)
        avg_val_loss = val_loss_epoch / len(val_loader)
        avg_val_acc = val_acc_epoch / len(val_loader)
        # Calculate Compound level acc:
        cpnd_acc = calc_cpnd_acc(val_meta, val_vals, val_pred_proba)
        time_taken = time.process_time() - batch_start

        # Display:
        print(
            'Epoch {}: Train Loss: {:.3f} | Train Well Acc: {:.2f} |'
            'Val Loss: {:.3f} | Val Well Acc:{:.2f} | Val Cpnd Acc:{:.2f} | Time Taken:{:.0f}s'.format(
                f'{epoch + 1}/{args.n_epochs}', avg_train_loss, avg_train_acc,
                avg_val_loss, avg_val_acc, cpnd_acc, time_taken)
        )

        if (epoch+1) % 5 == 0:
            mdl_list.append(copy.deepcopy(model))

        # ---------------------
        #  Manual Callbacks
        # ---------------------
        # Saving best model (epoch with lowest av. validation loss) throughout training:
        if avg_val_loss <= min_val_loss:
            # Remove previous best model/ Confusion matrices in dir:
            for modpath in glob.glob(os.path.join(mod_dir, '*.pth')):
                os.remove(modpath)
            for cm_path in glob.glob(os.path.join(plot_dir, '*_CM.png')):
                os.remove(cm_path)
            # Save model and plot CM for best model during training:
            mdl_save_loc = os.path.join(mod_dir, 'CV{}_{:.4f}_Epc{:.0f}_VL.pth'.format(cv_fold, avg_val_loss, (epoch+1)))
            torch.save(model.state_dict(), mdl_save_loc)
            cm_save_loc = os.path.join(plot_dir, 'CV{}_{:.4f}_Epc{:.0f}_VL_CM.png'.format(cv_fold, avg_val_loss, (epoch+1)))
            conf_mat(np.array(val_vals), np.array(val_preds), args.model_type, cm_save_loc)
            # Updating min_val_loss to new lowest:
            min_val_loss = avg_val_loss
            best_vl_model = copy.deepcopy(model)
        # # Saving best model (epoch with greatest well validation accuracy) throughout training:
        if avg_val_acc >= max_val_acc:
            # Remove previous best model/ Confusion matrices in dir:
            for modpath in glob.glob(os.path.join(mod_dir, '*VA.pth')):
                os.remove(modpath)
            for cm_path in glob.glob(os.path.join(plot_dir, '*VA_CM.png')):
                os.remove(cm_path)
            # Save model and plot CM for best model during training:
            mdl_save_loc = os.path.join(mod_dir, 'CV{}_{:.4f}_Epc{:.0f}_VA.pth'.format(cv_fold, avg_val_acc, (epoch+1)))
            torch.save(model.state_dict(), mdl_save_loc)
            cm_save_loc = os.path.join(plot_dir, 'CV{}_{:.4f}_Epc{:.0f}_VA_CM.png'.format(cv_fold, avg_val_acc, (epoch+1)))
            conf_mat(np.array(val_vals), np.array(val_preds), args.model_type, cm_save_loc)
            # Updating min_val_loss to new lowest:
            max_val_acc = avg_val_acc
            best_va_model = copy.deepcopy(model)

        # ---- HELPER FUNCTIONS UPDATE ---- #
        epoch_lr = opt.param_groups[-1]['lr']
        # print(f"Epoch Learning Rate: {epoch_lr}")
        lr_tracker.append(epoch_lr)
        if scheduler is not None:
            if args.sched_type == 'cosine':
                scheduler.step(epoch + 1)
            if args.sched_type == 'linear':
                scheduler.step(avg_val_loss)
        if early_stopper is not None:
            if early_stopper.early_stop(avg_val_loss):
                break

        # Appending lists outside epoch loop to store results:
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)
        cpnd_acc_list.append(cpnd_acc)

    # ---------------------
    #  Storing Metrics
    # ---------------------
    res_dict = {'train_loss': train_loss_list, 'train_acc': train_acc_list, 'val_loss': val_loss_list,
                'val_acc': val_acc_list, 'cpnd_acc': cpnd_acc_list}

    # Plotting training stats:
    plot_training_curves(res_dict, plot_dir)
    plot_learning_rate(lr_tracker, epoch+1, plot_dir)

    min_vl_idx = val_loss_list.index(min(val_loss_list))
    max_va_idx = val_acc_list.index(max(val_acc_list))
    print('-----> CV-FOLD {} Training Complete: | Lowest Val Loss: {:.3f} @ Epoch {} | Max. Val Acc:{:.3f} @'
          ' Epoch {}:'.format(cv_fold+1, min(val_loss_list), min_vl_idx + 1, max(val_acc_list), max_va_idx + 1))

    # Save final model after training:
    final_model_sl = os.path.join(mod_dir, 'CV{}_{:.4f}_Epc{:.0f}_END.pth'.format(cv_fold, avg_val_loss, (epoch + 1)))
    torch.save(model.state_dict(), final_model_sl)

    return best_vl_model, best_va_model, mdl_list


def test_eval(cv_fold, model, X_test, y_test, test_meta, device, run_dir, model_type, verbose=True):
    # Target data:
    y_test_res = y_test.astype('float32').reshape((-1, 1))
    y_test_tens = torch.tensor(y_test_res, dtype=torch.long)

    # Feature Data
    X_test_tens = torch.tensor(X_test.values, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tens, y_test_tens)

    # Initialize Pytorch DataLoader:
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=0,
                            shuffle=False, drop_last=False)

    test_vals = []
    test_preds = []
    test_proba = []

    with torch.no_grad():
        model.eval()

        for X_batch, y_batch in test_loader:
            # Assign loader objects to device:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Make predictions on test batch:
            test_pred_batch = model(X_batch)
            proba, class_preds = pred_probs(test_pred_batch)

            # Append items to lists outside of loop
            [test_vals.append(v.item()) for v in y_batch]
            [test_preds.append(c.item()) for c in class_preds]
            [test_proba.append(p.detach().cpu().numpy()) for p in proba]

    # Create dataframe of results:
    res_df = pd.DataFrame({'preds': test_preds, 'proba': test_proba, 'actual': y_test,
                           'cpnd_ids': test_meta})
    # Save Eval. Stats:
    eval_dir = os.path.join(run_dir, 'eval/')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    res_df.to_csv(os.path.join(eval_dir, 'well_preds.csv'), index=False)

    cpnd_df = amalgamate_preds(res_df, 'cpnd_ids')
    cpnd_df.to_csv(os.path.join(eval_dir, 'cpnd_preds.csv'), index=False)
    acc, f1 = summary_stats(cpnd_df, 'Cpnd-level', verbose)

    # Confusion Matrix:
    plot_dir = os.path.join(run_dir, 'plots/')
    cm_save_loc = os.path.join(plot_dir, 'CV{}_{:.2f}Acc_{:.2f}F1_TEST_CM.png'.format(cv_fold, acc, f1))
    conf_mat(np.array(test_vals), np.array(test_preds), model_type, cm_save_loc)

    return acc, f1, test_proba


def calc_acc(y_pred, y_test):
    # Applies a softmax followed by a log to the y_pred tensor:
    y_pred_torch = torch.log_softmax(y_pred, dim=1)

    # Returns the predicted values for each batch datapoint
    _, y_pred_values = torch.max(y_pred_torch, dim=1)

    # Calculates correct predictions by comparing to actual values
    correct_pred = (y_pred_values == y_test).float()
    accuracy = correct_pred.sum() / len(correct_pred)

    accuracy = accuracy * 100

    return accuracy, y_pred_values


def calc_cpnd_acc(val_meta, y_vals, y_pred_proba):
    pred_df = pd.DataFrame({'Metadata_JCP2022': val_meta, 'Actuals': y_vals})
    grouped_df = pred_df.groupby('Metadata_JCP2022').agg(lambda x: x.mode().iloc[0]).reset_index()

    pred_prob_df = pd.DataFrame({'Metadata_JCP2022': val_meta, 'Pred_Probs': y_pred_proba})
    df_mean_predictions = pred_prob_df.groupby('Metadata_JCP2022')['Pred_Probs'].apply(lambda x: np.mean(
        x.tolist(), axis=0)).reset_index()
    df_mean_predictions['Max_Index'] = df_mean_predictions['Pred_Probs'].apply(lambda x: np.argmax(x))
    av_prob_acc = accuracy_score(grouped_df.Actuals.values, df_mean_predictions.Max_Index.values) * 100

    return av_prob_acc


def pred_probs(torch_preds):
    class_probs = F.softmax(torch_preds, dim=1)

    _, class_preds = torch.max(class_probs, dim=1)

    return class_probs, class_preds


def amalgamate_preds(res_df, id_type):
    # Group by the ids and average across predictions made for that id (well or compound):
    df_mean_preds = res_df.groupby(id_type)['proba'].apply(
        lambda x: np.mean(x.tolist(), axis=0)).reset_index()

    # Create new column for the actual prediction at that level:
    df_mean_preds['preds'] = df_mean_preds['proba'].apply(lambda x: np.argmax(x))

    # Add column for prediction certainty (i.e. 88% certain of being class 'X'):
    df_mean_preds['cert'] = df_mean_preds['proba'].apply(lambda x: round(max(x) * 100, 2))

    # Add back in the actual values (using mode as all wells/compounds with same id have same label):
    df_mean_preds['actuals'] = res_df.groupby(id_type)['actual'].apply(
        lambda x: x.mode().iloc[0]).reset_index().actual.values

    return df_mean_preds


def summary_stats(df, pred_level, verbose):
    acc = accuracy_score(df.actuals, df.preds) * 100
    f1 = f1_score(df.actuals, df.preds, average='macro') * 100
    # roc_auc = roc_auc_score(df.actuals, np.array(df['proba'].tolist()), average='macro', multi_class='ovr') * 100
    if verbose:
        print("{} Accuracy: {:.2f}%".format(pred_level, acc))
        print("{} f1 score: {:.2f}%".format(pred_level, f1))
    # print("{} ROC_AUC: {:.2f}".format(pred_level, roc_auc))

    return acc, f1


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
    if save_loc is not None:
        plt.savefig(save_loc, bbox_inches='tight')
    plt.close('all')


def plot_training_curves(res_dict, run_dir):
    """
    Plots training and validation loss and accuracy throughout training.
    """
    train_data = pd.DataFrame.from_dict(res_dict)                   # Convert to pandas dataframe
    train_data.insert(0, 'epoch', range(0, len(train_data)))        # Insert epoch column
    csv_save = os.path.join(run_dir, 'train_data.csv')
    train_data.to_csv(csv_save, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), dpi=100)

    sns.lineplot(data=train_data, x="epoch", y="train_loss", ax=ax1, color='coral', label="Train Loss")
    sns.lineplot(data=train_data, x="epoch", y="val_loss", ax=ax1, color='mediumaquamarine', label="Val. Loss")
    sns.lineplot(data=train_data, x="epoch", y="train_acc", ax=ax2, color='coral', label="Train Accuracy")
    sns.lineplot(data=train_data, x="epoch", y="val_acc", ax=ax2, color='mediumaquamarine', label="Val. Accuracy")
    sns.lineplot(data=train_data, x="epoch", y="cpnd_acc", ax=ax2, color='deeppink', label="Compound Accuracy")

    ax1.set_xlabel("Training Epochs")
    ax1.set_ylabel("Cross Entropy Loss")
    ax2.set_xlabel("Training Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax1.set_xlim(0)
    ax2.set_xlim(0)

    ax2.legend(loc='lower right')
    plt.savefig('%s/Training_Curves.png' % run_dir, bbox_inches='tight')
    plt.close('all')


class EarlyStopper:
    # Source: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

