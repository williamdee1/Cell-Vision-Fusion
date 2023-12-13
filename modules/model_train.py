import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import numpy as np
from modules.model_utils import calc_acc, CwpDataset, well_acc
from modules.viz import plot_training_curves, conf_mat, plot_batch, plot_learning_rate
import random
import time
import logging
import os
import glob

# Setting seeds for reproducibility:
seed = 5
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_nn_model(X_train, X_val, y_train, y_val, data_type, std_data, moa_dict,
                   model, opt, criterion, device, run_dir, args,
                   scheduler):

    # Create folder for training plots:
    plot_dir = os.path.join(run_dir, 'plots/')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Converting to tensors:
    # Target data:
    y_train_res = y_train.astype('float32').reshape((-1, 1))
    y_val_res = y_val.astype('float32').reshape((-1, 1))
    y_train_tens = torch.tensor(y_train_res, dtype=torch.long)
    y_val_tens = torch.tensor(y_val_res, dtype=torch.long)

    # Feature Data:
    if data_type == 'tabular':
        # For tabular data, X_train and X_val are dataframes:
        X_train_tens = torch.tensor(X_train.values, dtype=torch.float32)
        X_val_tens = torch.tensor(X_val.values, dtype=torch.float32)
        train_dataset = TensorDataset(X_train_tens, y_train_tens)
        val_dataset = TensorDataset(X_val_tens, y_val_tens)

    elif data_type == 'image':
        # Initialize transformations:
        transform = torch.nn.Sequential(
            transforms.RandomCrop(args.crop_size),
            transforms.Resize((args.resize, args.resize)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        )
        # Don't apply additional transforms to validation set:
        val_t = torch.nn.Sequential(
            transforms.CenterCrop(args.crop_size),
            transforms.Resize((args.resize, args.resize)),
        )

        # For image data, X_train and X_val are np array file paths:
        train_dataset = CwpDataset(X_train, y_train_tens, std_data, args, aug=args.aug, transform=transform)
        val_dataset = CwpDataset(X_val, y_val_tens, std_data, args, aug=False, transform=val_t)

    if args.dist_type == 'ddp':
        # Initialize Distributed Sampler:
        sampler_train = torch.utils.data.DistributedSampler(train_dataset, shuffle=True, seed=seed)
        sampler_val = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)

        # Initialize Pytorch DataLoader:
        train_loader = DataLoader(train_dataset, sampler=sampler_train, batch_size=args.batch_size,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)

        val_loader = DataLoader(val_dataset, sampler=sampler_val, batch_size=args.batch_size,
                                num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                  shuffle=True, drop_last=True)

        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                shuffle=False, drop_last=False)

    # To track the min/max validation loss/ accuracy through epochs, set initially to infinity/zero:
    min_val_loss = np.Inf
    max_val_acc = 0

    # Lists to hold accuracies and losses through epochs:
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    well_acc_list = []

    # List to track learning rate and gradients through epochs:
    lr_tracker = []

    for epoch in range(args.n_epochs):
        # Setting initial losses/accuracy to zero, to be updated within batch loops
        train_loss_epoch = 0
        train_acc_epoch = 0
        val_loss_epoch = 0
        val_acc_epoch = 0
        batch_start = time.process_time()

        # Lists to calculate validation results at a well-level rather than a field level:
        train_well_ids = []
        train_field_vals = []
        train_field_preds = []
        val_well_ids = []
        val_field_vals = []
        val_field_preds = []

        # ---- Set model to TRAINING mode ---- #
        model.train()

        for i, batch in enumerate(train_loader):
            X_batch, y_batch, batch_ids = batch['pixel_vals'].to(device), batch['labels'].to(device), \
                                          batch['well_id']

            # Save images of first batch of train loader:
            # if i == 0:
            #     plot_batch(X_batch, y_batch, batch_ids, moa_dict, args.batch_size, epoch, plot_dir)

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

            # Update well-level tracking lists:
            [train_well_ids.append(x) for x in batch_ids]
            [train_field_vals.append(v.item()) for v in y_batch]
            [train_field_preds.append(p.item()) for p in ypred_vals]

        # Skip validation if not using a val set (i.e. training on train+val set):
        if args.full_train:
            # ---------------------
            #  Log Results
            # ---------------------
            # Calculate metrics across epoch:
            avg_train_loss = train_loss_epoch / len(train_loader)
            avg_train_acc = train_acc_epoch / len(train_loader)
            # Calculate accuracy at a well-level:
            well_train_acc, _, _ = well_acc(train_well_ids, train_field_vals, train_field_preds)
            time_taken = time.process_time() - batch_start

            # Display:
            logging.info(
                'Epoch {}: | Train Loss: {:.3f} | Train Field Acc: {:.2f} | Train Well Acc: {:.2f} |'
                'Time Taken:{:.0f}s'.format(
                    f'{epoch + 1}/{args.n_epochs}', avg_train_loss, avg_train_acc, well_train_acc, time_taken)
            )

        else:
            # Disabling gradient calculation to speed up process, as backward will not be called during validation:
            with torch.no_grad():
                # ---- Set model to EVALUATION mode ---- #
                model.eval()

                for batch in val_loader:
                    X_batch, y_batch, batch_ids = batch['pixel_vals'].to(device), batch['labels'].to(device), \
                                                  batch['well_id']

                    # ---- FIELD-LEVEL VALIDATION ---- #
                    val_pred_batch = model(X_batch)

                    # Calculate loss and accuracy on val set:
                    val_batch_loss = criterion(val_pred_batch, y_batch.flatten())
                    val_batch_acc, ypred_vals = calc_acc(val_pred_batch, y_batch.flatten())

                    # Update the running loss/accuracy totals:
                    val_loss_epoch += val_batch_loss.item()
                    val_acc_epoch += val_batch_acc.item()

                    # Update well-level tracking lists:
                    [val_well_ids.append(x) for x in batch_ids]
                    [val_field_vals.append(v.item()) for v in y_batch]
                    [val_field_preds.append(p.item()) for p in ypred_vals]

            # ---------------------
            #  Log Results
            # ---------------------
            # Calculate metrics across epoch:
            avg_train_loss = train_loss_epoch / len(train_loader)
            avg_train_acc = train_acc_epoch / len(train_loader)
            avg_val_loss = val_loss_epoch / len(val_loader)
            avg_val_acc = val_acc_epoch / len(val_loader)
            # Calculate accuracy at a well-level:
            well_train_acc, _, _ = well_acc(train_well_ids, train_field_vals, train_field_preds)
            well_val_acc, well_vals, well_preds = well_acc(val_well_ids, val_field_vals, val_field_preds)

            time_taken = time.process_time() - batch_start

            # Display:
            logging.info(
                'Epoch {}: | Train Loss: {:.3f} | Train Field Acc: {:.2f} | Train Well Acc: {:.2f} |'
                'Val Loss: {:.3f} | Val Field Acc:{:.2f} | Val Well Acc: {:.2f} | Time Taken:{:.0f}s'.format(
                    f'{epoch + 1}/{args.n_epochs}', avg_train_loss, avg_train_acc, well_train_acc,
                    avg_val_loss, avg_val_acc, well_val_acc, time_taken)
            )

        if (epoch + 1) == args.n_epochs:
            torch.save(model.state_dict(), '%s/CV%s_%s_epc.pth' % (run_dir, args.cv_fold, epoch + 1))

        # ---------------------
        #  Manual Callbacks
        # ---------------------
        if args.full_train:
            pass
        else:
            # Saving best model (epoch with lowest av. validation loss) throughout training:
            if avg_val_loss <= min_val_loss:
                # Remove previous best model/ Confusion matrices in dir:
                for modpath in glob.glob(os.path.join(run_dir, '*_VL.pth')):
                    os.remove(modpath)
                for cm_path in glob.glob(os.path.join(plot_dir, '*_CM.png')):
                    os.remove(cm_path)
                # Save model and plot CM for best model during training:
                mdl_save_loc = os.path.join(run_dir, '{:.2f}_Epc{:.0f}_VL.pth'.format(avg_val_loss, (epoch + 1)))
                torch.save(model.state_dict(), mdl_save_loc)
                cm_save_loc = os.path.join(plot_dir, '{:.2f}_Epc{:.0f}_VL_CM.png'.format(avg_val_loss, (epoch + 1)))
                conf_mat(np.array(well_vals), np.array(well_preds), args.model_type, cm_save_loc)
                # Updating min_val_loss to new lowest:
                min_val_loss = avg_val_loss
            # # Saving best model (epoch with greatest well validation accuracy) throughout training:
            if well_val_acc >= max_val_acc:
                # Remove previous best model/ Confusion matrices in dir:
                for modpath in glob.glob(os.path.join(run_dir, '*_VA.pth')):
                    os.remove(modpath)
                for cm_path in glob.glob(os.path.join(plot_dir, '*VA_CM.png')):
                    os.remove(cm_path)
                # Save model and plot CM for best model during training:
                mdl_save_loc = os.path.join(run_dir, '{:.2f}_Epc{:.0f}_VA.pth'.format(well_val_acc, (epoch + 1)))
                torch.save(model.state_dict(), mdl_save_loc)
                cm_save_loc = os.path.join(plot_dir, '{:.2f}_Epc{:.0f}_VA_CM.png'.format(well_val_acc, (epoch + 1)))
                conf_mat(np.array(well_vals), np.array(well_preds), args.model_type, cm_save_loc)
                # Updating min_val_loss to new lowest:
                max_val_acc = well_val_acc

        # ---- HELPER FUNCTIONS UPDATE ---- #
        epoch_lr = opt.param_groups[-1]['lr']
        lr_tracker.append(epoch_lr)
        if scheduler is not None:
            if args.sched_type == 'cosine':
                scheduler.step(epoch+1)
            if args.sched_type == 'linear':
                scheduler.step(avg_val_loss)

        if args.full_train:
            # Appending lists outside epoch loop to store results:
            train_loss_list.append(avg_train_loss)
            train_acc_list.append(avg_train_acc)

        else:
            train_loss_list.append(avg_train_loss)
            train_acc_list.append(avg_train_acc)
            val_loss_list.append(avg_val_loss)
            val_acc_list.append(avg_val_acc)
            well_acc_list.append(well_val_acc)

    # ---------------------
    #  Storing Metrics
    # ---------------------
    if args.full_train:
        res_dict = {'train_loss': train_loss_list, 'train_acc': train_acc_list}
        logging.info('-----> Training Complete')
    else:
        res_dict = {'train_loss': train_loss_list, 'train_acc': train_acc_list, 'val_loss': val_loss_list,
                    'val_acc': val_acc_list, 'well_acc': well_acc_list}
        min_vl_idx = val_loss_list.index(min(val_loss_list))
        logging.info('-----> Training Complete: | Lowest Val Loss: {:.3f}| Val Acc:{:.3f} | Epoch {}:'.format(
            min(val_loss_list), well_acc_list[min_vl_idx], min_vl_idx + 1))

    # Plotting training stats:
    plot_training_curves(res_dict, plot_dir, args)
    plot_learning_rate(lr_tracker, args.n_epochs, plot_dir)




