import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from modules.models import FusionModel, MlpIBP, MlpStruct, ENetB1
from modules.swin_v2 import SwinTransformerV2


class CwpDataset(torch.utils.data.Dataset):
    """
    Pytorch Dataset Object to load np arrays efficiently for the DataLoader
    Performs channel-wise contrast normalization.
    """

    def __init__(self, numpy_files, y_data, std_data, args, aug, transform=None, tab_data=None, str_data=None):
        self.numpy_files = numpy_files
        self.y_data = y_data
        self.std_data = std_data
        self.ctrl_type = args.ctrl_type
        self.std_type = args.std_type
        self.stnd = args.stnd
        self.norm = args.norm
        self.clip = args.clip
        self.aug = aug
        self.aug_type = args.aug_type
        self.p_ch_aug = args.p_ch_aug
        self.transform = transform
        self.tab_data = tab_data
        self.str_data = str_data

    def __len__(self):
        return len(self.numpy_files)

    def __getitem__(self, idx):
        # Loading numpy array and converting to tensor:
        img_path = self.numpy_files[idx]

        # Generate a well-level ID to amalgamate predictions to:
        well_id = os.path.split(img_path)[-1].split('~f')[0]

        # Load image array:
        arr = np.load(img_path)

        # Clip pixel intensities:
        if self.clip:
            arr = [np.clip(x, np.percentile(x, 0.1), np.percentile(x, 99.9)) for x in arr]

        # Apply standardization on a per-channel, plate-wise basis (PCS):
        if self.stnd:
            arr = standardize_img(arr, img_path, self.std_data, self.ctrl_type, self.std_type)

        # Apply PCN normalization if specified:
        if self.norm:
            arr = np.array([(x - x.min()) / (x.max() - x.min()) for x in arr])

        # Convert np array to a tensor:
        tensor = torch.from_numpy(arr).float()

        # Apply additional channel-weighted augmentation (CWA) if specified:
        if self.aug:
            tensor = apply_aug(tensor, self.aug_type, self.p_ch_aug)

        # Apply transformation if it exists
        if self.transform:
            tensor = self.transform(tensor)

        # If there is tabular/structural data passed to the Dataset class, then return that info for fusion model:
        if self.tab_data is not None:
            return {'pixel_vals': tensor, 'tab_vals': self.tab_data[idx], 'str_vals': self.str_data[idx],
                    'labels': self.y_data[idx], 'well_id': well_id}
        else:
            return {'pixel_vals': tensor, 'labels': self.y_data[idx], 'well_id': well_id}


def model_init(args):
    """ Initializes the model to train. """
    if args.model_type == 'swin':
        model = SwinTransformerV2(img_size=args.resize, drop_path_rate=args.drop_out, window_size=args.window_size,
                                  in_chans=5, num_classes=args.n_classes, patch_size=4,
                                  embed_dim=args.swin_embed, depths=args.swin_depths, num_heads=args.swin_heads)
    elif args.model_type == 'enet':
        # Note: EfficientNetB1 takes in images of size 240*240:
        model = ENetB1(channels=5, num_classes=args.n_classes, drop=args.drop_out)
    elif args.model_type == 'fusion':
        img_model = SwinTransformerV2(img_size=args.resize, drop_path_rate=args.drop_out, window_size=args.window_size,
                                      in_chans=5, num_classes=args.n_classes, patch_size=4,
                                      embed_dim=args.swin_embed, depths=args.swin_depths, num_heads=args.swin_heads)
        tab_model = MlpIBP(args.n_features, args.n_classes, args.n_neurons, args.tab_drop_out)
        struct_model = MlpStruct(2048, args.mlp_hidden, args.mlp_output, args.str_drop_out)
        # Convert final layers to a linear layers:
        img_model.head = nn.Linear(768, int(2 * args.mlp_output))  # Output size = 64
        tab_model.fc6 = nn.Linear(256, args.mlp_output)  # Output size = 32, same as structural model
        # Combine to form CVF fusion model:
        model = FusionModel(img_model, tab_model, struct_model, args.n_classes, args.mlp_output)

    return model


def set_reprod(args):
    # Set seeds to aid reproducibility:
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set benchmark to true for static models to iterate through processing algorithms:
    cudnn.benchmark = True


def standardize_img(arr, img_path, norm_data, ctrl_type, std_type):
    # Extract meta information from image path:
    img_meta = os.path.split(img_path)[-1].split('~')

    # Return the standardization statistics for that metadata:
    stats = norm_data[img_meta[0]][img_meta[1]][img_meta[2]][ctrl_type]

    # Standardize on a channel-wise basis:
    for c in range(5):
        if std_type == 'mean_std':
            arr[c] = (arr[c] - stats['mean'][c]) / stats['std'][c]
        elif std_type == 'med_mad':
            arr[c] = (arr[c] - stats['median'][c]) / stats['MAD'][c]

    return arr


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


def well_acc(well_ids, field_vals, field_preds):
    """ Amalgamates field-level predictions according to well-id and calculates accuracy"""
    uniq_ids = list(set(well_ids))
    correct = 0
    well_level_vals = []
    well_level_preds = []

    for well_id in uniq_ids:
        # Return indices of the unique matching well ids:
        res_idx = [i for i, w in enumerate(well_ids) if w == well_id]
        # Get the actual and model predicted classes at those indices:
        well_f_vals = [field_vals[i] for i in res_idx]
        well_f_preds = [field_preds[i] for i in res_idx]
        # Return the overall well prediction as a mode of field values:
        well_value = max(set(well_f_vals), key=well_f_vals.count)
        well_level_vals.append(well_value)
        well_pred = max(set(well_f_preds), key=well_f_preds.count)
        well_level_preds.append(well_pred)
        # Majority class voting accuracy:
        if well_value == well_pred:
            correct += 1
        else:
            pass

    acc = (correct/len(uniq_ids))*100

    return acc, well_level_vals, well_level_preds


def data_mean_std(img_paths, y_data):
    """
    Returns the mean and std. of a dataset of np arrays.

    :param img_paths: Paths to numpy arrays of images.
    :param y_data: The class labels associated with the paths.
    :return: The mean and standard devaition of the whole image dataset.
    """
    # Instantiate a dataset class and dataloader for the image dataset:
    dataset = CwpDataset(img_paths, y_data)
    dataloader = DataLoader(dataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=0)

    cum_mean = 0
    cum_std = 0

    for images, labels in dataloader:
        # shape of images = [b,c,w,h] -> calculates metrics by channel:
        mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
        cum_mean += mean
        cum_std += std

    data_mean = cum_mean / len(dataloader)
    data_std = cum_std / len(dataloader)

    return data_mean, data_std


def remove_prefix(state_dict):
    new_state_dict = {}
    # Remove "module." prefix from keys
    for key in state_dict.keys():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]

    return new_state_dict


def random_intensity_shift(img, intensity_range):
    """
    Randomly shifts the intensity of the image by a factor in the given range (-1 and 1).
    """
    factor = random.uniform(intensity_range[0], intensity_range[1])
    img = np.clip(img * (1 + factor), 0, 1)
    return img


def random_brightness_change(img, brightness_range):
    """
    Randomly changes the brightness of the image by a factor in the given range (-1 and 1).
    """
    offset = random.uniform(brightness_range[0], brightness_range[1])
    img = np.clip(img + offset, 0, 1)
    return img


def apply_aug(img_tensor, aug_type, p_ch_aug):
    """
    Applies brightness and intensity augmentations to the tensor image.
    "Prob apply" is the probability that each augmentation is applied.
    """
    prob_aug = 0.8
    brightness_range = (-0.1, 0.1)
    intensity_range = (-0.6, 0.6)
    adjusted_channels = []

    if aug_type == 'channel':
        # Applying channel-specific augmentations:
        if np.random.rand() < prob_aug:
            for channel in img_tensor:
                if np.random.rand() < p_ch_aug:
                    channel = random_brightness_change(channel, brightness_range)
                if np.random.rand() < p_ch_aug:
                    channel = random_intensity_shift(channel, intensity_range)
                adjusted_channels.append(channel)
            # Stack resulting tensor list:
            combined_tensor = torch.stack(adjusted_channels)
        else:
            combined_tensor = img_tensor

    elif aug_type == 'field':
        # Apply same augmentations across channels:
        if np.random.rand() < prob_aug:
            b_factor = random.uniform(brightness_range[0], brightness_range[1])
            i_factor = random.uniform(intensity_range[0], intensity_range[1])
            for channel in img_tensor:
                channel = np.clip(channel + b_factor, 0, 1)
                channel = np.clip(channel * (1 + i_factor), 0, 1)
                adjusted_channels.append(channel)
            # Stack resulting tensor list:
            combined_tensor = torch.stack(adjusted_channels)
        else:
            combined_tensor = img_tensor

    return combined_tensor


def apply_gradient_clipping(optimizer, clip_value):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param.grad, clip_value)


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

    return torch.tensor(fprints)

