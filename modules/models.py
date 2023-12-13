import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.moab import append_0_s, append_0, append_1, append_1_d, ConvStack
from efficientnet_pytorch import EfficientNet


class FusionModel(nn.Module):
    """
    CVF Fusion Model
    """
    def __init__(self, img_model, tab_model, struct_model, num_classes, model_ouput_size):
        super(FusionModel, self).__init__()
        self.img_model = img_model
        self.tab_model = tab_model
        self.struct_model = struct_model

        # MOAB Combination:
        self.conv_stack = ConvStack(4, 1)
        moab_dim = int((2*model_ouput_size+1) ** 2)
        self.fc = nn.Linear(moab_dim, 512)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_out = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_img, x_tab, x_struct):

        x_img = self.img_model(x_img)
        x_tab = self.tab_model(x_tab)
        x_struct = self.struct_model(x_struct)

        # Concatenate structural and ibp data:
        x_str_tab = torch.cat((x_tab, x_struct), dim=1)

        # Apply MOAB:
        # outer addition branch (appending 0)
        x_add = append_0(x_img, x_str_tab)
        x_add = torch.unsqueeze(x_add, 1)
        # outer subtraction branch (appending 0)
        x_sub = append_0_s(x_img, x_str_tab)
        x_sub = torch.unsqueeze(x_sub, 1)
        # outer product branch (appending 1)
        x_pro = append_1(x_img, x_str_tab)
        x_pro = torch.unsqueeze(x_pro, 1)
        # outer division branch (appending 1)
        x_div = append_1_d(x_img, x_str_tab)
        x_div = torch.unsqueeze(x_div, 1)
        # combine 4 branches on the channel dim
        x = torch.cat((x_add, x_sub, x_pro, x_div), dim=1)

        # Post-MOAB Layers:
        x = self.conv_stack(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


class MlpIBP(nn.Module):
    """
    MLP Model for IBP Tabular Data
    """
    def __init__(self, input_size, num_classes, n_neurons, tab_drop_out):
        super(MlpIBP, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, n_neurons)
        self.fc2 = nn.Linear(n_neurons, int(n_neurons / 2))
        self.fc3 = nn.Linear(int(n_neurons / 2), int(n_neurons / 4))
        self.fc4 = nn.Linear(int(n_neurons / 4), int(n_neurons / 6))
        self.fc5 = nn.Linear(int(n_neurons / 6), int(n_neurons / 8))
        self.fc6 = nn.Linear(int(n_neurons / 8), num_classes)

        self.bn1 = nn.BatchNorm1d(n_neurons)
        self.bn2 = nn.BatchNorm1d(int(n_neurons / 2))
        self.bn3 = nn.BatchNorm1d(int(n_neurons / 4))
        self.bn4 = nn.BatchNorm1d(int(n_neurons / 6))
        self.bn5 = nn.BatchNorm1d(int(n_neurons / 8))

        self.dropout = nn.Dropout(tab_drop_out)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = F.elu(self.fc3(x))
        x = self.bn3(x)
        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        x = F.selu(self.fc5(x))
        x = self.bn5(x)
        x = self.fc6(x)

        return x


class MlpStruct(nn.Module):
    """
    MLP Model for Compound Chemical structure data
    """
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(MlpStruct, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        return out


class ENetB1(nn.Module):
    """
    EfficientNet Model, adapted from code:
    https://github.com/pharmbio/CP-Chem-MoA/blob/main/Image_based_model/Efficient_net.ipynb
    """
    def __init__(self, channels, num_classes, drop=0.30):
        super(ENetB1, self).__init__()

        # Define the EfficientNet-B1 model
        # "include_top" removes the final three layers, so they can be replaced.
        base_model = EfficientNet.from_name(model_name='efficientnet-b1', in_channels=channels, include_top=False)
        # print(base_model)

        # Replace with the input feature # from last batch norm. layer:
        num_ftrs = 1280

        # Add dropout layers and classification layer to the modified base model
        self.cnn_model = nn.Sequential(
            base_model,
            nn.Dropout(drop),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(drop),
            nn.Flatten(),
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.cnn_model(x)