import numpy as np
import random
import math
from math import ceil
from glob import glob
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
import plotly
import plotly.express as px
import plotly.io as pio
import seaborn_image as isns
isns.set_context("notebook")
import seaborn as sns
import pandas as pd
import os


def plot_multiple(paths, channel_list, source, img_n, cols, save_loc, seed):
    """
    Plots multiple field images from the same channel and source side-by-side.

    :param paths: Image file full paths
    :param channel_list: list of channels to iterate through e.g. [0,1,2]
    :param source: Which CPG source to extract the images for
    :param img_n: How many images to stack together to make plot
    :param cols: Number of columns to plot data along, must be a multiple of img_n
    :param save_loc: Save location for plot
    :param seed: Ensures same images can be plotted again if needed
    :return: None
    """

    # Retain only file paths of specified source:
    src_fpaths = [x for x in paths if source in x]
    print("-----> %s Image Paths: %s" % (source, len(src_fpaths)))

    # Set img_n to min of img_n and available images:
    img_n = min(img_n, len(src_fpaths))
    img_n = img_n - (img_n % cols)
    # print("-----> IMAGES PLOTTING: ", img_n)

    # Shuffle list so random images are picked:
    random.Random(seed).shuffle(src_fpaths)

    # Create a combined image of channels from different field views from the same source:
    for c in channel_list:
        arrays = []
        for i in range(img_n):
            array = np.load(src_fpaths[i])
            # extract the specified channel array
            ch_arr = array[c]
            arrays.append(ch_arr)

        # Combining arrays to plot:
        rows = ceil(len(arrays) / cols)

        row_arrays = []
        for r in range(rows):
            row_arr = np.hstack(arrays[cols * r: cols * (r + 1)])
            row_arrays.append(row_arr)

        # Vertically stacking the row arrays:
        stacked = np.vstack(np.array(row_arrays))

        plt.figure(figsize=(16, rows * 4))
        plt.imshow(stacked, cmap='turbo')
        plt.axis('off')

        # Save figure:
        save_file = save_loc + 'MULT_%s_ch_%s.png' % (source, c)
        plt.savefig(save_file, dpi=300)
        plt.close()


def plot_channel_dist(paths, channel_list, save_loc, source='All', height=5):
    """
    Plots a heatmap and histogram of the average pixel values for each image channel.

    :param paths: Image file full paths
    :param channel_list: list of channels to iterate through e.g. [0,1,2]
    :param save_loc: Save location for plot
    :param source: What CPG Source to plot data for
    :param height: Height of the graphic
    :return: None
    """
    if source != 'All':
        # Retain only file paths of specified source:
        src_fpaths = [x for x in paths if source in x]
    else:
        src_fpaths = paths

    for c in channel_list:
        # Retain images from specified channel:
        chann_imgs = []

        for path in src_fpaths:
            print("Loading path: ", path)
            a = np.load(path)
            c_img = a[c]
            chann_imgs.append(c_img)

        # Calculate mean values across all images
        mean_val = np.mean(chann_imgs, axis=0)

        # Plot mean image value and distribution for that channel/source:
        h = isns.imghist(mean_val, cmap="magma",
                         height=height, orientation='h', vmin=0, vmax=1200)
        save_file = save_loc + 'DIST_%s_ch_%s.png' % (source, c)
        h.savefig(save_file, dpi=300)


def plot_training_curves(res_dict, run_dir, args):
    """
    Plots training and validation loss and accuracy throughout training.
    """
    train_data = pd.DataFrame.from_dict(res_dict)                   # Convert to pandas dataframe
    train_data.insert(0, 'epoch', range(0, len(train_data)))        # Insert epoch column
    csv_save = os.path.join(run_dir, 'train_data.csv')
    train_data.to_csv(csv_save, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), dpi=100)

    # Only plot train set metrics if using the full training data (i.e. no validation set):
    if args.full_train:
        sns.lineplot(data=train_data, x="epoch", y="train_loss", ax=ax1, color='coral', label="Training Loss")
        sns.lineplot(data=train_data, x="epoch", y="train_acc", ax=ax2, color='coral', label="Training Accuracy")

    else:
        sns.lineplot(data=train_data, x="epoch", y="train_loss", ax=ax1, color='coral', label="Training Loss")
        sns.lineplot(data=train_data, x="epoch", y="val_loss", ax=ax1, color='mediumaquamarine', label="Field Val. Loss")
        sns.lineplot(data=train_data, x="epoch", y="train_acc", ax=ax2, color='coral', label="Training Accuracy")
        sns.lineplot(data=train_data, x="epoch", y="val_acc", ax=ax2, color='mediumaquamarine', label="Field Val. Accuracy")
        sns.lineplot(data=train_data, x="epoch", y="well_acc", ax=ax2, color='deeppink', label="Well Val. Accuracy")

    ax1.set_xlabel("Training Epochs")
    ax1.set_ylabel("Cross Entropy Loss")
    ax2.set_xlabel("Training Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax1.set_xlim(0)
    ax2.set_xlim(0)

    ax2.legend(loc='lower right')
    plt.savefig('%s/Training_Curves.png' % run_dir, bbox_inches='tight')
    plt.close('all')


def plot_learning_rate(lr_per_epoch, num_epoch, run_dir):
    plt.plot([i for i in range(num_epoch)], lr_per_epoch, color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    # Create a ScalarFormatter for the y-axis ticks in scientific notation
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    plt.savefig('%s/LearningRate.png' % run_dir, bbox_inches='tight')
    plt.close('all')


def plot_grad_stats(gn_per_epoch, num_epoch, run_dir):
    epc_num = [i for i in range(num_epoch)]

    means = [np.mean(norms) for norms in gn_per_epoch]
    medians = [np.median(norms) for norms in gn_per_epoch]
    maxima = [np.max(norms) for norms in gn_per_epoch]
    minima = [np.min(norms) for norms in gn_per_epoch]

    plt.plot(epc_num, means, label='Mean', color='salmon')
    plt.plot(epc_num, medians, label='Median', color='goldenrod')
    plt.plot(epc_num, maxima, label='Maximum', color='magenta')
    plt.plot(epc_num, minima, label='Minimum', color='darkcyan')

    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Summary Statistics of Gradient Norms Over Training')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig('%s/GradientNorms.png' % run_dir, bbox_inches='tight')
    plt.close('all')


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
    plt.figure(figsize=(16, 12), dpi=300)
    sns.heatmap(confusion_matrix2, annot=labels, fmt='', cmap='BuPu', vmin=0, vmax=1)
    plt.title("%s Confusion Matrix\n" % model_type, fontweight='bold')
    if save_loc is not None:
        plt.savefig(save_loc, bbox_inches='tight')
    plt.close()


def plot_batch(X_batch, y_batch, batch_ids, moa_dict, batch_size, epoch, plot_dir):
    field_arrays = []
    well_ids = []

    # Limit the plots to 16 images maximum:
    image_limit = 16
    batch_size = min(image_limit, batch_size)

    for i in range(batch_size):
        field_arrays.append(np.array(X_batch[i].cpu()))
        moa = [k for k,v in moa_dict.items() if v == y_batch[i].item()][0]
        well_ids.append(batch_ids[i] + ' - MOA: ' + moa)

    # Stack channels horizontally:
    row_arrs = []
    for i in range(batch_size):
        row_arrs.append(np.hstack(field_arrays[i]))

    # Stack images within batch vertically:
    stacked = np.vstack(np.array(row_arrs))

    # Plot figure:
    plt.figure(figsize=(batch_size * 2, batch_size * 2), dpi=300)
    plt.imshow(stacked, cmap='turbo')

    for well_id, y_pos in zip(well_ids, range(50, stacked.shape[0],  int(stacked.shape[0]/batch_size))):
        plt.text(50, y_pos, well_id, color='white', fontsize='x-small')

    plt.axis('off')

    # Save figure:
    save_loc = os.path.join(plot_dir, 'batch_images_EPC_{:.0f}.png'.format(epoch + 1))
    plt.savefig(save_loc, bbox_inches='tight')
    plt.close('all')


def plot_umap(umap_embedding, cat_dict, int_cat, cmap, title, column, save_loc, jitter=False):
    # jitter the points if required to avoid them obscuring each other:
    if jitter:
        jitter_amount = 0.5
        umap_embedding = umap_embedding + np.random.uniform(
            low=-jitter_amount, high=jitter_amount, size=umap_embedding.shape)

    # Create a scatter plot of the UMAP embedding, color-coded by labels
    scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1],
                          c=int_cat, cmap=cmap, s=30, marker='o', alpha=0.8,
                          edgecolor='black', linewidths=0.5)
    plt.title(title + "\n", fontweight='bold')
    plt.xlabel('UMAP X')
    plt.ylabel('UMAP Y')

    # Create a custom legend with the color and label mapping
    legend_elements = [plt.Line2D([0], [0], marker='o', markeredgecolor='black', linewidth=0.5,
                                  color='w', markerfacecolor=cmap(cat_dict[label]), label=label)
                       for label in cat_dict]
    plt.legend(handles=legend_elements, title=column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(save_loc, bbox_inches='tight')
    plt.close()


def plotly_umap(umap_embedding, labels, lbl_name, plt_title, cmap,
                save_plot=None, jitter=False, point_size=12, point_opac=1.0):
    if jitter:
        jitter_amount = 0.5
        umap_embedding = umap_embedding + np.random.uniform(
            low=-jitter_amount, high=jitter_amount, size=umap_embedding.shape)

    fig = px.scatter(umap_embedding, x=0, y=1,
                     color=labels,
                     labels={'color': lbl_name},
                     # width=800,
                     color_discrete_map=cmap,
                     )

    # Customize the point size, opacity, and add borders
    fig.update_traces(marker=dict(size=point_size, opacity=point_opac,
                                  line=dict(width=0.5, color='#241E4E')))
    fig.update_layout(
        font_size=13,
        legend=dict(font=dict(size=12), itemsizing='constant'),
        legend_title_font_size=point_size,
        xaxis_title="UMAP X",
        yaxis_title="UMAP Y",
        title=plt_title,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.show()

    if save_plot is not None:
        pio.write_image(fig, save_plot, engine='kaleido', scale=3)



def discrete_colors(c_len, c_type):
    colors = [(255, 186, 8), (28, 49, 68), (65, 211, 189), (121, 30, 148), (151, 219, 79), (239, 17, 54),
              (63, 136, 197), (255, 95, 20), (209, 204, 220), (239, 132, 252),
              (100, 200, 150), (180, 90, 120), (50, 150, 200), (230, 50, 180), (65, 123, 90), ]

    if c_len > len(colors):
        raise ValueError(f"The cmap length is {len(c_len)}, but it should be less than {len(colors)}.")

    color_subset = colors[0: c_len]

    # Create colormap from color subset:
    rgb_cm = [tuple([x / 255 for x in i]) for i in color_subset]

    if c_type == 'rgb':
        return rgb_cm
    elif c_type == 'cmap':
        cmap = mcolors.ListedColormap(rgb_cm)
        return cmap


def load_and_plot(path, cmap, orient):
    arr = np.load(path)

    arr_list = []
    [arr_list.append(x) for x in arr]

    if orient == 'vert':
        for a in arr_list:
            plt.figure(figsize=(3, 3))
            plt.imshow(a, cmap=cmap)
            plt.axis('off')
            plt.show()
    elif orient == 'hor':
        stack_arr = np.hstack(arr_list)
        plt.figure(figsize=(12, 16))
        plt.imshow(stack_arr, cmap=cmap)
        plt.axis('off')
        plt.show()
