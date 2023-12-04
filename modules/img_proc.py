import numpy as np


def save_channel_array(well_imgs, data_id, out_dir, channels=5):
    """
    Takes a list of well images (n = channels * fields), separates them by field
    Saves to output dir as a numpy array of shape (channels * width * height).

    :param well_imgs: well images downloaded from s3 bucket
    :param illum_funcs: Illumination functions for the plate
    :param save_file: name to save resulting datapoint
    :param out_dir: directory to save datapoint
    :param channels: number of cell painting channels (default = 5)
    """
    # Number of field images for the image data passed:
    field_no = len(well_imgs) / channels

    # Iterate through the fields and channels and save each field-level array (c*w*h):
    for f in range(int(field_no)):
        # Create an empty list to contain the channel images for each field
        ch_images = [None] * channels

        for c in range(channels):
            well_no = (channels * f) + c
            well_data = well_imgs[well_no]
            # Ensuring channels are saved in correct order:
            ch_images.pop(well_data.channel)
            ch_images.insert(int(well_data.channel), well_data.array)

        # Save array:
        field_arr = np.array(ch_images)

        # Convert list to array and save using unique id and field number:
        save_loc = out_dir + '/' + data_id + '~f' + str(f) + '.npy'
        print("Saving file... ", save_loc)
        np.save(save_loc, field_arr)


def norm_2d(m):
    """ Normalizes a 2D matrix """
    norm = np.linalg.norm(m)
    m = m/norm
    return m


def apply_illumF(image_arr, illumF, norm=False):
    """
    Takes an image as a numpy array and adjusts it according to the channel-specific
    illumination function (illumF).

    :param image_arr: np image array
    :param illumF: channel-specific illumination function
    :param norm = True -> normalizes the image array first.
    :return: np array of a corrected image
    """
    if norm:
        image_arr = norm_2d(image_arr)

    # Divide the image by the illumination function:
    corr_img = np.divide(image_arr, illumF)

    return corr_img

