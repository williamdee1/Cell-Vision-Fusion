# Notebook to download images directly from JUMP-CP AWS Bucket

import argparse
import time
import traceback
import os
import pandas as pd

from modules.s3_funcs import load_data_paths, extract_img_paths, process_images, dl_illum_funcs
from modules.img_proc import save_channel_array

NUM_CHANNELS = 5


def import_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Dataset file containing sample metadata', required=True)
    parser.add_argument('--output_dir', help='Directory to save downloaded images', required=True)
    args = parser.parse_args()
    start = time.time()

    # ---------------------
    #  Output Directories
    # ---------------------
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    fail_dir = os.path.join(output_dir, "errors")
    os.makedirs("illum", exist_ok=True)

    # ---------------------
    #  Load S3 Locations
    # ---------------------
    dataset = pd.read_csv(args.data)
    # Return the locations of the "load_data_csv.csv" files for each datapoint:
    load_data_fmt = 's3://cellpainting-gallery/cpg0016-jump/' \
                    '{Metadata_Source}/workspace/load_data_csv/' \
                    '{Metadata_Batch}/''{Metadata_Plate}/load_data_with_illum.csv.gz'
    load_data, _ = load_data_paths(dataset, load_data_fmt)

    # ---------------------
    #  Image Download
    # ---------------------
    for i, load_path in enumerate(load_data):
        # Return datapoint metadata and print status:
        data_src = dataset.Metadata_Source[i]
        data_batch = dataset.Metadata_Batch[i]
        data_plate = dataset.Metadata_Plate[i]
        well_id = dataset.Metadata_Well[i]
        data_id = '~'.join([data_src, data_batch, data_plate, well_id])
        print("Processing --> Well: %s | Plate: %s | Batch: %s | Source: %s" %
              (well_id, data_plate, data_batch, data_src))

        try:
            # --- Use load_data_csv file data to return well data paths --- #
            # Read load_data_csv file from s3 path:
            ldc = pd.read_csv(load_path)
            # Return just the rows from the plate data which are associated with the selected well datapoint:
            well_data = ldc[ldc['Metadata_Well'] == well_id].reset_index(drop=True)
            # Return the illumination functions for the well data:
            well_illum = dl_illum_funcs(well_data)
            # Return the specific image paths for the well data:
            image_paths = extract_img_paths(well_data)
            print("----> %s Images Processing..." % (len(image_paths) * NUM_CHANNELS))

            # --- Download the well images in parallel and apply illumination functions --- #
            well_imgs = process_images(image_paths, well_illum)

            # Separate well arrays into field-level arrays and save each array separately:
            save_channel_array(well_imgs, data_id, args.output_dir, channels=NUM_CHANNELS)

        except KeyboardInterrupt:
            # Propagate KeyboardInterrupt upwards so process can still be terminated.
            raise
        except Exception as e:
            print()
            print("| Datapoint: %s FAILED | %s / %s complete |\n----------------------------" % (
                data_id, i + 1, len(load_data)))

            # Ensure dir for saving exceptions exists.
            os.makedirs(fail_dir, exist_ok=True)

            # Save the exception using the plate name and current timestamp as
            # the file name.
            fail_file = os.path.join(fail_dir, data_id)
            fail_file += f'-{int(time.time())}'
            with open(fail_file, 'w') as f:
                # Write the full exception stacktrace to the file.
                f.write(traceback.format_exc())

    print("Time taken per image: %s seconds." % round((time.time()-start)/len(load_data), 2))


if __name__ == "__main__":
    import_data()
