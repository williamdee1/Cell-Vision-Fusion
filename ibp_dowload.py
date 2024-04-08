# This work is made available under the Creative Commons Corporation CC BY 4.0 Legal Code.
# To view a copy of this license, visit
# https://github.com/williamdee1/Cell-Vision-Fusion/LICENSE.txt

# Notebook to download CellProfiler profiles directly from JUMP-CP AWS Bucket

import pandas as pd
import argparse
from modules.s3_funcs import process_profiles, load_data_paths


def import_cp_profiles():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Dataset file', required=True)
    parser.add_argument('--out_path', help='Path to save downloaded profiles', required=True)
    args = parser.parse_args()

    # ---------------------
    #  Load S3 Locations
    # ---------------------
    # Read dataset:
    dataset = pd.read_csv(args.data)
    load_data_fmt = 's3://cellpainting-gallery/cpg0016-jump/' \
                    '{Metadata_Source}/workspace/profiles/' \
                    '{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.parquet'
    data_paths, row_data = load_data_paths(dataset, load_data_fmt)

    # ---------------------
    #  Profile Download
    # ---------------------

    # interval to amalgamate datapoints at:
    interval = 500

    # Create empty dataframe to append results to:
    ibp_data = pd.DataFrame()

    for i in range(0, len(data_paths), interval):
        # Download CellProfiler profiles as a list of well datapoints, and concat. into DataFrame:
        ibp_list = process_profiles(data_paths[i:i+interval], row_data[i:i+interval])

        # Insert previously downloaded dataframe into ibp_list:
        ibp_list.insert(0, ibp_data)

        # # Concatenate all the merged dataframes vertically to create the final dataset:
        ibp_data = pd.concat(ibp_list, axis=0, ignore_index=True)

        # Save DataFrame as csv file:
        print("Storing as csv file...")
        ibp_data.to_csv(args.out_path, index=False)


if __name__ == "__main__":
    import_cp_profiles()



