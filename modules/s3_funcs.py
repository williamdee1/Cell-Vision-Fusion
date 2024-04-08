# This work is made available under the Creative Commons Corporation CC BY 4.0 Legal Code.
# To view a copy of this license, visit
# https://github.com/williamdee1/Cell-Vision-Fusion/LICENSE.txt

import multiprocessing
import os
import time
from dataclasses import dataclass
from urllib.parse import ParseResult as URL
from urllib.parse import urlparse
import boto3
import numpy as np
import pandas as pd
from botocore.client import Config
from PIL import Image
from modules.img_proc import apply_illumF

# --------------------------------- #
# ------- STATIC VARIABLES -------- #
# --------------------------------- #

CONFIG = Config(connect_timeout=60, retries={'max_attempts': 3})

REGION = 'us-east-1'

# Create a lock object
lock = multiprocessing.Lock()

# Illum formats.
AGP_ILM_FMT = '{PathName_IllumAGP}/{FileName_IllumAGP}'
RNA_ILM_FMT = '{PathName_IllumRNA}/{FileName_IllumRNA}'
MITO_ILM_FMT = '{PathName_IllumMito}/{FileName_IllumMito}'
ER_ILM_FMT = '{PathName_IllumER}/{FileName_IllumER}'
DNA_ILM_FMT = '{PathName_IllumDNA}/{FileName_IllumDNA}'

# Image formats.
AGP_IMG_FMT = '{PathName_OrigAGP}{FileName_OrigAGP}'
RNA_IMG_FMT = '{PathName_OrigRNA}{FileName_OrigRNA}'
MITO_IMG_FMT = '{PathName_OrigMito}{FileName_OrigMito}'
ER_IMG_FMT = '{PathName_OrigER}{FileName_OrigER}'
DNA_IMG_FMT = '{PathName_OrigDNA}{FileName_OrigDNA}'


# --------------------------------- #
# ---------- DATA CLASSES --------- #
# --------------------------------- #

@dataclass
class JobParams:
    channel: int
    img_path: str
    illum_func: np.ndarray


@dataclass
class ProfileParams:
    profile_path: str
    row: object


@dataclass
class JobResult:
    channel: int
    array: np.ndarray


@dataclass
class ProfileResult:
    dframe: pd.DataFrame


@dataclass
class WellImagePaths:
    agp_img: str
    rna_img: str
    mito_img: str
    er_img: str
    dna_img: str


@dataclass
class IllumFuncs:
    agp: np.ndarray
    rna: np.ndarray
    mito: np.ndarray
    er: np.ndarray
    dna: np.ndarray


# ----------------------------------- #
# -------- GENERAL FUNCTIONS -------- #
# ----------------------------------- #

# Create a unique s3 client for each process.
def initialize():
    global s3_client
    s3_client = boto3.resource('s3', region_name=REGION, config=CONFIG)


def load_data_paths(dataset: pd.DataFrame, load_format: str) -> list:
    """
    For each datapoint (row) in dataset, lists the s3 path to the corresponding file.

    :param dataset: selected dataset
    :param load_format: format of data to extract path from dataset row
    :return -> s3 paths as a list
    """
    # List to hold paths to load_data_csv files:
    load_paths = []
    row_data = []

    for _, row in dataset.iterrows():
        load_data_path = load_format.format(**row.to_dict())
        load_paths.append(load_data_path)
        row_data.append(row)

    return load_paths, row_data


# --------------------------------- #
# -------- IMAGE FUNCTIONS -------- #
# --------------------------------- #

def extract_img_paths(well_data: pd.DataFrame) -> list:
    """
    From the passed Dataframe, extract image paths for each image channel.
    :param well_data: Dataframe of image well data metadata.
    :return: well image paths as a list
    """
    well_imgs = []
    
    # Iterate over rows within selected well data and extract image paths using metadata:
    for _, row in well_data.iterrows():
        row_dict = row.to_dict()
        well_imgs.append(WellImagePaths(
            urlparse(AGP_IMG_FMT.format(**row_dict)),
            urlparse(RNA_IMG_FMT.format(**row_dict)),
            urlparse(MITO_IMG_FMT.format(**row_dict)),
            urlparse(ER_IMG_FMT.format(**row_dict)),
            urlparse(DNA_IMG_FMT.format(**row_dict)),
        ))

    return well_imgs


def process_images(img_paths: list, illum_funcs: IllumFuncs):
    """
    Downloads images, applies illum funcs and stores resulting array.

    :param img_paths: full paths of image files in s3
    :param illum_funcs: the illum funcs for each channel in the well images.
    """
    start = time.perf_counter()

    # Set up a multiprocessing pool to download 10 images at once.
    # The initialize func will create a s3 client for each process.
    pool = multiprocessing.Pool(10, initialize)

    # Create list of jobs.
    jobs = []
    for path in img_paths:
        jobs.append(JobParams(0, path.agp_img, illum_funcs.agp))
        jobs.append(JobParams(1, path.rna_img, illum_funcs.rna))
        jobs.append(JobParams(2, path.mito_img, illum_funcs.mito))
        jobs.append(JobParams(3, path.er_img, illum_funcs.er))
        jobs.append(JobParams(4, path.dna_img, illum_funcs.dna))

    # Run the jobs.
    results = pool.map(run_job, jobs)

    end = time.perf_counter()
    print()
    print(f"Processed %s well images (took {end - start:0.2f} seconds)." % len(results))

    return results


def run_job(params: JobParams) -> JobResult:
    """
    Reads an image from s3 bucket path, applies the correct illumination function, stores as a np array
    :param params: parallel job parameters
    """
    # Get the image from s3.
    im = read_image_from_s3(params.img_path)

    # Apply the illumination function.
    corr_img = apply_illumF(im, params.illum_func)

    print(".", flush=True, end='')
    return JobResult(params.channel, corr_img)


def read_image_from_s3(url: URL) -> Image.Image:
    # Get s3 bucket and key from provided URL.
    bucket = url.netloc
    key = url.path.lstrip('/')

    obj = s3_client.Bucket(bucket).Object(key)
    response = obj.get()
    file_stream = response['Body']
    try:
        im = Image.open(file_stream)
    except Exception as e:
        raise Exception("failed to parse image %s/%s" % (bucket, key)) from e

    return im


# -------------------------------- #
# ---- ILLUMINATION FUNCTIONS ---- #
# -------------------------------- #

def dl_illum_funcs(well_data: pd.DataFrame) -> IllumFuncs:
    # Illumination functions are the same across a plate, so only need to extract paths for one datapoint:
    row_dict = well_data.iloc[0].to_dict()

    return IllumFuncs(
        dl_illum_func(urlparse(AGP_ILM_FMT.format(**row_dict))),
        dl_illum_func(urlparse(RNA_ILM_FMT.format(**row_dict))),
        dl_illum_func(urlparse(MITO_ILM_FMT.format(**row_dict))),
        dl_illum_func(urlparse(ER_ILM_FMT.format(**row_dict))),
        dl_illum_func(urlparse(DNA_ILM_FMT.format(**row_dict))),
    )


def dl_illum_func(url: URL) -> np.ndarray:
    """
    Downloads illuminations functions from s3 if they don't already exist locally.
    Loads the function into memory to be used for creating the datapoints.

    :param url: URL to illumination function on s3
    :return: illumination function as np array
    """

    # Get s3 bucket and key from provided URL.
    bucket = url.netloc
    key = url.path.lstrip('/')

    ILLUM_CACHE_DIR = "illum"

    # Ensure the illum func exists locally. Download it if not.
    # Parse s3 path to extract plate name and file name.
    plate_name = key.split('/')[-2]
    file_name = key.split('/')[-1]

    cache_subdir = os.path.join(ILLUM_CACHE_DIR, plate_name)
    local_file_path = os.path.join(cache_subdir, file_name)

    if os.path.isfile(local_file_path):
        print("Loading cached illum func: %s" % file_name)
    else:
        print("Downloading illum func: %s" % file_name)
        os.makedirs(cache_subdir, exist_ok=True)
        s3 = boto3.resource('s3', region_name=REGION, config=CONFIG)
        try:
            s3.Bucket(bucket).download_file(key, local_file_path)
        except Exception as e:
            raise Exception("failed to download illum func %s/%s" % (bucket, key)) from e

    # Load the func and return it.
    return np.load(local_file_path)


# -------------------------------- #
# ---- CELLPROFILER PROFILES ----- #
# -------------------------------- #

def process_profiles(profile_paths: list, row_data: list):
    """
    Downloads image-based profiles from s3 bucket.
    :param profile_paths: full paths of IBP files in s3
    :param row_data: row data from original jump dataset
    :param parq_loc: location to store/ load parquet data
    """
    start = time.perf_counter()

    # Set up a multiprocessing pool to download 10 images at once.
    pool = multiprocessing.Pool(10, initialize)

    # Create list of jobs.
    jobs = []
    for path, row in zip(profile_paths, row_data):
        jobs.append(ProfileParams(path, row))

    # Run the jobs.
    results = pool.map(extract_profiles, jobs)

    end = time.perf_counter()
    print()
    print(f"Processed %s image-based profiles (took {end - start:0.2f} seconds)." % len(results))

    return results


def extract_profiles(params: ProfileParams):
    """
    Reads an IBP from s3 bucket path, stores as a pandas DataFrame.
    :param params: parallel job parameters
    """
    # Return a dataframe containing the well-specific data:
    dframe = load_parquet(params)

    print(".", flush=True, end='')
    return dframe


def load_parquet(params):
    """
    Downloads parquet files of IBP data from s3 if they don't already exist locally.
    Loads the dataframe into memory.

    :param params: parallel job parameters
    :return -> parquet data as dataframe
    """
    # Information to retain from original dataset:
    retain_info = ['moa', 'target', 'smiles', 'clinical_phase', 'moa_src',
                   'Metadata_JCP2022', 'Metadata_InChIKey', 'Metadata_PlateType']

    # Get s3 bucket and key from provided URL.
    url = urlparse(params.profile_path)
    bucket = url.netloc
    key = url.path.lstrip('/')

    PARQ_CACHE_DIR = '../../../DERI-DEE-JumpCP/ibp/parquet/'

    # Ensure the illum func exists locally. Download it if not.
    # Parse s3 path to extract plate name and file name.
    plate_name = key.split('/')[-2]
    file_name = key.split('/')[-1]

    cache_subdir = os.path.join(PARQ_CACHE_DIR, plate_name)
    local_file_path = os.path.join(cache_subdir, file_name)

    if os.path.isfile(local_file_path):
        print("Loading cached parquet file: %s" % file_name)
    else:
        print("Downloading parquet file: %s" % file_name)
        os.makedirs(cache_subdir, exist_ok=True)
        s3 = boto3.resource('s3', region_name=REGION, config=CONFIG)
        try:
            with lock:
                s3.Bucket(bucket).download_file(key, local_file_path)
        except Exception as e:
            raise Exception("Failed to download parquet file %s/%s" % (bucket, key)) from e

    # Loading dataframe:
    dframe = pd.read_parquet(local_file_path)

    # Force to be string. Needed for merge
    dframe['Metadata_Plate'] = dframe['Metadata_Plate'].astype(str)
    # Adding in batch column:
    dframe['Metadata_Batch'] = params.row.Metadata_Batch

    print("Extracting profile data for well: ", params.row.Metadata_Well)
    well_data = dframe[dframe['Metadata_Well'] == params.row.Metadata_Well].reset_index(drop=True)

    # Retain the original data and concat. it with the data downloaded from S3:
    row_data = params.row[retain_info]
    merged_data = pd.concat([well_data, row_data.to_frame().T.reset_index(drop=True)], axis=1)

    return merged_data









