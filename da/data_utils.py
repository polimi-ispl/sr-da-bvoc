"""
Giganti, A.; Mandelli, S.; Bestagini, P.; Tubaro, S.
Learn from Simulations, Adapt to Observations: Super-Resolution of Isoprene Emissions via Unpaired Domain Adaptation.
Remote Sens. 2024, 16, 3963. https://doi.org/10.3390/rs16213963

© 2024 Antonio Giganti - Image and Sound Processing Lab (ISPL) - Politecnico di Milano, Italy.
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer
from pickle import dump
from time import time
from joblib import parallel_backend
from multiprocessing import Pool


########################################
# General Utils                        #
########################################


def load_and_process(file_tuple):
    """
    Load and process the HR and LR arrays from the given file tuple.
    """
    f_name, train_folder_hr, train_folder_lr = file_tuple
    HR_file_path = os.path.join(train_folder_hr, f_name)
    LR_file_path = os.path.join(train_folder_lr, f_name)
    HR_arr = np.load(HR_file_path)
    LR_arr = np.load(LR_file_path)
    return HR_arr, LR_arr  # , HR_arr.max(), LR_arr.max()


def compute_quantile_transformation(filelist_path_hr, filelist_path_lr, n_quantiles = 1000, flag_name=''):
    """
    Compute the quantile-based data transformation for the given HR and LR filelists.
    """
    # Set the max number of spawned threads
    os.environ['OMP_NUM_THREADS'] = str(4)
    os.environ['OPENBLAS_NUM_THREADS'] = str(4)  # if PIP was used

    star_time = time()
    DATA_PATH = filelist_path_hr[0].split("HR")[0]
    print(f'[QT] Data Path: \t{DATA_PATH}')

    # Extract filenames and parent directories from the filelist paths
    file_names = [f.split('/')[-1] for f in filelist_path_hr]
    train_folder_hr = filelist_path_hr[0].split(file_names[0])[0]
    train_folder_lr = filelist_path_lr[0].split(file_names[0])[0]

    # Use multiprocessing to parallelize loading and processing
    print(f'[QT] Loading data ...')
    with Pool() as pool:
        results = list(tqdm(pool.imap(load_and_process, [(f_name, train_folder_hr, train_folder_lr) for f_name in file_names]), total=len(file_names)))

    # Process results
    stacked_HR_arrays = []
    stacked_LR_arrays = []
    for HR_arr, LR_arr in results:
        stacked_HR_arrays.append(HR_arr)
        stacked_LR_arrays.append(LR_arr)

    # Stack arrays
    stacked_HR_arrays = np.dstack(stacked_HR_arrays)
    stacked_LR_arrays = np.dstack(stacked_LR_arrays)

    print(f'[QT] Start quantile transformations calculation ...')

    # Reshaping
    shape_HR = stacked_HR_arrays.shape
    shape_LR = stacked_LR_arrays.shape
    num_examples_HR = int(shape_HR[0] * shape_HR[1] * shape_HR[2])
    num_examples_LR = int(shape_LR[0] * shape_LR[1] * shape_LR[2])

    HR_data_flatten = stacked_HR_arrays.reshape(-1, 1)  # (num_examples_unrolled, 1)
    LR_data_flatten = stacked_LR_arrays.reshape(-1, 1)  # (num_examples_unrolled, 1)

    # Multiple test on transformer params HR
    assert num_examples_HR > n_quantiles, f'The "subsample" field is set to {num_examples_HR} and has to be > than "n_quantiles" field {n_quantiles}'
    with parallel_backend('threading', n_jobs=8):
        # -------- transformer HR
        quantile_transformer_HR_1e3qua_fullsub = QuantileTransformer(output_distribution='uniform', n_quantiles=n_quantiles,
                                                                     subsample=num_examples_HR,
                                                                     random_state=10).fit(HR_data_flatten)
        dump(quantile_transformer_HR_1e3qua_fullsub,
             open(os.path.join(DATA_PATH, f'quantile_transformer_HR_1e3qua_fullsub{flag_name}.pkl'), 'wb'))
        print(f'HR Done <--')

        # -------- transformer LR
        quantile_transformer_LR_1e3qua_fullsub = QuantileTransformer(output_distribution='uniform', n_quantiles=n_quantiles,
                                                                     subsample=num_examples_LR,
                                                                     random_state=10).fit(LR_data_flatten)
        dump(quantile_transformer_LR_1e3qua_fullsub,
             open(os.path.join(DATA_PATH, f'quantile_transformer_LR_1e3qua_fullsub{flag_name}.pkl'), 'wb'))
        print(f'LR Done <--')

    end_time = time()
    # print(f'[QT] Duration {(end_time - star_time) / 60} min')

    return quantile_transformer_HR_1e3qua_fullsub, quantile_transformer_LR_1e3qua_fullsub


def calculate_patches_and_leftovers(H, W, P_h, P_w, S_h, S_w):
    """
    Calculate the number of patches that fit in an image and the leftover pixels in both dimensions.

    Parameters:
    H (int): Height of the image.
    W (int): Width of the image.
    P_h (int): Height of each patch.
    P_w (int): Width of each patch.
    S_h (int): Stride in the vertical direction (height).
    S_w (int): Stride in the horizontal direction (width).

    Returns:
    tuple: A tuple containing:
        - num_patches_h (int): Number of full patches that fit within the image height.
        - covered_height (int): Total height covered by the patches.
        - leftover_height (int): Number of pixels left after placing the full patches in the height dimension.
        - num_patches_w (int): Number of full patches that fit within the image width.
        - covered_width (int): Total width covered by the patches.
        - leftover_width (int): Number of pixels left after placing the full patches in the width dimension.
    """
    # Number of patches that can fit in height and width
    num_patches_h = (H - P_h) // S_h + 1
    num_patches_w = (W - P_w) // S_w + 1

    # Total length covered by the patches in height and width
    covered_height = (num_patches_h - 1) * S_h + P_h
    covered_width = (num_patches_w - 1) * S_w + P_w

    # Number of leftover pixels in height and width
    leftover_height = H - covered_height
    leftover_width = W - covered_width

    print(f'num_patches_h: {num_patches_h}, num_patches_w: {num_patches_w}, covered_h: {covered_height}, covered_w: {covered_width}, leftover_h: {leftover_height}, leftover_w: {leftover_width}\n')

    return num_patches_h, covered_height, leftover_height, num_patches_w, covered_width, leftover_width
