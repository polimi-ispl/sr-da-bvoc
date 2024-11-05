"""
Giganti, A.; Mandelli, S.; Bestagini, P.; Tubaro, S.
Learn from Simulations, Adapt to Observations: Super-Resolution of Isoprene Emissions via Unpaired Domain Adaptation.
Remote Sens. 2024, 16, 3963. https://doi.org/10.3390/rs16213963

© 2024 Antonio Giganti - Image and Sound Processing Lab (ISPL) - Politecnico di Milano, Italy.
"""

import glob
import random
import os
import numpy as np
from pickle import load
from torch.utils.data import Dataset
from da.data_utils import compute_quantile_transformation


class BVOCDatasetSR(Dataset):
    def __init__(self, root, downscale_factor: int, dataset_mode='train', quantile_transform=False,
                 downscaling_mode='bicubic', percentage=100, seed=10):
        self.downscale_factor = downscale_factor
        self.dataset_mode = dataset_mode  # train | val | test
        self.quantile_transform = quantile_transform
        self.percentage = percentage

        print(f"\nDataset mode: {self.dataset_mode}")

        # ———————— File lists ————————
        hr_filepaths = sorted(glob.glob(os.path.join(root, f'HR/{self.dataset_mode}/*.npy')))
        lr_filepaths = sorted(
            glob.glob(os.path.join(root, f'LR/{self.dataset_mode}/x{self.downscale_factor}/{downscaling_mode}/*.npy')))

        old_hr_filepaths = hr_filepaths
        old_lr_filepaths = lr_filepaths

        # ———————— Data filtering ————————
        # Select a subset of A patches based on a percentage
        if self.percentage < 100 and 'test' not in self.dataset_mode:
            # We reduce the size of the source domain (A) only
            subset_size = int(len(hr_filepaths) * (self.percentage / 100))
            random.seed(seed)
            hr_filepaths = random.sample(hr_filepaths, subset_size)
            random.seed(seed)  # to fetch the same indices
            lr_filepaths = random.sample(lr_filepaths, subset_size)
            print(f"Selected {self.percentage}% of the source domain (A) patches"
                  f"\nTotal patches: {len(hr_filepaths)}/{len(old_hr_filepaths)}")

        self.files = list(zip(hr_filepaths, lr_filepaths))

        # ———————— Quantile transformation ————————
        qt_flag_name = f'_perc{self.percentage}'
        if self.quantile_transform and self.dataset_mode == 'train':
            # Compute quantile transformation from scratch and save it
            self.qt, _ = compute_quantile_transformation(hr_filepaths, lr_filepaths, flag_name=qt_flag_name)  # use the HR one
            print(f"[QT] Computed quantile transformer for training — from HR maps")
        elif self.quantile_transform and self.dataset_mode == 'val':
            # or load the precomputed quantile transformers, the one used for training
            qt_path = os.path.join(root, f'quantile_transformer_HR_1e3qua_fullsub{qt_flag_name}.pkl')
            self.qt = load(open(qt_path, 'rb'))
            print(f"[QT] Loaded quantile transformer {qt_path.split('/')[-1]} for validation")
        elif self.quantile_transform and self.dataset_mode == 'test':
            # load a user defined quantile transformer for the test set. We use the HR one, but you can also use the LR one.
            qt_path = os.path.join(root, f'quantile_transformer_HR_1e3qua_fullsub.pkl')
            self.qt = load(open(qt_path, 'rb'))
            print(f"[QT] Loaded quantile transformer {qt_path.split('/')[-1]} for test")

    def __getitem__(self, index):
        # Paths
        hr_path, lr_path = self.files[index]
        # Filename
        filename = os.path.basename(hr_path).split('.')[0]
        # Load
        hr_img = np.load(hr_path)
        lr_img = np.load(lr_path)
        # Expand dims (1-channel)
        hr_img = np.expand_dims(hr_img, axis=0)
        lr_img = np.expand_dims(lr_img, axis=0)

        lr_img[lr_img < 0.0] = 0.0  # negative emissions are meaningless

        # Quantile transform
        if self.quantile_transform:
            hr_img_n = self.qt.transform(hr_img.reshape(-1, 1)).reshape(hr_img.shape)
            lr_img_n = self.qt.transform(lr_img.reshape(-1, 1)).reshape(lr_img.shape)
        else:  # No quantile transform
            hr_img_n = hr_img
            lr_img_n = lr_img

        data = {'HR': hr_img_n.astype(np.float32),
                'LR': lr_img_n.astype(np.float32),
                'filename': filename}
        if self.dataset_mode == 'test':
            data['original_HR'] = hr_img.astype(np.float32)
            data['original_LR'] = lr_img.astype(np.float32)

        return data

    def __len__(self):
        return len(self.files)
