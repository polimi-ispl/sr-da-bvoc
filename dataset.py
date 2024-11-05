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
from torch.utils.data import Dataset
from pickle import load
from da.data_utils import compute_quantile_transformation


class BVOCDataset_end2end(Dataset):
    """
    Dataset class for end-to-end training.
    - Domain Adaptation needs: [QT_A_HR_A, QT_B_HR_B], A and B domains
    - Super-Resolution needs: [QT_B_HR_B, QT_B_LR_B], B-domain only
    """

    def __init__(self, root_source_domain_dataset, root_target_domain_dataset, downscale_factor: int, unaligned=True,
                 dataset_mode='train', quantile_transform=True, downscaling_mode='bicubic', percentage=100, seed=10):
        self.downscale_factor = downscale_factor
        self.unaligned = unaligned
        self.dataset_mode = dataset_mode  # train | val | test
        self.quantile_transform = quantile_transform
        self.percentage = percentage

        # ———————— File lists ————————
        # Load the files from the source domain OMI/GOMI (A) and target domain CAMS-GLOB-BIO (B)
        self.files_A_hr = sorted(
            glob.glob(os.path.join(root_source_domain_dataset, 'HR', self.dataset_mode) + '/*.npy'))  # OMI/GOMI (A)
        self.files_B_hr = sorted(
            glob.glob(os.path.join(root_target_domain_dataset, 'HR', self.dataset_mode) + '/*.npy'))  # CAMS-GLOB-BIO (B)

        self.files_A_lr = sorted(glob.glob(
            os.path.join(root_source_domain_dataset, 'LR', self.dataset_mode, f'x{self.downscale_factor}',
                         downscaling_mode) + '/*.npy'))
        self.files_B_lr = sorted(glob.glob(
            os.path.join(root_target_domain_dataset, 'LR', self.dataset_mode, f'x{self.downscale_factor}',
                         downscaling_mode) + '/*.npy'))

        old_files_A_hr = self.files_A_hr

        # ———————— Data filtering (only for A domain) ————————
        # Select a subset of A patches based on a percentage
        if self.percentage < 100:
            # We reduce the size of the source domain (A) only
            subset_size_A = int(len(self.files_A_hr) * (self.percentage / 100))
            random.seed(seed)
            self.files_A_hr = random.sample(self.files_A_hr, subset_size_A)
            random.seed(seed)  # to fetch the same indices
            self.files_A_lr = random.sample(self.files_A_lr, subset_size_A)
            print(f"Selected {self.percentage}% of the source domain (A) patches"
                  f"\nTotal patches: {len(self.files_A_hr)}/{len(old_files_A_hr)}")

        # ———————— Quantile transformation ————————
        qt_flag_name = f'_perc{self.percentage}'
        if self.quantile_transform and self.dataset_mode == 'train':
            # Compute quantile transformation from scratch
            self.qt_A, _ = compute_quantile_transformation(self.files_A_hr, self.files_A_lr, flag_name=qt_flag_name)
            self.qt_B, _ = compute_quantile_transformation(self.files_B_hr, self.files_B_lr, flag_name='_perc100_all')  # Always 100% for the target domain
            print(f"[QT] Computed quantile transformers in for training — from HR maps")
        elif self.quantile_transform and self.dataset_mode == 'val':
            # or load the precomputed quantile transformers
            qt_path_A = os.path.join(root_source_domain_dataset, f'quantile_transformer_HR_1e3qua_fullsub{qt_flag_name}.pkl')
            qt_path_B = os.path.join(root_target_domain_dataset, f'quantile_transformer_HR_1e3qua_fullsub_perc100_all.pkl')
            self.qt_A = load(open(qt_path_A, 'rb'))
            self.qt_B = load(open(qt_path_B, 'rb'))
            print(f"[QT] Loaded quantile transformers in:\nA:{qt_path_A}\nB:{qt_path_B}\nfor validation")
        elif self.quantile_transform and self.dataset_mode == 'test':
            print("[QT] PAY ATTENTION !!!\nNo quantile transformation for test mode. The end-to-end test uses an instance of BVOCDatasetSR class")

    def __getitem__(self, index):
        idx_A = index % len(self.files_A_hr)
        item_A_hr = np.load(self.files_A_hr[idx_A])
        item_A_lr = np.load(self.files_A_lr[idx_A])
        if self.unaligned:
            idx_B = random.randint(0, len(self.files_B_hr) - 1)
            item_B_hr = np.load(self.files_B_hr[idx_B])
            item_B_lr = np.load(self.files_B_lr[idx_B])
        else:
            item_B_hr = np.load(self.files_B_hr[index % len(self.files_B_hr)])
            item_B_lr = np.load(self.files_B_lr[index % len(self.files_B_lr)])

        # Expand dims (1-channel)
        item_A_hr = np.expand_dims(item_A_hr, axis=0)
        item_B_hr = np.expand_dims(item_B_hr, axis=0)
        item_A_lr = np.expand_dims(item_A_lr, axis=0)
        item_B_lr = np.expand_dims(item_B_lr, axis=0)

        item_A_lr[item_A_lr < 0.0] = 0.0  # negative emissions are meaningless
        item_B_lr[item_B_lr < 0.0] = 0.0  # negative emissions are meaningless

        # Quantile transform
        if self.quantile_transform:
            item_A_n_hr = self.qt_A.transform(item_A_hr.reshape(-1, 1)).reshape(item_A_hr.shape)
            item_B_n_hr = self.qt_B.transform(item_B_hr.reshape(-1, 1)).reshape(item_B_hr.shape)
            item_A_n_lr = self.qt_A.transform(item_A_lr.reshape(-1, 1)).reshape(item_A_lr.shape)
            item_B_n_lr = self.qt_B.transform(item_B_lr.reshape(-1, 1)).reshape(item_B_lr.shape)

        data = {'A_hr': item_A_n_hr.astype(np.float32),
                'B_hr': item_B_n_hr.astype(np.float32),
                'A_lr': item_A_n_lr.astype(np.float32),
                'B_lr': item_B_n_lr.astype(np.float32),
                'A_filename': self.files_A_hr[index % len(self.files_A_hr)],
                'B_filename': self.files_B_hr[index % len(self.files_B_hr)]}
        if self.dataset_mode == 'test':
            data['original_A_hr'] = item_A_hr.astype(np.float32)
            data['original_A_lr'] = item_A_lr.astype(np.float32)

        return data

    def __len__(self):
        # return max(len(self.files_A_hr), len(self.files_B_hr))
        return min(len(self.files_A_hr), len(self.files_B_hr))  # since usually #A<#B, use min to reduce the training time !!
