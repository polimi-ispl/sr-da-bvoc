"""
Giganti, A.; Mandelli, S.; Bestagini, P.; Tubaro, S.
Learn from Simulations, Adapt to Observations: Super-Resolution of Isoprene Emissions via Unpaired Domain Adaptation.
Remote Sens. 2024, 16, 3963. https://doi.org/10.3390/rs16213963

© 2024 Antonio Giganti - Image and Sound Processing Lab (ISPL) - Politecnico di Milano, Italy.
"""

import random
import torch
from skimage.exposure import match_histograms
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np


########################################
# General Utils                        #
########################################


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


########################################
# Histogram Matching Functions         #
########################################


def hist_match_std(source_batch, reference_batch):
    """
    Match histograms of a batch of source images to a batch of reference images.

    Parameters:
    source_batch (torch.Tensor): A batch of source images with shape (batch_size, channels, height, width).
    reference_batch (torch.Tensor): A batch of reference images with shape (batch_size, channels, height, width).

    Returns:
    torch.Tensor: Batch of images after histogram matching with shape (batch_size, channels, height, width).
    """
    # Convert tensors to numpy arrays
    source_batch_np = source_batch.cpu().numpy()
    reference_batch_np = reference_batch.cpu().numpy()
    # Initialize an array for the matched images
    matched_batch_np = np.empty_like(source_batch_np)
    batch_dim = source_batch_np.shape[0]
    # Apply histogram matching for each image in the batch, with tqdm for progress bar
    for i in tqdm(range(batch_dim), desc='Matching histograms...'):
        matched_image = match_histograms(source_batch_np[i], reference_batch_np[i], multichannel=False)
        matched_batch_np[i] = matched_image
    # Convert the result back to a PyTorch tensor
    matched_batch = torch.from_numpy(matched_batch_np).float().cuda()

    return matched_batch

########################################
# Emission Consistency Loss            #
########################################


def emission_consistency_loss(sr_output, lr_original, overlapping_factor, patch_size_lr, upscaling_factor):
    '''
    Compute the consistency loss between two tensors, ensuring that the mean of values within corresponding patches in
    the original low-resolution tensor "lr_original" is approx. equal to the mean of values within
    corresponding patches in the super-resolved tensor "sr_output".
    The consistency loss is calculated by comparing patches of the same spatial area in both tensors. The size of these
    patches is determined by the parameter "patch_size", which applies to both dimensions of the low-resolution tensor.
    This patch size is then upscaled by the specified upscaling factor to determine the size of patches in the
    super-resolved tensor.
    The number of patches used for comparison is fixed and computed based on the dimensions of the low-resolution
    tensor "lr_original". The overlapping factor, defined by the parameter "overlapping_factor", determines the extent
    of overlap between adjacent patches in both dimensions of the low-resolution tensor, thus the stride.
    For example, if upscaling from a 3x3 to a 6x6 tensor (thus using a 2x upscaling factor) with a patch size of 2x2
    (on the low-resolution tensor) and an overlapping factor of 1, resulting in a 4x4 patch size on the super-resolved
    tensor, there will be 4 patches in both the lr_original and sr_output tensors. The function ensures that the mean of
    values within each 4x4 patch of the super-resolved tensor is approx. equal to the mean of values
    within the corresponding 2x2 patch of the low-resolution tensor.
    :param sr_output: The super-resolved tensor.
    :param lr_original: The original low-resolution tensor.
    :param overlapping_factor: The factor determining the extent of overlap between adjacent patches.
    :param patch_size_lr: The size of patches (observation window) in the low-resolution tensor.
    :param upscaling_factor: The factor by which the patch size is upscaled to match the super-resolved tensor.
    :return: The consistency loss.
    '''
    # Compute the patch size on the super-resolved tensor
    patch_size_sr = patch_size_lr * upscaling_factor
    # Compute the stride of the overlapping patches
    stride = patch_size_lr - overlapping_factor
    # Compute the number of patches in the x and y dimensions
    num_patches_x = (lr_original.shape[-1] - patch_size_lr + 1) // stride
    num_patches_y = (lr_original.shape[-2] - patch_size_lr + 1) // stride
    # Loss initialization
    consistency_loss = 0

    for i in range(num_patches_x):
        for j in range(num_patches_y):
            # Extract the subpatches from the original tensor
            subpatch_original = lr_original[:, :, i * stride:i * stride + patch_size_lr,
                             j * stride:j * stride + patch_size_lr]
            # Calculate corresponding indices for the super-resolved tensor
            sr_i_start = i * stride * upscaling_factor
            sr_j_start = j * stride * upscaling_factor
            # Extract the patch from the super-resolved tensor using the calculated indices
            subpatch_output = sr_output[:, :, sr_i_start:sr_i_start + patch_size_sr, sr_j_start:sr_j_start + patch_size_sr]
            # Compute the consistency loss for the current patch.
            # The loss is the absolute difference between the mean values of the patches
            consistency_loss += torch.abs(torch.mean(subpatch_original) - torch.mean(subpatch_output))

    # Normalize the consistency loss by the number of patches
    consistency_loss /= num_patches_x * num_patches_y
    return consistency_loss
