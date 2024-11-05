"""
Giganti, A.; Mandelli, S.; Bestagini, P.; Tubaro, S.
Learn from Simulations, Adapt to Observations: Super-Resolution of Isoprene Emissions via Unpaired Domain Adaptation.
Remote Sens. 2024, 16, 3963. https://doi.org/10.3390/rs16213963

© 2024 Antonio Giganti - Image and Sound Processing Lab (ISPL) - Politecnico di Milano, Italy.
"""

import random
import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure, peak_signal_noise_ratio, error_relative_global_dimensionless_synthesis, universal_image_quality_index
from torchmetrics.functional.regression import mean_squared_error, mean_absolute_error
from sr.metrics.sre import signal_to_reconstruction_error
from sr.metrics.scc import spatial_correlation_coefficient

########################################
# Paths                                #
########################################
BASE_ROOT_PATH = '/nas/home/agiganti/sr-da-bvoc/'
BASE_DATASET_PATH = '/nas/home/agiganti/green_theme/Datasets/JSTAR/'
BASE_OUTPUT_DIR_PATH_DA = '/nas/home/agiganti/sr-da-bvoc/runs/da/'
BASE_OUTPUT_DIR_PATH_SR = '/nas/home/agiganti/sr-da-bvoc/runs/sr/'
BASE_OUTPUT_DIR_PATH_END2END = '/nas/home/agiganti/sr-da-bvoc/runs/end2end/'

########################################
# Training Utils                       #
########################################


def set_backend():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gradient_norm(model):
    # Filter out parameters with None gradients
    parameters = [p for p in model.parameters() if p.grad is not None]
    # Compute the norm of gradients for each parameter and sum them
    gradient_norm = sum(p.grad.data.norm(2).item() for p in parameters)
    return gradient_norm


def check_gradients(model):
    problematic_params = []
    for name, param in model.named_parameters():
        if param.grad is not None and (torch.isnan(param.grad) | torch.isinf(param.grad)).any():
            problematic_params.append(name)
    if problematic_params:
        problematic_params_str = ', '.join(problematic_params)
        return True, problematic_params_str  # Gradient contains NaN or Inf values
    else:
        return False, ''  # Gradients are okay


def shorten_datetime(datetime_obj):
    # Extract date and time components
    date_part = datetime_obj.strftime("%Y%m%d")
    time_part = datetime_obj.strftime("%H%M%S")
    # Combine date and time components with hyphen
    formatted_datetime = f"{date_part}-{time_part}"
    return formatted_datetime


########################################
# Evaluation Metrics                   #
########################################


def tensor2img(tensor, max_val=1):
    tensor = torch.clamp(tensor, 0, max_val)  # Ensure values are between 0 and max_val
    tensor = (tensor * 255.0).round()
    return tensor


def compute_metrics(output, original, metrics, ten2img=False, mean=False):
    results = {}
    output[output < 0.0] = 0.0  # Clip negative SR values to 0

    if ten2img:
        # Need to convert to uint8 for SSIM
        original = tensor2img(original)
        output = tensor2img(output)

    # Compute metrics for each single image in the batch
    if 'SSIM' in metrics:
        a = [structural_similarity_index_measure(torch.unsqueeze(output[i], dim=0), torch.unsqueeze(original[i], dim=0)).item() for i in range(output.size(0))]
        results['SSIM'] = a
    if 'PSNR' in metrics:
        a = [peak_signal_noise_ratio(output[i], original[i]).item() for i in range(output.size(0))]
        results['PSNR'] = a
    if 'MSE' in metrics:
        a = [mean_squared_error(output[i], original[i]).item() for i in range(output.size(0))]
        results['MSE'] = a
    if 'NMSE' in metrics:
        # Log NMSE, dB
        a = [(10 * torch.log10(mean_squared_error(output[i], original[i]) / torch.mean(original[i] ** 2))).item() for i in range(output.size(0))]
        results['NMSE'] = a
    if 'MAE' in metrics:
        a = [mean_absolute_error(output[i], original[i]).item() for i in range(output.size(0))]
        results['MAE'] = a
    if 'MaxAE' in metrics:
        a = [torch.max(torch.abs(output[i] - original[i])).item() for i in range(output.size(0))]
        results['MaxAE'] = a
    if 'ERGAS' in metrics:
        a = [error_relative_global_dimensionless_synthesis(torch.unsqueeze(output[i], dim=0), torch.unsqueeze(original[i], dim=0)).item() for i in range(output.size(0))]
        results['ERGAS'] = a
    if 'UIQ' in metrics:
        a = [universal_image_quality_index(torch.unsqueeze(output[i], dim=0), torch.unsqueeze(original[i], dim=0)).item() for i in range(output.size(0))]
        results['UIQ'] = a
    if 'SCC' in metrics:
        a = [spatial_correlation_coefficient(output[i], original[i]).item() for i in range(output.size(0))]
        results['SCC'] = a
    if 'SRE' in metrics:
        # Log SRE, dB
        a = [signal_to_reconstruction_error(torch.unsqueeze(output[i], dim=0), torch.unsqueeze(original[i], dim=0)).item() for i in range(output.size(0))]
        results['SRE'] = a

    if mean:
        for key in results.keys():
            results[key] = np.mean(results[key])

    return results
