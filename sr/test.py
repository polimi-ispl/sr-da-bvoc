"""
Giganti, A.; Mandelli, S.; Bestagini, P.; Tubaro, S.
Learn from Simulations, Adapt to Observations: Super-Resolution of Isoprene Emissions via Unpaired Domain Adaptation.
Remote Sens. 2024, 16, 3963. https://doi.org/10.3390/rs16213963

© 2024 Antonio Giganti - Image and Sound Processing Lab (ISPL) - Politecnico di Milano, Italy.
"""

from tqdm import tqdm
import numpy as np
import torch
import os
import json
import argparse
import pickle
from pickle import load
from datetime import datetime
from torch.utils.data import DataLoader
from sr.dataset import BVOCDatasetSR
from utils import BASE_OUTPUT_DIR_PATH_SR, BASE_DATASET_PATH, set_backend, set_seed, compute_metrics, shorten_datetime
from sr.models.SAN.san import SAN

set_backend()
set_seed(10)

########################################
# Params ArgumentParser()              #
########################################
# RUN PATH EXAMPLE --> runs/sr/<train_dataset_folder>/x<upscaling_factor>/<model_name>/<timestamp>_<flag>/results

parser = argparse.ArgumentParser(description='BVOC Super-Resolution Network Testing')
# Data specs
parser.add_argument('--train_qt_flag', type=str, default='perc100', help='Flag for the Quantile Transformer adopted in training')

# Train run specs
parser.add_argument('--hr_patch_size', type=int, default=32, help='Size of high-resolution patches')
parser.add_argument('--train_upscaling_factor', type=int, default=2, help='Upscaling factor used in training')
parser.add_argument('--train_dataset_folder', type=str, default='', help='Dataset used for training')
parser.add_argument('--model_name', type=str, default='SAN', help='Name of the model')
parser.add_argument('--flag', type=str, default='', help='Flag of the train run')
parser.add_argument('--train_run_timestamp', type=str, default='',
                    help='Super Resolution training run timestamp')

# Test specs
parser.add_argument('--batch_size', type=int, default=4000, help='Batch size')
parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader')
parser.add_argument('--test_upscaling_factor', type=int, default=2, help='Upscaling factor for testing')
parser.add_argument('--test_dataset_folder', type=str, default='', help='Dataset that will be used for training. Usually the same as the training one')

args = parser.parse_args()
print(args)

########################################
# Paths and Folders                    #
########################################
start_time = datetime.now()
run_name = f'{args.train_run_timestamp}_{args.flag}'
model_folder = os.path.join(BASE_OUTPUT_DIR_PATH_SR, args.train_dataset_folder, f'x{args.train_upscaling_factor}', args.model_name, run_name)
model_pth = os.path.join(model_folder, f'{args.model_name}_best.pth')
current_output_dir_path = os.path.join(model_folder, 'results', f'x{args.test_upscaling_factor}_{args.test_dataset_folder.split("_")[0]}', shorten_datetime(start_time))  # divided by dataset
map_dir_path = os.path.join(current_output_dir_path, 'maps')
os.makedirs(current_output_dir_path, exist_ok=True)
print(f'==> Output dir: {current_output_dir_path}\nRun name: {run_name} <==')

########################################
# Variables Definition                 #
########################################
# GPU
torch.cuda.set_device(args.gpu_id)
print("Using GPU: " + str(torch.cuda.current_device()))

# Networks
network = []
if args.model_name == 'SAN':
    network = SAN(scale=args.test_upscaling_factor)
else:
    print('Network not implemented !!!')

network.cuda()
network.load_state_dict(torch.load(model_pth, map_location=torch.device('cuda')))

# Datasets and DataLoaders
dataset_kwargs = {'root': os.path.join(BASE_DATASET_PATH, args.test_dataset_folder),
                  'downscale_factor': args.test_upscaling_factor,
                  'quantile_transform': True,
                  'downscaling_mode': 'bicubic',
                  }

dataloader_kwargs = {'batch_size': args.batch_size,
                     'num_workers': args.num_workers,
                     'pin_memory': True,
                     }

test_dataloader = DataLoader(BVOCDatasetSR(**dataset_kwargs, dataset_mode='test'), **dataloader_kwargs)

print(f'Test batches: {len(test_dataloader)}')

# Quantile Transformer
# Here a QT from LR is used. We suppose to only have LR data to be superresolved
qt_path = os.path.join(BASE_DATASET_PATH, args.test_dataset_folder, f'quantile_transformer_LR_1e3qua_fullsub_{args.train_qt_flag}.pkl')
qt = load(open(qt_path, 'rb'))
print(f"[QT] Loaded quantile transformer {qt_path.split('/')[-1]} for test")

########################################
# Testing                              #
########################################
start_time = datetime.now()
metrics = ['SSIM', 'PSNR', 'MSE', 'NMSE', 'MAE', 'MaxAE', 'ERGAS', 'UIQ', 'SCC', 'SRE']
metrics_complete_1 = {key: [] for key in metrics}
metrics_complete_2 = {key: [] for key in metrics}
print(f'\n\n{start_time}\tTest starts !')
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Batch', unit='batch'):
        # Extract filenames
        filelist = batch['filename']
        filenames = [f.split('/')[-1] for f in filelist]

        hr_img = batch['HR'].cuda()
        lr_img = batch['LR'].cuda()
        original_output = network(lr_img)

        # Compute metrics (mean over a single batch)
        metrics_1 = compute_metrics(original_output, hr_img, metrics)  # QT domain
        original_output = original_output.cpu().detach().numpy()
        original_hr = batch['original_HR'].cuda()
        original_output = qt.inverse_transform(original_output.reshape(-1, 1)).reshape(original_output.shape)
        original_output = torch.from_numpy(original_output).cuda()
        metrics_2 = compute_metrics(original_output, original_hr, metrics)  # BVOC domain

        # Append results obtained for each batch
        for key in metrics:
            metrics_complete_1[key].append(metrics_1[key])
            metrics_complete_2[key].append(metrics_2[key])

# Compute mean and std dev of the result dictionaries
results = {['mean', 'std'][i]: {} for i in range(2)}

# Flatten the list (batch) of lists (elements of the batch)
for key in metrics:
    metrics_complete_1[key] = [item for sublist in metrics_complete_1[key] for item in sublist]
    metrics_complete_2[key] = [item for sublist in metrics_complete_2[key] for item in sublist]

# Compute the mean and the std deviation of the metrics
for key in metrics:
    mean_1 = round(float(np.mean(metrics_complete_1[key])), 3)  # Metric 1
    mean_2 = round(float(np.mean(metrics_complete_2[key])), 3)  # Metric 2
    std_1 = round(float(np.std(metrics_complete_1[key])), 3)  # Metric 1
    std_2 = round(float(np.std(metrics_complete_2[key])), 3)  # Metric 2
    results['mean'][key] = [mean_1, mean_2]
    results['std'][key] = [std_1, std_2]

# Save the results
results['metrics_1'] = metrics_complete_1
results['metrics_2'] = metrics_complete_2

# Print the mean and the std deviation of all the metrics (type and position)
for key in metrics:
    print(f'{key}:\t{results["mean"][key]}\t{results["std"][key]}')

# Save the results
with open(os.path.join(current_output_dir_path, "results.pkl"), 'wb') as file:
    pickle.dump(results, file)
# Save the arguments
with open(os.path.join(current_output_dir_path, "args.json"), 'w') as file:
    json.dump(vars(args), file)

end_time = datetime.now()
print(f'{end_time}\tTest ends !\nElapsed time: {end_time - start_time}')
print(f'==> Models in {model_folder} <==')
print(f'==> Output dir: {current_output_dir_path}\nRun name: {run_name} <==')
