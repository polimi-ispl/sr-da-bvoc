"""
Giganti, A.; Mandelli, S.; Bestagini, P.; Tubaro, S.
Learn from Simulations, Adapt to Observations: Super-Resolution of Isoprene Emissions via Unpaired Domain Adaptation.
Remote Sens. 2024, 16, 3963. https://doi.org/10.3390/rs16213963

© 2024 Antonio Giganti - Image and Sound Processing Lab (ISPL) - Politecnico di Milano, Italy.
"""

import argparse
import json
import os
from pickle import load
import torch
import pickle
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from da.models import Generator
from da.utils import hist_match_std
from utils import compute_metrics, set_backend, set_seed, BASE_DATASET_PATH, BASE_ROOT_PATH, shorten_datetime
from sr.dataset import BVOCDatasetSR
from sr.models.SAN.san import SAN

set_backend()
set_seed(10)

# A domain --> Observed (OMI/GOME)
# B domain --> Simulated (MEGAN)

########################################
# Params ArgumentParser()              #
########################################
# RUN PATH EXAMPLE --> runs/end2end/<source_domain_dataset>/<target_domain_dataset>/<timestamp>_<flag>

parser = argparse.ArgumentParser(description='BVOC Domain Adaptation + Super-Resolution Networks Testing')

# Data specs
parser.add_argument('--main_dataset', type=str, default='',
                    help='Main dataset name used for inference. Usually, the same dataset of the DA model. It will consider the "test" partition')
parser.add_argument('--train_qt_flag', type=str, default='perc100', help='Flag for the Quantile Transformer adopted in training')

# Train specs
parser.add_argument('--source_domain_dataset', type=str, default='',
                    help='Domain Adaptation source (A) dataset name')
parser.add_argument('--target_domain_dataset', type=str, default='',
                    help='Domain Adaptation target (B) dataset name')
parser.add_argument('--flag', type=str, default='', help='Domain Adaptation run flag')
parser.add_argument('--train_run_timestamp', type=str, default='',
                    help='Domain Adaptation training run timestamp')
parser.add_argument('--ptr_sr', action='store_true',
                    help='Flag to use a pretrained SR model. Please, check the train.')
parser.add_argument('--ptr_sr_run_flag', type=str, default='',
                    help='Run flag of the pretrained model weights of the Super-Resolution nerwork')
parser.add_argument('--sr_dataset', type=str, default='', help='Dataset of the pretrained SR network. '
                                                               'Leave empty to use the same simulated dataset of the DA network')

# Test specs
parser.add_argument('--batch_size', type=int, default=4000, help='Batch size')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader')
parser.add_argument('--sr_test_upscaling_factor', type=int, default=2, help='Super-Resolution test upscaling factor')
parser.add_argument('--avoid_histogram_matching', action='store_true', help='Flag to avoid histogram matching between LR and SR patches from the source domain')
parser.add_argument('--histogram_matching_domain', type=str, default='BVOC', help='Domain for histogram matching. "QT" or "BVOC"')

args = parser.parse_args()
print(args)

########################################
# Paths and Folders                    #
########################################
# RUN PATH EXAMPLE --> runs/end2end/<source_domain_dataset>/<target_domain_dataset>/<timestamp>_<flag>/results

start_time = datetime.now()
run_name = f'{args.train_run_timestamp}_{args.flag}' if args.flag is not '' else f'{args.train_run_timestamp}'
END2END_MODELS_DIR_PATH = os.path.join(BASE_ROOT_PATH, 'runs', 'end2end')
model_folder = os.path.join(END2END_MODELS_DIR_PATH, args.source_domain_dataset, args.target_domain_dataset, run_name)
current_output_dir_path = os.path.join(model_folder, 'results',
                                       f'x{args.sr_test_upscaling_factor}_{args.source_domain_dataset.split("_")[0]}2{args.target_domain_dataset.split("_")[0]}',
                                       shorten_datetime(start_time))  # divided by dataset and timestamp
# Target, OMI/GOME (A) seems like MEGAN (B) <-- THIS IS THE ONE WE WANT
hr_hat_A_dir_path = current_output_dir_path
map_dir_path = os.path.join(current_output_dir_path, 'maps')
os.makedirs(current_output_dir_path, exist_ok=True)  # OMI/GOMI (A) seems MEGAN (B) <-- THIS IS THE ONE WE WANT
print(f'==> Output dir: {current_output_dir_path}\nRun name: {run_name} <==')

print(f'==> End2End Models in {model_folder} <==')

with open(os.path.join(current_output_dir_path, 'info.txt'), "w") as file:
    file.write(f'End2End Models in {model_folder}\n')

########################################
# Variables Definition                 #
########################################
# GPU
torch.cuda.set_device(args.gpu_id)
print("Using GPU: " + str(torch.cuda.current_device()))
# Domain Adaptation Nets
input_nc, output_nc = 1, 1
da_netG_A2B = Generator(input_nc, output_nc)
hr_da_netG_A2B = Generator(input_nc, output_nc)  # only for test metrics 0
da_netG_B2A = Generator(output_nc, input_nc)
da_netG_A2B.cuda()
hr_da_netG_A2B.cuda()
da_netG_B2A.cuda()
da_netG_A2B.load_state_dict(
    torch.load(os.path.join(model_folder, 'netG_A2B_lr_best.pth'), map_location=torch.device('cuda')))
hr_da_netG_A2B.load_state_dict(torch.load(os.path.join(model_folder, 'netG_A2B_hr_best.pth'),
                                          map_location=torch.device('cuda')))  # only for test metrics 0
da_netG_B2A.load_state_dict(
    torch.load(os.path.join(model_folder, 'netG_B2A_hr_best.pth'), map_location=torch.device('cuda')))
# Super-Resolution Net
sr_net_B = SAN()
if args.ptr_sr:
    # Decide whether to use the same simulated dataset of the DA network or a different one, for the SR network
    if args.sr_dataset is not '':  sr_dataset = args.sr_dataset
    else:  sr_dataset = args.target_domain_dataset

    pretrained_sr_model = os.path.join(BASE_ROOT_PATH, 'runs', 'sr', sr_dataset, 'x2', 'SAN', args.ptr_sr_run_flag, 'SAN_best.pth')
    sr_net_B.load_state_dict(torch.load(pretrained_sr_model, map_location=torch.device('cuda')))
    print(f'==> Pretrained SR net from run {args.ptr_sr_run_flag} loaded !')
else:
    sr_net_B.load_state_dict(
        torch.load(os.path.join(model_folder, f'sr_net_B_best.pth'), map_location=torch.device('cuda')))

sr_net_B.cuda()

# Dataset and Dataloader
# we need only the LR data from the source domain (OMI/GOME)
dataset_kwargs = {'root': os.path.join(BASE_DATASET_PATH, args.main_dataset),
                  'downscale_factor': args.sr_test_upscaling_factor,
                  'quantile_transform': True,
                  'qt_folder': os.path.join(BASE_DATASET_PATH, args.main_dataset),
                  'downscaling_mode': 'bicubic',
                  'percentage': 100,  # 100% of the data for testing
                  'train_qt_flag': args.train_qt_flag,
                  }

dataloader_kwargs = {'batch_size': args.batch_size,
                     'num_workers': args.num_workers,
                     'pin_memory': True,
                     }

test_dataloader = DataLoader(BVOCDatasetSR(**dataset_kwargs, dataset_mode='test'), **dataloader_kwargs)

# Quantile Transformer
# Here a QT from LR is used. We suppose to only have LR data to be superresolved
qt_A_path = os.path.join(BASE_DATASET_PATH, args.source_domain_dataset, f'quantile_transformer_LR_1e3qua_fullsub_{args.train_qt_flag}.pkl')
qt_A = load(open(qt_A_path, 'rb'))
print(f"[QT] Loaded quantile transformer {qt_A_path.split('/')[-1]} for test")

########################################
# Inference                            #
########################################
start_time = datetime.now()
metrics = ['SSIM', 'PSNR', 'MSE', 'NMSE', 'MAE', 'MaxAE', 'ERGAS', 'UIQ', 'SCC', 'SRE']
# Initialize the metrics dictionaries
metrics_complete_0 = {key: [] for key in metrics}
metrics_complete_1 = {key: [] for key in metrics}
metrics_complete_2 = {key: [] for key in metrics}

print(f'\n\n{start_time}\tTest starts !')
with torch.no_grad():
    # use tqdm for progress bar
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Batch', unit='batch'):
        # Extract filenames
        filelist = batch['filename']
        filenames = [f.split('/')[-1] for f in filelist]

        # LR_o --> QT_o(LR_o)
        qtA_lr_A = batch['LR'].cuda()

        # DA_o-s(QT_o(LR_o) aka ^QT_s(LR_o), Domain Adaptation, A --> B
        qtB_hat_lr_A = da_netG_A2B(qtA_lr_A).data  # A --> B (OMI/GOMI --> fake MEGAN)
        qtB_hat_lr_A = torch.clamp(qtB_hat_lr_A, 0, 1)

        # ^QT_s(^HR_o), Super-Resolution, B --> B
        qtB_hat_hr_hat_A = sr_net_B(qtB_hat_lr_A).data

        # Clip above 1 and below 0 [Not necessary, but just in case]
        # qtB_hat_hr_hat_A = torch.clamp(qtB_hat_hr_hat_A, 0, 1)

        # DA_s-o(^QT_s(^HR_o), aka ^QT_o(^HR_o), Domain Adaptation, B --> A (fake MEGAN --> OMI/GOME)
        qtA_hat_hr_hat_A = da_netG_B2A(qtB_hat_hr_hat_A).data
        qtA_hat_hr_hat_A = torch.clamp(qtA_hat_hr_hat_A, 0, 1)

        # Histogram Matching on QT domain data <==
        if not args.avoid_histogram_matching and args.histogram_matching_domain == 'QT':
            matched = hist_match_std(qtA_hat_hr_hat_A, qtA_lr_A)  # (input, target)
            qtA_hat_hr_hat_A_old = qtA_hat_hr_hat_A
            qtA_hat_hr_hat_A = matched

        # ^HR_o, Inverse Quantile Transform
        qtA_hat_hr_hat_A = qtA_hat_hr_hat_A.cpu().numpy()
        hr_hat_A = qt_A.inverse_transform(qtA_hat_hr_hat_A.reshape(-1, 1)).reshape(qtA_hat_hr_hat_A.shape).astype(
            np.float32)
        hr_hat_A = torch.from_numpy(hr_hat_A).cuda()

        # Histogram Matching on BVOC domain data <==
        if not args.avoid_histogram_matching and args.histogram_matching_domain == 'BVOC':
            lr_A = batch['original_LR'].cuda()
            matched = hist_match_std(hr_hat_A, lr_A)  # (input, target)
            hr_hat_A_old = hr_hat_A
            hr_hat_A = matched

        ########## Compute metrics ##########
        # numpy --> torch --> cuda
        hr_A = batch['original_HR'].cuda()  # HR_o
        qtA_hat_hr_hat_A = torch.from_numpy(qtA_hat_hr_hat_A).cuda()  # ^QT_o(^HR_o)

        # Metrics 0: DA_o-s(QT_o(HR_o)) - ^QT_s(^HR_o)	    | Domain: Synth qt | Purpose: SRNET I/O
        da_qtA_hr_A = hr_da_netG_A2B(hr_A).data
        metrics_0 = compute_metrics(da_qtA_hr_A, qtB_hat_hr_hat_A, metrics)

        # Metrics 1: QT_o(HR_o) - ^QT_o(^HR_o)				| Domain: Obs qt   | Purpose: SRNET + CycleGAN I/O
        qtA_hr_A = qt_A.transform(hr_A.cpu().numpy().reshape(-1, 1)).reshape(hr_A.cpu().numpy().shape).astype(
            np.float32)
        qtA_hr_A = torch.from_numpy(qtA_hr_A).cuda()
        metrics_1 = compute_metrics(qtA_hat_hr_hat_A, qtA_hr_A, metrics)

        # Metrics 2: HR_o - ^HR_o					        | Domain: Obs      | Purpose: entire pipeline I/O
        metrics_2 = compute_metrics(hr_hat_A, hr_A, metrics)

        # Append results obtained for each batch
        for key in metrics:
            metrics_complete_0[key].append(metrics_0[key])
            metrics_complete_1[key].append(metrics_1[key])
            metrics_complete_2[key].append(metrics_2[key])

###################################
# Compute mean and std dev of the result dictionaries
results = {['mean', 'std'][i]: {} for i in range(2)}

# Flatten the list (batch) of lists (elements of the batch)
for key in metrics:
    metrics_complete_0[key] = [item for sublist in metrics_complete_0[key] for item in sublist]
    metrics_complete_1[key] = [item for sublist in metrics_complete_1[key] for item in sublist]
    metrics_complete_2[key] = [item for sublist in metrics_complete_2[key] for item in sublist]

# Compute the mean and the std deviation of the metrics
for key in metrics:
    mean_0 = round(float(np.mean(metrics_complete_0[key])), 3)  # Metric 0
    mean_1 = round(float(np.mean(metrics_complete_1[key])), 3)  # Metric 1
    mean_2 = round(float(np.mean(metrics_complete_2[key])), 3)  # Metric 2
    std_0 = round(float(np.std(metrics_complete_0[key])), 3)  # Metric 0
    std_1 = round(float(np.std(metrics_complete_1[key])), 3)  # Metric 1
    std_2 = round(float(np.std(metrics_complete_2[key])), 3)  # Metric 2
    results['mean'][key] = [mean_0, mean_1, mean_2]
    results['std'][key] = [std_0, std_1, std_2]

# Save the results
results['metrics_0'] = metrics_complete_0
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
print(f'{end_time}\tInference ends !\nElapsed time: {end_time - start_time}')
print(f'==> End2End Models in {model_folder} <==')
print(f'==> Output dir: {current_output_dir_path}\nRun name: {run_name} <==')
