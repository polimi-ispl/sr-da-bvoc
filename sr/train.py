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
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import BVOCDatasetSR
from utils import set_backend, set_seed, gradient_norm, check_gradients, shorten_datetime, BASE_OUTPUT_DIR_PATH_SR, BASE_DATASET_PATH, compute_metrics
from sr.models.SAN.san import SAN

set_backend()
set_seed(10)

########################################
# Params ArgumentParser()              #
########################################
# RUN PATH EXAMPLE --> runs/sr/<train_dataset_folder>/x<upscaling_factor>/<model_name>/<timestamp>_<flag>

parser = argparse.ArgumentParser(description='BVOC Super-Resolution Network Training')

# Network
parser.add_argument('--model_name', type=str, default='SAN', help='Name of the Super-Resolution network')
parser.add_argument('--pretrained', action='store_true', help='Use a pretrained model')
parser.add_argument('--pretrained_model_path', type=str, default='', help='Path to the pretrained model weights')

# Data Specs
parser.add_argument('--hr_patch_size', type=int, default=32, help='Size of high-resolution patches')
parser.add_argument('--dataset_folder', type=str, default='',
                    help='Specific folder for the dataset')
parser.add_argument('--percentage', type=int, default=100, help='The fraction of the total available elements that will be randomly selected for training.'
                                                                'This will be applied after the areas selection')

# Training Specs
parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
parser.add_argument('--n_epochs', type=int, default=100000, help='Number of epochs to train')
parser.add_argument('--upscaling_factor', type=int, default=2, help='Upscaling factor for the super-resolution')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--lr_patience', type=int, default=10,
                    help='Learning rate scheduler patience, after which the learning rate is reduced. Works only with on-validation scheduler type.')
parser.add_argument('--es_patience', type=int, default=50,
                    help='Early stopping patience, after which the training stops')
parser.add_argument('--prefetch_factor', type=int, default=20, help='Prefetch factor for the dataloader')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--train_batches_percentage', type=float, default=1.0,
                    help='Percentage of train batches, after which the validation starts. 1.0=all the train batches are considered before the validation starts')
parser.add_argument('--val_batches_percentage', type=float, default=1.0,
                    help='Percentage of validation batches, after which the next epochs starts, 1.0=all the validation batches are considered before the next epoch starts')
parser.add_argument('--warmup', type=int, default=3, help='Warmup epochs')
parser.add_argument('--es_min_delta', type=float, default=0, help='Minimum delta for early stopping')
parser.add_argument('--num_workers', type=int, default=6, help='Number of workers of the dataloader')
parser.add_argument('--scheduler_type', type=str, default='on-validation',
                    help='Type of lr scheduler. Options: global (CosineAnnealingWarmRestarts), on-validation (ReduceLROnPlateau)')

# Training Regularization
parser.add_argument('--clip_gradient', action='store_true', help='Clip the gradient to prevent grad explosion')
parser.add_argument('--opt_weight_decay', type=float, default=0.0,
                    help='L2 regularization of the loss function. This penalizes large weights in the model, which indirectly discourages large output values')

# Additional Specs
parser.add_argument('--flag', type=str, default='', help='suffix for the output dir')

args = parser.parse_args()
print(args)

########################################
# Paths and Folders                    #
########################################
start_time = datetime.now()
output_folder_name = f'{shorten_datetime(start_time)}_{args.flag}' if args.flag is not '' else f'{shorten_datetime(start_time)}'
output_dir_path = os.path.join(BASE_OUTPUT_DIR_PATH_SR, args.dataset_folder, f'x{args.upscaling_factor}',
                               f'{args.model_name}', output_folder_name)  # divided by dataset
tmp_run_name = f'{shorten_datetime(start_time)}_{args.flag}'
writer = SummaryWriter(log_dir=os.path.join(BASE_OUTPUT_DIR_PATH_SR, 'logs', tmp_run_name))
os.makedirs(output_dir_path, exist_ok=True)
print(
    f'==> Output dir: {output_dir_path}\nRun name: {output_folder_name}\nTensorboard dir: {os.path.join(BASE_OUTPUT_DIR_PATH_SR, "logs", tmp_run_name)} <==')

########################################
# Variables Definition                 #
########################################
# GPU
torch.cuda.set_device(args.gpu_id)
print("Using GPU: " + str(torch.cuda.current_device()))

# Networks
network = []
if args.model_name == 'SAN':
    network = SAN(scale=args.upscaling_factor)
else:
    print('SR Network not implemented !!!')

if args.pretrained:
    network.load_state_dict(torch.load(args.pretrained_model_path))
network.cuda()

# Losses
criterion = torch.nn.MSELoss()

# Optimizer & LR scheduler
optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.opt_weight_decay)

if args.scheduler_type == 'global':
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-07)
elif args.scheduler_type == 'on-validation':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                              patience=args.lr_patience, min_lr=1e-07)

# Datasets and DataLoaders
dataset_kwargs = {'root': os.path.join(BASE_DATASET_PATH, args.dataset_folder),
                  'downscale_factor': args.upscaling_factor,
                  'quantile_transform': True,
                  'downscaling_mode': 'bicubic',
                  'percentage': args.percentage,
                  }

dataloader_kwargs = {'batch_size': args.batch_size,
                     'num_workers': args.num_workers,
                     'pin_memory': True,
                     'prefetch_factor': args.prefetch_factor,
                     }

train_dataloader = DataLoader(BVOCDatasetSR(**dataset_kwargs, dataset_mode='train'), shuffle=True, drop_last=True,
                              **dataloader_kwargs)
val_dataloader = DataLoader(BVOCDatasetSR(**dataset_kwargs, dataset_mode='val'), shuffle=False, drop_last=True,
                            **dataloader_kwargs)

# Info
print(f'\nBatches: \tTR {len(train_dataloader)} | VL {len(val_dataloader)}'
      f'\nN. Maps: \tTR {len(train_dataloader) * args.batch_size} | VL {len(val_dataloader) * args.batch_size}')

########################################
# Training                             #
########################################
metrics = ['SSIM', 'PSNR', 'NMSE', 'UIQ', 'SCC', 'MaxAE']
best_val_loss = np.inf
best_epoch = 1
batches_before_valid = round(len(train_dataloader) * args.train_batches_percentage)
batches_before_break_valid = round(len(val_dataloader) * args.val_batches_percentage)
tr_len = min(len(train_dataloader), batches_before_valid)
vl_len = min(len(val_dataloader), batches_before_break_valid)
tb_img_idx = 1  # TensorBoard selected image's index of the current batch

print(f'\n{start_time}\tTrain starts !')
print(f'Considered batches TR: {batches_before_valid} | VL: {batches_before_break_valid}')
for epoch in range(1, args.n_epochs):
    # ———————————————— Epoch Starts ————————————————
    ###################################### T R A I N ######################################
    total_train_loss, total_val_loss = 0, 0
    total_train_mse, total_val_mse = 0, 0
    total_train_em_penalty, total_val_em_penalty = 0, 0
    total_train_em_consistency, total_val_em_consistency = 0, 0
    network.train()
    for i, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}/{args.n_epochs}', unit='batch')):
        optimizer.zero_grad()
        # (b, c, h, w)
        hr_img = batch['HR'].cuda()
        lr_img = batch['LR'].cuda()
        output = network(lr_img)

        train_loss = criterion(output, hr_img)

        train_loss.backward()
        if args.clip_gradient:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5)  # Clip the gradient to prevent grad explosion
        optimizer.step()

        total_train_loss += train_loss.item()

        # Progress report (TensorBoard + Log)
        x_axis_train = epoch * tr_len
        writer.add_scalar('BatchLoss/Train', train_loss.item(), x_axis_train)
        writer.add_image('Maps/Train/HR', hr_img[tb_img_idx], x_axis_train)
        writer.add_image('Maps/Train/LR', lr_img[tb_img_idx], x_axis_train)
        writer.add_image('Maps/Train/SR', output[tb_img_idx], x_axis_train)
        writer.add_text('Stats/Train', '\n'.join(
            f'{name}| Min {img[tb_img_idx].min().item():.4f} / Max {img[tb_img_idx].max().item():.4f}\n' for name, img
            in zip(['LR', 'HR', 'SR'], [lr_img, hr_img, output])), x_axis_train)
        if not torch.isnan(output[tb_img_idx]).any() and not torch.isinf(output[tb_img_idx]).any():
            writer.add_histogram('Histograms/Train/SR', output[tb_img_idx], x_axis_train)
            writer.add_histogram('Histograms/Train/HR', hr_img[tb_img_idx], x_axis_train)
        # Gradient monitorning
        writer.add_scalar('Gradients', gradient_norm(network), x_axis_train)
        problematic, problematic_params = check_gradients(network)
        if problematic:
            writer.add_text('ProblematicParams', problematic_params, x_axis_train)
            print(f'Problematic gradients found in {problematic_params}')

        # Early validation, when a certain percentage of train batches is processed
        if i == batches_before_valid:
            break  # exit from the inner for loop (TR BATCHES), start the validation

    ###################################### V A L I D A T I O N ######################################
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch}/{args.n_epochs}', unit='batch')):
            hr_img = batch['HR'].cuda()
            lr_img = batch['LR'].cuda()
            output = network(lr_img)

            val_loss = criterion(output, hr_img)

            total_val_loss += val_loss.item()

            # Progress report (TensorBoard)
            x_axis_val = epoch * vl_len
            writer.add_scalar('BatchLoss/Val', val_loss.item(), x_axis_val)
            writer.add_image('Maps/Val/HR', hr_img[tb_img_idx], x_axis_val)
            writer.add_image('Maps/Val/LR', lr_img[tb_img_idx], x_axis_val)
            writer.add_image('Maps/Val/SR', output[tb_img_idx], x_axis_val)
            writer.add_text('Stats/Val', '\n'.join(
                f'{name}| Min {img[tb_img_idx].min().item():.4f} / Max {img[tb_img_idx].max().item():.4f}\n' for
                name, img in zip(['LR', 'HR', 'SR'], [lr_img, hr_img, output])), x_axis_val)
            if not torch.isnan(output[tb_img_idx]).any() and not torch.isinf(output[tb_img_idx]).any():
                writer.add_histogram('Histograms/Val/SR', output[tb_img_idx], x_axis_val)
                writer.add_histogram('Histograms/Val/HR', hr_img[tb_img_idx], x_axis_val)

            # Short validation, when a certain percentage of val batches is processed
            if i == batches_before_break_valid:
                break  # exit from the inner for loop (VL BATCHES), pass to the next epoch

    # Epoch Averaged Losses
    avg_train_loss, avg_val_loss = total_train_loss / tr_len, total_val_loss / vl_len

    # Epoch Report (TensorBoard + Log)
    results = compute_metrics(output, hr_img, metrics=metrics, mean=True)  # Compute metrics on the LAST batch ONLY, thus no mean
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning Rate', current_lr, epoch)
    writer.add_scalar('Loss/Train/Total', avg_train_loss, epoch)
    writer.add_scalar('Loss/Val/Total', avg_val_loss, epoch)
    writer.add_scalar('Metrics/Val/SSIM', results['SSIM'], epoch)
    writer.add_scalar('Metrics/Val/NMSE', results['NMSE'], epoch)
    writer.add_scalar('Metrics/Val/PSNR', results['PSNR'], epoch)

    print(f'Epoch {epoch}/{args.n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}'
          f'\nSSIM: {results["SSIM"]:.4f} | NMSE: {results["NMSE"]:.4f}dB | MAE: {results["MAE"]:.4f}')

    print(f'Learning Rate: {current_lr:.8f}')
    # ———————————————— Epoch Ends ————————————————

    # Save the best model
    if avg_val_loss < best_val_loss - args.es_min_delta:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        best_SSIM = results['SSIM']
        best_NMSE = results['NMSE']
        # save BEST models checkpoints
        torch.save(network.state_dict(), os.path.join(output_dir_path, f'{args.model_name}_best.pth'))
        print(f'Best model saved at epoch {epoch} with loss {best_val_loss:.8f}!')
    else:  # Early stopping, after initial warmup
        if (epoch - best_epoch) > args.es_patience and epoch > args.warmup:
            last_loss = avg_val_loss
            # Save LAST models checkpoints
            torch.save(network.state_dict(), os.path.join(output_dir_path, f'{args.model_name}_last.pth'))
            print(f'Early stopping at epoch {epoch} with loss {last_loss:.8f}!'
                  f'\nBest model saved at epoch {best_epoch} with loss {best_val_loss:.8f}!')
            break  # to exit from the main for loop (EPOCHS)
        else:
            print(f'No improvement. Early stopping in {epoch - best_epoch}/{args.es_patience}')

    # Update learning rate
    if args.scheduler_type == 'global':
        lr_scheduler.step()
    elif args.scheduler_type == 'on-validation':
        lr_scheduler.step(avg_val_loss)

###################################
writer.close()
end_time = datetime.now()
print(f'{end_time}\tTrain ends !\nElapsed time: {end_time - start_time}')
print(f'Best loss {best_val_loss:.4f} and SSIM {best_SSIM:.4f}, NMSE {best_NMSE:.4f}dB at epoch {best_epoch}')
print(f'==> {output_folder_name} <==')
