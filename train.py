"""
Giganti, A.; Mandelli, S.; Bestagini, P.; Tubaro, S.
Learn from Simulations, Adapt to Observations: Super-Resolution of Isoprene Emissions via Unpaired Domain Adaptation.
Remote Sens. 2024, 16, 3963. https://doi.org/10.3390/rs16213963

© 2024 Antonio Giganti - Image and Sound Processing Lab (ISPL) - Politecnico di Milano, Italy.
"""

import argparse
import itertools
import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from da.models import Generator, Discriminator
from utils import set_backend, set_seed, shorten_datetime, BASE_OUTPUT_DIR_PATH_END2END, BASE_DATASET_PATH, \
    compute_metrics, BASE_ROOT_PATH
from da.utils import ReplayBuffer, weights_init_normal, emission_consistency_loss
from dataset import BVOCDataset_end2end
from sr.models.SAN.san import SAN

set_backend()
set_seed(10)

# A domain --> Observed (OMI/GOME)
# B domain --> Simulated (MEGAN)

######################################
# Params ArgumentParser()            #
######################################
# RUN PATH EXAMPLE --> runs/end2end/<source_domain_dataset>/<target_domain_dataset>/<timestamp>_<flag>

parser = argparse.ArgumentParser(description='BVOC Super-Resolution + Domain Adaptation Networks Training')

# Network
parser.add_argument('--pretrained', action='store_true', help='Use a pretrained model')
parser.add_argument('--pretrained_da_model_run_flag', type=str, default='',
                    help='Run flag of the pretrained model weights for the Domain-Adaptation networks')
parser.add_argument('--ptr_sr_run_flag', type=str, default='',
                    help='Run flag of the pretrained model weights of the Super-Resolution nerwork')
parser.add_argument('--sr_dataset', type=str, default='', help='Dataset of the pretrained SR network. Leave empty to use the same simulated dataset of the DA network')
parser.add_argument('--da_generator_n_residual_blocks', type=int, default=9,
                    help='Number of residual blocks in the generator')
parser.add_argument('--da_generator_n_layers', type=int, default=2,
                    help='Number of layers of the generator"s encoder and decoder. The number of layers of the generator"s encoder and decoder are supposed to be the same')
parser.add_argument('--da_generator_kernel_size', type=int, default=3, help='Kernel size of the generator')
parser.add_argument('--da_generator_stride', type=int, default=2, help='Stride of the generator')
parser.add_argument('--da_discriminator_kernel_size', type=int, default=2,
                    help='Kernel size of the discriminator')
parser.add_argument('--da_discriminator_stride', type=int, default=1, help='Stride of the discriminator')

# Data Specs
parser.add_argument('--source_domain_dataset', type=str, default='',
                    help='root directory of the source domain (A) dataset')
parser.add_argument('--target_domain_dataset', type=str, default='',
                    help='root directory of the target domain (B) dataset')
parser.add_argument('--percentage', type=int, default=100,
                    help='The fraction of the total available elements that will be randomly selected for training.')

# Training Specs
parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
parser.add_argument('--n_epochs', type=int, default=100000, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--lr_patience', type=int, default=10,
                    help='Learning rate scheduler patience, after which the learning rate is reduced. Works only with on-validation scheduler type.')
parser.add_argument('--lr_delta', type=float, default=1e-4, help='Learning rate scheduler min delta improvement')
parser.add_argument('--es_patience', type=int, default=50,
                    help='Early stopping patience, after which the training stops')
parser.add_argument('--prefetch_factor', type=int, default=20, help='Prefetch factor for the dataloader')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--train_batches_percentage', type=float, default=1.0,
                    help='Percentage of train batches, after which the validation starts. 1.0=all the train batches are considered before the validation starts')
parser.add_argument('--val_batches_percentage', type=float, default=1.0,
                    help='Percentage of validation batches, after which the next epochs starts, 1.0=all the validation batches are considered before the next epoch starts')
parser.add_argument('--warmup', type=int, default=3, help='Warmup epochs')
parser.add_argument('--num_workers', type=int, default=6, help='Number of workers of the dataloader')
parser.add_argument('--scheduler_type', type=str, default='on-validation',
                    help='Type of lr scheduler. Options: global (CosineAnnealingWarmRestarts), on-validation (ReduceLROnPlateau)')
parser.add_argument('--pseudo_e2e', action='store_true',
                    help='Flag that enable the training as "pseudo end2end", meaning that the input of the DA network is NOT the actual output of the SR network; it is like training the DA networks separately')
parser.add_argument('--ptr_sr', action='store_true',
                    help='Flag that avoid the training as of the Super-Resolution network, using a pretrained models trained only on B domain data. The SR network is frozen during training')

# Training Regularization
parser.add_argument('--opt_weight_decay', type=float, default=0.0,
                    help='For Adam, L2 regularization of the loss function. This penalizes large weights in the model, which indirectly discourages large output values. For AdamW, the behaviour is totally different !')
parser.add_argument('--opt_beta1', type=float, default=0.9,
                    help='Adam optimizer decay of first order momentum of gradient for the Domain Adaptation network')
# Coefficients
parser.add_argument('--beta', type=float, default=1.0,
                    help='Domain balance of the Super-Resolution loss: [beta] for B domain loss; [1-beta] for A domain loss. Used only if the SR network is trained from scratch')
parser.add_argument('--feature_alignment_loss', action='store_true',
                    help='Flag that enables the addition of the Feature Alignment loss contribution to the total loss')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='Discriminators domain balance of the Feature Alignment loss: [gamma] for A domain discriminators (D_A_lr, D_A_hr); [1-gamma] for B domain discriminators (D_B_lr, D_B_hr)')
parser.add_argument('--delta', type=float, default=1.0,
                    help='Task balance of the Feature Alignment loss: [delta] weight the intermediate task (LR, thus extracting features from D_A_lr, D_B_lr); L_align_1 loss on the paper.'
                         '[1-delta] weight the final task (HR, thus extracting features from D_A_hr, D_B_hr), L_align_2 loss on the paper.'
                         'Please, check the paper for more in depth details')
parser.add_argument('--em_consistency_loss', action='store_true',
                    help='Flag that enables the addition of the Emission Consistency loss (EmC) contribution to the total loss')

# Scale Factors
parser.add_argument('--loss_SR_B_scale_factor', type=float, default=20000,
                    help='Scale factor of the loss term related to the B domain of the SR network')
parser.add_argument('--loss_SR_A_scale_factor', type=float, default=500,
                    help='Scale factor of the loss term related to the A domain of the SR network')
parser.add_argument('--loss_emc_scale_factor', type=float, default=10,
                    help='Scale factor of the loss term related to the Emission Consistency (EmC)')

# Additional Specs
parser.add_argument('--flag', type=str, default='', help='suffix for the output dir')

args = parser.parse_args()
print(args)

########################################
# Paths and Folders                    #
########################################
start_time = datetime.now()
output_folder_name = f'{shorten_datetime(start_time)}_{args.flag}' if args.flag is not '' else f'{shorten_datetime(start_time)}'
output_dir_path = os.path.join(BASE_OUTPUT_DIR_PATH_END2END, args.source_domain_dataset, args.target_domain_dataset,
                               output_folder_name)  # divided by dataset
tmp_run_name = f'{shorten_datetime(start_time)}_{args.flag}'
writer = SummaryWriter(log_dir=os.path.join(BASE_OUTPUT_DIR_PATH_END2END, 'logs', tmp_run_name))
os.makedirs(output_dir_path, exist_ok=True)
print(
    f'==> Output dir: {output_dir_path}\nRun name: {output_folder_name}\nTensorboard dir: {os.path.join(BASE_OUTPUT_DIR_PATH_END2END, "logs", tmp_run_name)} <==')

########################################
# Variables Definition                 #
########################################
# GPU
torch.cuda.set_device(args.gpu_id)
print("Using GPU: " + str(torch.cuda.current_device()))

# Networks
# SR
sr_net_B = SAN(scale=args.upscaling_factor)

# DA
input_nc, output_nc = 1, 1
# HR
netG_A2B_hr = Generator(input_nc, output_nc, n_layers=args.da_generator_n_layers,
                        kernel_size=args.da_generator_kernel_size, stride=args.da_generator_stride,
                        n_residual_blocks=args.da_generator_n_residual_blocks)
netG_B2A_hr = Generator(output_nc, input_nc, n_layers=args.da_generator_n_layers,
                        kernel_size=args.da_generator_kernel_size, stride=args.da_generator_stride,
                        n_residual_blocks=args.da_generator_n_residual_blocks)
netD_A_hr = Discriminator(input_nc, kernel_size=args.da_discriminator_kernel_size, stride=args.da_discriminator_stride)
netD_B_hr = Discriminator(output_nc, kernel_size=args.da_discriminator_kernel_size, stride=args.da_discriminator_stride)
# LR
netG_A2B_lr = Generator(input_nc, output_nc, n_layers=args.da_generator_n_layers,
                        kernel_size=args.da_generator_kernel_size, stride=args.da_generator_stride,
                        n_residual_blocks=args.da_generator_n_residual_blocks)
netG_B2A_lr = Generator(output_nc, input_nc, n_layers=args.da_generator_n_layers,
                        kernel_size=args.da_generator_kernel_size, stride=args.da_generator_stride,
                        n_residual_blocks=args.da_generator_n_residual_blocks)
netD_A_lr = Discriminator(input_nc, kernel_size=args.da_discriminator_kernel_size, stride=args.da_discriminator_stride)
netD_B_lr = Discriminator(output_nc, kernel_size=args.da_discriminator_kernel_size, stride=args.da_discriminator_stride)

# Initialization
sr_net_B.cuda()
netG_A2B_hr.cuda()
netG_B2A_hr.cuda()
netD_A_hr.cuda()
netD_B_hr.cuda()

netG_A2B_lr.cuda()
netG_B2A_lr.cuda()
netD_A_lr.cuda()
netD_B_lr.cuda()

netG_A2B_hr.apply(weights_init_normal)
netG_B2A_hr.apply(weights_init_normal)
netD_A_hr.apply(weights_init_normal)
netD_B_hr.apply(weights_init_normal)

netG_A2B_lr.apply(weights_init_normal)
netG_B2A_lr.apply(weights_init_normal)
netD_A_lr.apply(weights_init_normal)
netD_B_lr.apply(weights_init_normal)

# Pretrained Nets
if args.ptr_sr or args.pretrained:
    # Decide whether to use the same simulated dataset of the DA network or a different one, for the SR network
    if args.sr_dataset is not '':  sr_dataset = args.sr_dataset
    else:  sr_dataset = args.target_domain_dataset

    pretrained_sr_model = os.path.join(BASE_ROOT_PATH, 'runs', 'sr', sr_dataset, 'x' + args.upscaling_factor, 'SAN', args.ptr_sr_run_flag, 'SAN_best.pth')
    sr_net_B.load_state_dict(torch.load(pretrained_sr_model, map_location=torch.device('cuda')))
    print(f'==> Pretrained SR net from run {args.ptr_sr_run_flag} loaded !')

if args.pretrained:
    pretrained_da_model_folder = os.path.join(BASE_ROOT_PATH, 'runs', 'da',
                                              args.source_domain_dataset, args.target_domain_dataset,
                                              args.pretrained_da_model_run_flag)
    netG_A2B_hr.load_state_dict(
        torch.load(os.path.join(pretrained_da_model_folder, 'netG_A2B_best.pth'), map_location=torch.device('cuda')))
    netG_B2A_hr.load_state_dict(
        torch.load(os.path.join(pretrained_da_model_folder, 'netG_B2A_best.pth'), map_location=torch.device('cuda')))
    netD_A_hr.load_state_dict(
        torch.load(os.path.join(pretrained_da_model_folder, 'netD_A_best.pth'), map_location=torch.device('cuda')))
    netD_B_hr.load_state_dict(
        torch.load(os.path.join(pretrained_da_model_folder, 'netD_B_best.pth'), map_location=torch.device('cuda')))

    netG_A2B_lr.load_state_dict(
        torch.load(os.path.join(pretrained_da_model_folder, 'netG_A2B_best.pth'), map_location=torch.device('cuda')))
    netG_B2A_lr.load_state_dict(
        torch.load(os.path.join(pretrained_da_model_folder, 'netG_B2A_best.pth'), map_location=torch.device('cuda')))
    netD_A_lr.load_state_dict(
        torch.load(os.path.join(pretrained_da_model_folder, 'netD_A_best.pth'), map_location=torch.device('cuda')))
    netD_B_lr.load_state_dict(
        torch.load(os.path.join(pretrained_da_model_folder, 'netD_B_best.pth'), map_location=torch.device('cuda')))
    print(f'==> Pretrained DA nets from run {args.pretrained_da_model_run_flag} loaded !')

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_feature = torch.nn.MSELoss()
criterion_SR = torch.nn.MSELoss()
# Emission Consistency (EmC) Loss args
emc_loss_kwargs = {'overlapping_factor': 1,
                   'patch_size_lr': 4,
                   'upscaling_factor': 2}

# Optimizers & LR schedulers
optimizer_kwargs = {'lr': args.lr, 'betas': (args.opt_beta1, 0.999), 'weight_decay': args.opt_weight_decay}
optimizer_COMB = torch.optim.Adam(
    itertools.chain((sr_net_B.parameters() if not args.ptr_sr else ()), netG_A2B_hr.parameters(),
                    netG_B2A_hr.parameters(), netG_A2B_lr.parameters(), netG_B2A_lr.parameters()), **optimizer_kwargs)
optimizer_Ds = torch.optim.Adam(
    itertools.chain(netD_A_hr.parameters(), netD_A_lr.parameters(), netD_B_hr.parameters(),
                    netD_B_lr.parameters()), **optimizer_kwargs)

if args.scheduler_type == 'global':
    lr_scheduler_kwargs = {'T_0': 5, 'T_mult': 2, 'eta_min': 1e-07}
    lr_scheduler_COMB = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_COMB, **lr_scheduler_kwargs)
    lr_scheduler_Ds = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_Ds, **lr_scheduler_kwargs)
elif args.scheduler_type == 'on-validation':
    lr_scheduler_kwargs = {'mode': 'min', 'factor': 0.1, 'patience': args.lr_patience, 'min_lr': 1e-07,
                           'threshold': args.lr_delta}
    lr_scheduler_COMB = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_COMB, **lr_scheduler_kwargs)
    lr_scheduler_Ds = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_Ds, **lr_scheduler_kwargs)

# Inputs & targets memory allocation for the DA network
Tensor = torch.cuda.FloatTensor
target_real = Variable(Tensor(args.batch_size, 1).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(args.batch_size, 1).fill_(0.0), requires_grad=False)

fake_A_buffer_hr = ReplayBuffer()
fake_B_buffer_hr = ReplayBuffer()

fake_A_buffer_lr = ReplayBuffer()
fake_B_buffer_lr = ReplayBuffer()

# Datasets and DataLoaders
dataset_kwargs = {'root_source_domain_dataset': os.path.join(BASE_DATASET_PATH, args.source_domain_dataset),
                  # OMI/GOME (A) source
                  'root_target_domain_dataset': os.path.join(BASE_DATASET_PATH, args.target_domain_dataset),
                  # MEGAN (B) target
                  'downscale_factor': 2,
                  'unaligned': True,
                  'quantile_transform': True,
                  'downscaling_mode': 'bicubic',
                  'percentage': args.percentage,
                  }

dataloader_kwargs = {'batch_size': args.batch_size,
                     'num_workers': args.num_workers,
                     'pin_memory': True,
                     'prefetch_factor': args.prefetch_factor,
                     }

train_dataloader = DataLoader(
    BVOCDataset_end2end(**dataset_kwargs, dataset_mode='train'), shuffle=True, drop_last=True, **dataloader_kwargs)

val_dataloader = DataLoader(
    BVOCDataset_end2end(**dataset_kwargs, dataset_mode='val'), shuffle=False, drop_last=True, **dataloader_kwargs)

# Info
print(f'\nBatches: \tTR {len(train_dataloader)} | VL {len(val_dataloader)}'
      f'\nN. Maps: \tTR {len(train_dataloader) * args.batch_size} | VL {len(val_dataloader) * args.batch_size}')

########################################
# Training                             #
########################################
sr_metrics = ['SSIM', 'PSNR', 'NMSE', 'UIQ', 'SCC', 'MaxAE']
COMB_best_val_loss = np.inf
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
    # Loss and nets initialization
    total_train_loss_DA_part, total_train_loss_Ds_part, total_val_loss_DA_part, total_val_loss_Ds_part = 0, 0, 0, 0
    total_train_loss_SR_part, total_val_loss_SR_part, = 0, 0

    total_train_loss_feature, total_val_loss_feature = 0, 0
    total_train_loss_em_consistency, total_val_loss_em_consistency = 0, 0
    total_train_loss_feature_intermediate, total_train_loss_feature_final = 0, 0
    total_val_loss_feature_intermediate, total_val_loss_feature_final = 0, 0

    total_train_loss_COMB, total_val_loss_COMB = 0, 0  # <== Combined Loss

    total_train_loss_G, total_train_loss_D_A, total_train_loss_D_B = 0, 0, 0
    total_train_loss_G_hr, total_train_loss_D_A_hr, total_train_loss_D_B_hr = 0, 0, 0
    total_train_loss_G_lr, total_train_loss_D_A_lr, total_train_loss_D_B_lr = 0, 0, 0
    total_train_loss_G_Identity_hr, total_train_loss_G_GAN_hr, total_train_loss_G_Cycle_hr = 0, 0, 0
    total_train_loss_G_Identity_lr, total_train_loss_G_GAN_lr, total_train_loss_G_Cycle_lr = 0, 0, 0

    total_val_loss_G, total_val_loss_D_A, total_val_loss_D_B = 0, 0, 0
    total_val_loss_G_hr, total_val_loss_D_A_hr, total_val_loss_D_B_hr = 0, 0, 0
    total_val_loss_G_lr, total_val_loss_D_A_lr, total_val_loss_D_B_lr = 0, 0, 0
    total_val_loss_G_Identity_hr, total_val_loss_G_GAN_hr, total_val_loss_G_Cycle_hr, total_val_loss_D_hr = 0, 0, 0, 0
    total_val_loss_G_Identity_lr, total_val_loss_G_GAN_lr, total_val_loss_G_Cycle_lr, total_val_loss_D_lr = 0, 0, 0, 0

    netG_A2B_hr.train()
    netG_A2B_lr.train()
    netG_B2A_hr.train()
    netG_B2A_lr.train()
    netD_A_hr.train()
    netD_A_lr.train()
    netD_B_hr.train()
    netD_B_lr.train()
    if args.ptr_sr:
        sr_net_B.eval()
    else:
        sr_net_B.train()

    for i, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}/{args.n_epochs}', unit='batch')):
        # —————————————————————————————————————————————— S T E P starts ———————————————————————————————————
        # Set model input
        # (b, c, h, w)
        real_A_hr = batch['A_hr'].cuda()  # HR, OMI/GOMI (A), for DA
        real_A_lr = batch['A_lr'].cuda()  # LR, OMI/GOMI (A), for testing the SR --> DA
        real_B_hr = batch['B_hr'].cuda()  # HR, MEGAN (B), for SR and/or DA
        real_B_lr = batch['B_lr'].cuda()  # LR, MEGAN (B), for SR

        optimizer_COMB.zero_grad()  # it acts for the A2B (hr and lr), B2A (hr and lr)
        ###################################### Super-Resolution ########################################################
        if not args.ptr_sr:
            # Loss 1: qt_B_hr_hat_B - qtB_hr_B | B domain
            # Classic SR I/O loss
            real_B_hr_hat = sr_net_B(real_B_lr)
            train_loss_SR_B = criterion_SR(real_B_hr_hat, real_B_hr)  # Loss on B domain
            # Loss 2: qt_A_hat_hr_hat_B - qt_A_hat_hr_B | A domain
            # Aka fake_sr_A - fake_A, or better DA_B2A(SR_B(real_B_lr)) - DA_B2A(real_B), thus B elements on the A domain
            # Optimal case: they are the same. Needed to force the SR network to super-resolve also in the A domain, but using ONLY B domain data (qt_A_hat)
            qt_A_hat_hr_hat_B = netG_B2A_hr(real_B_hr_hat)
            qt_A_hat_hr_B = netG_B2A_hr(real_B_hr)
            train_loss_SR_A = criterion_SR(qt_A_hat_hr_hat_B, qt_A_hat_hr_B)  # Loss on A domain
            # Scaling
            train_loss_SR_B = args.loss_SR_B_scale_factor * train_loss_SR_B
            train_loss_SR_A = args.loss_SR_A_scale_factor * train_loss_SR_A
            # Combining the losses
            train_loss_SR_part = args.beta * train_loss_SR_B + (1 - args.beta) * train_loss_SR_A  # <== SR part loss
        else:
            train_loss_SR_part = 0

        ###################################### Input Settings #####################################################
        # ATTENTION: these settings have to be done after the SR process, for pipeline coherence.
        # Check the "train_configuration.md" table for all the possible configs!
        intermediate_task = netG_A2B_lr(real_A_lr)  # <== (A2B(A))
        if args.ptr_sr:
            with torch.no_grad():
                intermediate_task_sr = sr_net_B(intermediate_task)  # <== SR(A2B(A))
        else:
            intermediate_task_sr = sr_net_B(intermediate_task)
        final_task = netG_B2A_hr(intermediate_task_sr)  # <== OUR FINAL TASK !!! B2A(SR((A2B(A))))

        if not args.pseudo_e2e:  # true end2end training (OUR PROPOSAL)
            real_B_hr = intermediate_task_sr.detach()  # shortcut to use the intermediate_task_sr as input of the DA hr networks
        # else, we use the real_B_hr as it is, thus performing a pseudo end2end training

        ##################################### Domain Adaptation ########################################################
        # PAY ATTENTION: we have 2 different CycleGAN architectures (aka A2B, B2A, D_A, D_B), one for the HR and one for the LR patches
        ###### Generators A2B and B2A ######
        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B_hr = netG_A2B_hr(real_B_hr)
        same_B_lr = netG_A2B_lr(real_B_lr)
        loss_identity_B_hr = criterion_identity(same_B_hr, real_B_hr) * 5.0
        loss_identity_B_lr = criterion_identity(same_B_lr, real_B_lr) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A_hr = netG_B2A_hr(real_A_hr)
        same_A_lr = netG_B2A_lr(real_A_lr)
        loss_identity_A_hr = criterion_identity(same_A_hr, real_A_hr) * 5.0
        loss_identity_A_lr = criterion_identity(same_A_lr, real_A_lr) * 5.0
        # GAN loss
        fake_B_hr = netG_A2B_hr(real_A_hr)
        fake_B_lr = netG_A2B_lr(real_A_lr)
        pred_fake_hr = netD_B_hr(fake_B_hr)
        pred_fake_lr = netD_B_lr(fake_B_lr)
        loss_GAN_A2B_hr = criterion_GAN(pred_fake_hr, target_real)
        loss_GAN_A2B_lr = criterion_GAN(pred_fake_lr, target_real)
        fake_A_hr = netG_B2A_hr(real_B_hr)
        fake_A_lr = netG_B2A_lr(real_B_lr)
        pred_fake_hr = netD_A_hr(fake_A_hr)
        pred_fake_lr = netD_A_lr(fake_A_lr)
        loss_GAN_B2A_hr = criterion_GAN(pred_fake_hr, target_real)
        loss_GAN_B2A_lr = criterion_GAN(pred_fake_lr, target_real)
        # Cycle loss
        recovered_A_hr = netG_B2A_hr(fake_B_hr)
        recovered_A_lr = netG_B2A_lr(fake_B_lr)
        loss_cycle_ABA_hr = criterion_cycle(recovered_A_hr, real_A_hr) * 10.0
        loss_cycle_ABA_lr = criterion_cycle(recovered_A_lr, real_A_lr) * 10.0
        recovered_B_hr = netG_A2B_hr(fake_A_hr)
        recovered_B_lr = netG_A2B_lr(fake_A_lr)
        loss_cycle_BAB_hr = criterion_cycle(recovered_B_hr, real_B_hr) * 10.0
        loss_cycle_BAB_lr = criterion_cycle(recovered_B_lr, real_B_lr) * 10.0
        # Total loss
        train_loss_G_hr = loss_identity_A_hr + loss_identity_B_hr + loss_GAN_A2B_hr + loss_GAN_B2A_hr + loss_cycle_ABA_hr + loss_cycle_BAB_hr
        train_loss_G_lr = loss_identity_A_lr + loss_identity_B_lr + loss_GAN_A2B_lr + loss_GAN_B2A_lr + loss_cycle_ABA_lr + loss_cycle_BAB_lr
        # Loss to be monitored
        train_loss_B2A_hr_related = loss_identity_A_hr + loss_GAN_B2A_hr + loss_cycle_ABA_hr
        train_loss_A2B_lr_related = loss_identity_B_lr + loss_GAN_A2B_lr + loss_cycle_BAB_lr
        # Combining the losses
        train_loss_DA_part = train_loss_G_hr + train_loss_G_lr  # <== DA part loss

        ### Here we guide the GAN generation ###
        # Feature alignment loss [to be done THEORETICALLY only at the END of the pipeline (B2A)]
        # Force the HR_A and HR_hat_A embeddings have to be the same. Needed for a correct final DA (B2A)
        # >> feature and NOT distribution, because we have the GT for the A domain
        # Needed JUST for D_A, but also done for D_B for loss regularization, if needed
        obj_pred_emb_A_hr, obj_pred_emb_B_hr = netD_A_hr.model(final_task), netD_B_hr.model(final_task)
        obj_gt_emb_A_hr, obj_gt_emb_B_hr = netD_A_hr.model(real_A_hr), netD_B_hr.model(real_A_hr)
        train_loss_feature_A_hr = criterion_feature(obj_pred_emb_A_hr, obj_gt_emb_A_hr)
        train_loss_feature_B_hr = criterion_feature(obj_pred_emb_B_hr, obj_gt_emb_B_hr)
        # Done also on the LR counterpart, thus in the intermediate_task point (A2B),even if we do not have the GT for the A domain
        obj_pred_emb_B_lr, obj_pred_emb_A_lr = netD_B_lr.model(intermediate_task), netD_A_lr.model(intermediate_task)
        obj_gt_emb_B_lr, obj_gt_emb_A_lr = netD_B_lr.model(real_B_lr), netD_A_lr.model(real_B_lr)
        train_loss_feature_B_lr = criterion_feature(obj_pred_emb_B_lr, obj_gt_emb_B_lr)
        train_loss_feature_A_lr = criterion_feature(obj_pred_emb_A_lr, obj_gt_emb_A_lr)
        # Combining the losses
        # Alignment Loss
        L_align_1 = args.gamma * train_loss_feature_A_lr + (1 - args.gamma) * train_loss_feature_B_lr
        L_align_2 = args.gamma * train_loss_feature_A_hr + (1 - args.gamma) * train_loss_feature_B_hr
        train_loss_feature = args.delta * L_align_1 + (1 - args.delta) * L_align_2  # <== Feature Alignment loss
        # Emission Consistency Loss
        train_loss_em_consistency = args.loss_emc_scale_factor * emission_consistency_loss(final_task, real_A_lr,
                                                                                           **emc_loss_kwargs)  # <== Emission Consistency Loss

        ##################################### Combined Loss #######################################################
        # ==> FINAL LOSS: Super-Resolution + Generators + Feature Alignment + Emission Consistency <==
        train_loss_COMB = train_loss_SR_part + train_loss_DA_part + args.feature_alignment_loss * train_loss_feature + args.em_consistency_loss * train_loss_em_consistency

        # Backward
        train_loss_COMB.backward()
        optimizer_COMB.step()

        ###################################
        ###### Discriminators A and B ######

        optimizer_Ds.zero_grad()  # it acts for the D_A (hr and lr) and D_B (hr and lr) networks
        ###### Discriminator A ######
        # Real loss
        pred_real_hr = netD_A_hr(real_A_hr)
        pred_real_lr = netD_A_lr(real_A_lr)
        loss_D_real_hr = criterion_GAN(pred_real_hr, target_real)
        loss_D_real_lr = criterion_GAN(pred_real_lr, target_real)
        # Fake loss
        fake_A_hr = fake_A_buffer_hr.push_and_pop(fake_A_hr)
        fake_A_lr = fake_A_buffer_lr.push_and_pop(fake_A_lr)
        pred_fake_hr = netD_A_hr(fake_A_hr.detach())
        pred_fake_lr = netD_A_lr(fake_A_lr.detach())
        loss_D_fake_hr = criterion_GAN(pred_fake_hr, target_fake)
        loss_D_fake_lr = criterion_GAN(pred_fake_lr, target_fake)
        # Total loss
        train_loss_D_A_hr = (loss_D_real_hr + loss_D_fake_hr) * 0.5
        train_loss_D_A_lr = (loss_D_real_lr + loss_D_fake_lr) * 0.5

        ###### Discriminator B ######
        # Real loss
        pred_real_hr = netD_B_hr(real_B_hr)
        pred_real_lr = netD_B_lr(real_B_lr)
        loss_D_real_hr = criterion_GAN(pred_real_hr, target_real)
        loss_D_real_lr = criterion_GAN(pred_real_lr, target_real)
        # Fake loss
        fake_B_hr = fake_B_buffer_hr.push_and_pop(fake_B_hr)
        fake_B_lr = fake_B_buffer_lr.push_and_pop(fake_B_lr)
        pred_fake_hr = netD_B_hr(fake_B_hr.detach())
        pred_fake_lr = netD_B_lr(fake_B_lr.detach())
        loss_D_fake_hr = criterion_GAN(pred_fake_hr, target_fake)
        loss_D_fake_lr = criterion_GAN(pred_fake_lr, target_fake)
        # Total loss
        train_loss_D_B_hr = (loss_D_real_hr + loss_D_fake_hr) * 0.5
        train_loss_D_B_lr = (loss_D_real_lr + loss_D_fake_lr) * 0.5

        ###### Losses Ds ######
        # Discriminators
        train_loss_Ds_HR = train_loss_D_A_hr + train_loss_D_B_hr
        train_loss_Ds_LR = train_loss_D_A_lr + train_loss_D_B_lr
        # Combining the losses
        train_loss_Ds_part = train_loss_Ds_HR + train_loss_Ds_LR  # <== Ds part loss
        # Backward
        train_loss_Ds_part.backward()
        optimizer_Ds.step()

        ################################################################################################################

        # —————————————————————————————————————————————— S T E P ends ———————————————————————————————————

        total_train_loss_DA_part += train_loss_DA_part.item()
        total_train_loss_Ds_part += train_loss_Ds_part.item()
        total_train_loss_SR_part += train_loss_SR_part

        total_train_loss_G_hr += train_loss_G_hr.item()
        total_train_loss_D_A_hr += train_loss_D_A_hr.item()
        total_train_loss_D_B_hr += train_loss_D_B_hr.item()
        total_train_loss_G_Identity_hr += loss_identity_A_hr + loss_identity_B_hr
        total_train_loss_G_GAN_hr += loss_GAN_A2B_hr + loss_GAN_B2A_hr
        total_train_loss_G_Cycle_hr += loss_cycle_ABA_hr + loss_cycle_BAB_hr

        total_train_loss_G_lr += train_loss_G_lr.item()
        total_train_loss_D_A_lr += train_loss_D_A_lr.item()
        total_train_loss_D_B_lr += train_loss_D_B_lr.item()
        total_train_loss_G_Identity_lr += loss_identity_A_lr + loss_identity_B_lr
        total_train_loss_G_GAN_lr += loss_GAN_A2B_lr + loss_GAN_B2A_lr
        total_train_loss_G_Cycle_lr += loss_cycle_ABA_lr + loss_cycle_BAB_lr

        total_train_loss_feature += train_loss_feature.item()
        total_train_loss_feature_intermediate += L_align_1.item()
        total_train_loss_feature_final += L_align_2.item()

        total_train_loss_em_consistency += train_loss_em_consistency

        total_train_loss_COMB += train_loss_COMB.item()

        # Progress report (TensorBoard)
        x_axis_train = epoch * tr_len
        writer.add_image('Maps/Train/LR (Real_A_lr, qt_A_hr_A)', real_A_lr[tb_img_idx], x_axis_train)
        writer.add_image('Maps/Train/HR (Real_A_hr, qt_A_hr_A)', real_A_hr[tb_img_idx], x_axis_train)
        with torch.no_grad():
            qt_A_hat_hr_hat_A = netG_B2A_hr(sr_net_B(netG_A2B_lr(real_A_lr)))  # <== OUR FINAL TASK !!!
        writer.add_image('Maps/Train/SR (B2A(SR(A2B(LR_A))), qt_A_hat_hr_hat_A)', qt_A_hat_hr_hat_A[tb_img_idx],
                         x_axis_train)
        if not args.ptr_sr:
            writer.add_image('Maps/Train/LR (Real_B_lr)', real_B_lr[tb_img_idx], x_axis_train)
            writer.add_image('Maps/Train/HR (Real_B_hr)', real_B_hr[tb_img_idx], x_axis_train)
            writer.add_image('Maps/Train/SR (Real_B_hr_hat)', real_B_hr[tb_img_idx], x_axis_train)

        writer.add_histogram('Hist/Train/HR (Real_A_hr, qt_A_hr_A)', real_A_hr[tb_img_idx], x_axis_train)
        writer.add_histogram('Hist/Train/SR (B2A(SR(A2B(LR_A))), qt_A_hat_hr_hat_A)', qt_A_hat_hr_hat_A[tb_img_idx],
                             x_axis_train)

        # Early validation, when a certain percentage of train batches is processed
        if i == batches_before_valid:
            break  # exit from the inner for loop (TR BATCHES), start the validation

    ###################################### V A L I D A T I O N ######################################
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch}/{args.n_epochs}', unit='batch')):
            # —————————————————————————————————————————————— S T E P starts ———————————————————————————————————
            # Set model input
            # (b, c, h, w)
            real_A_hr = batch['A_hr'].cuda()  # HR, OMI/GOMI (A), for DA
            real_A_lr = batch['A_lr'].cuda()  # LR, OMI/GOMI (A), for testing the SR --> DA
            real_B_lr = batch['B_lr'].cuda()  # LR, MEGAN (B), for SR
            real_B_hr = batch['B_hr'].cuda()  # HR, MEGAN (B), for SR and/or DA

            ###################################### Super-Resolution ########################################################
            if not args.ptr_sr:
                real_B_hr_hat = sr_net_B(real_B_lr)
                val_loss_SR_B = criterion_SR(real_B_hr_hat, real_B_hr)  # Loss on B domain
                qt_A_hat_hr_hat_B = netG_B2A_hr(real_B_hr_hat)
                qt_A_hat_hr_B = netG_B2A_hr(real_B_hr)
                val_loss_SR_A = criterion_SR(qt_A_hat_hr_hat_B, qt_A_hat_hr_B)  # Loss on A domain
                # Scaling
                val_loss_SR_B = args.loss_SR_B_scale_factor * val_loss_SR_B
                val_loss_SR_A = args.loss_SR_A_scale_factor * val_loss_SR_A
                # Combining the losses
                val_loss_SR_part = args.beta * val_loss_SR_B + (1 - args.beta) * val_loss_SR_A  # <== SR part loss
            else:
                val_loss_SR_part = 0

            ###################################### Input Settings #####################################################
            intermediate_task = netG_A2B_lr(real_A_lr)  # <== (A2B(A))
            intermediate_task_sr = sr_net_B(intermediate_task)  # <== SR(A2B(A))
            final_task = netG_B2A_hr(intermediate_task_sr)  # <== OUR FINAL TASK !!! B2A(SR((A2B(A))))

            if not args.pseudo_e2e:
                real_B_hr = intermediate_task_sr

            ##################################### Domain Adaptation ########################################################
            # PAY ATTENTION: we have 2 different CycleGAN architectures (aka A2B, B2A, D_A, D_B), one for the HR and one for the LR patches
            ###### Generators A2B and B2A ######
            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B_hr = netG_A2B_hr(real_B_hr)
            same_B_lr = netG_A2B_lr(real_B_lr)
            loss_identity_B_hr = criterion_identity(same_B_hr, real_B_hr) * 5.0
            loss_identity_B_lr = criterion_identity(same_B_lr, real_B_lr) * 5.0
            # G_B2A(A) should equal A if real A is fed
            same_A_hr = netG_B2A_hr(real_A_hr)
            same_A_lr = netG_B2A_lr(real_A_lr)
            loss_identity_A_hr = criterion_identity(same_A_hr, real_A_hr) * 5.0
            loss_identity_A_lr = criterion_identity(same_A_lr, real_A_lr) * 5.0
            # GAN loss
            fake_B_hr = netG_A2B_hr(real_A_hr)
            fake_B_lr = netG_A2B_lr(real_A_lr)
            pred_fake_hr = netD_B_hr(fake_B_hr)
            pred_fake_lr = netD_B_lr(fake_B_lr)
            loss_GAN_A2B_hr = criterion_GAN(pred_fake_hr, target_real)
            loss_GAN_A2B_lr = criterion_GAN(pred_fake_lr, target_real)
            fake_A_hr = netG_B2A_hr(real_B_hr)
            fake_A_lr = netG_B2A_lr(real_B_lr)
            pred_fake_hr = netD_A_hr(fake_A_hr)
            pred_fake_lr = netD_A_lr(fake_A_lr)
            loss_GAN_B2A_hr = criterion_GAN(pred_fake_hr, target_real)
            loss_GAN_B2A_lr = criterion_GAN(pred_fake_lr, target_real)
            # Cycle loss
            recovered_A_hr = netG_B2A_hr(fake_B_hr)
            recovered_A_lr = netG_B2A_lr(fake_B_lr)
            loss_cycle_ABA_hr = criterion_cycle(recovered_A_hr, real_A_hr) * 10.0
            loss_cycle_ABA_lr = criterion_cycle(recovered_A_lr, real_A_lr) * 10.0
            recovered_B_hr = netG_A2B_hr(fake_A_hr)
            recovered_B_lr = netG_A2B_lr(fake_A_lr)
            loss_cycle_BAB_hr = criterion_cycle(recovered_B_hr, real_B_hr) * 10.0
            loss_cycle_BAB_lr = criterion_cycle(recovered_B_lr, real_B_lr) * 10.0
            # Total loss
            val_loss_G_hr = loss_identity_A_hr + loss_identity_B_hr + loss_GAN_A2B_hr + loss_GAN_B2A_hr + loss_cycle_ABA_hr + loss_cycle_BAB_hr
            val_loss_G_lr = loss_identity_A_lr + loss_identity_B_lr + loss_GAN_A2B_lr + loss_GAN_B2A_lr + loss_cycle_ABA_lr + loss_cycle_BAB_lr
            # Loss to be monitored
            val_loss_B2A_hr_related = loss_identity_A_hr + loss_GAN_B2A_hr + loss_cycle_ABA_hr
            val_loss_A2B_lr_related = loss_identity_B_lr + loss_GAN_A2B_lr + loss_cycle_BAB_lr
            # Combining the losses
            val_loss_DA_part = val_loss_G_hr + val_loss_G_lr  # <== DA part loss

            # Feature alignment loss
            obj_pred_emb_A_hr, obj_pred_emb_B_hr = netD_A_hr.model(final_task), netD_B_hr.model(final_task)
            obj_gt_emb_A_hr, obj_gt_emb_B_hr = netD_A_hr.model(real_A_hr), netD_B_hr.model(real_A_hr)
            val_loss_feature_A_hr = criterion_feature(obj_pred_emb_A_hr, obj_gt_emb_A_hr)
            val_loss_feature_B_hr = criterion_feature(obj_pred_emb_B_hr, obj_gt_emb_B_hr)
            # LR counterpart
            obj_pred_emb_B_lr, obj_pred_emb_A_lr = netD_B_lr.model(intermediate_task), netD_A_lr.model(
                intermediate_task)
            obj_gt_emb_B_lr, obj_gt_emb_A_lr = netD_B_lr.model(real_B_lr), netD_A_lr.model(real_B_lr)
            val_loss_feature_B_lr = criterion_feature(obj_pred_emb_B_lr, obj_gt_emb_B_lr)
            val_loss_feature_A_lr = criterion_feature(obj_pred_emb_A_lr, obj_gt_emb_A_lr)
            # Combining the losses
            # Alignment Loss
            L_align_1 = args.gamma * val_loss_feature_A_lr + (1 - args.gamma) * val_loss_feature_B_lr
            L_align_2 = args.gamma * val_loss_feature_A_hr + (1 - args.gamma) * val_loss_feature_B_hr
            val_loss_feature = args.delta * L_align_1 + (1 - args.delta) * L_align_2  # <== Feature Alignment loss
            # Emission Consistency Loss
            val_loss_em_consistency = args.loss_emc_scale_factor * emission_consistency_loss(final_task, real_A_lr,
                                                                                             **emc_loss_kwargs)  # <== Emission Consistency Loss

            ##################################### Combined Loss #######################################################
            # ==> FINAL LOSS: Super-Resolution + Generators + Feature Alignment + Emission Consistency <==
            val_loss_COMB = val_loss_SR_part + val_loss_DA_part + args.feature_alignment_loss * val_loss_feature + args.em_consistency_loss * val_loss_em_consistency

            ###################################
            ###### Discriminator A ######
            # Real loss
            pred_real_hr = netD_A_hr(real_A_hr)
            pred_real_lr = netD_A_lr(real_A_lr)
            loss_D_real_hr = criterion_GAN(pred_real_hr, target_real)
            loss_D_real_lr = criterion_GAN(pred_real_lr, target_real)
            # Fake loss
            fake_A_hr = fake_A_buffer_hr.push_and_pop(fake_A_hr)
            fake_A_lr = fake_A_buffer_lr.push_and_pop(fake_A_lr)
            pred_fake_hr = netD_A_hr(fake_A_hr.detach())
            pred_fake_lr = netD_A_lr(fake_A_lr.detach())
            loss_D_fake_hr = criterion_GAN(pred_fake_hr, target_fake)
            loss_D_fake_lr = criterion_GAN(pred_fake_lr, target_fake)
            # Total loss
            val_loss_D_A_hr = (loss_D_real_hr + loss_D_fake_hr) * 0.5
            val_loss_D_A_lr = (loss_D_real_lr + loss_D_fake_lr) * 0.5

            ###### Discriminator B ######
            # Real loss
            pred_real_hr = netD_B_hr(real_B_hr)
            pred_real_lr = netD_B_lr(real_B_lr)
            loss_D_real_hr = criterion_GAN(pred_real_hr, target_real)
            loss_D_real_lr = criterion_GAN(pred_real_lr, target_real)
            # Fake loss
            fake_B_hr = fake_B_buffer_hr.push_and_pop(fake_B_hr)
            fake_B_lr = fake_B_buffer_lr.push_and_pop(fake_B_lr)
            pred_fake_hr = netD_B_hr(fake_B_hr.detach())
            pred_fake_lr = netD_B_lr(fake_B_lr.detach())
            loss_D_fake_hr = criterion_GAN(pred_fake_hr, target_fake)
            loss_D_fake_lr = criterion_GAN(pred_fake_lr, target_fake)
            # Total loss
            val_loss_D_B_hr = (loss_D_real_hr + loss_D_fake_hr) * 0.5
            val_loss_D_B_lr = (loss_D_real_lr + loss_D_fake_lr) * 0.5

            ###### Losses Ds ######
            # Discriminators
            val_loss_Ds_HR = val_loss_D_A_hr + val_loss_D_B_hr
            val_loss_Ds_LR = val_loss_D_A_lr + val_loss_D_B_lr
            # Combining the losses
            val_loss_Ds_part = val_loss_Ds_HR + val_loss_Ds_LR  # <== Ds part loss

            ################################################################################################################

            # —————————————————————————————————————————————— S T E P ends ———————————————————————————————————

            total_val_loss_DA_part += val_loss_DA_part.item()
            total_val_loss_Ds_part += val_loss_Ds_part.item()
            total_val_loss_SR_part += val_loss_SR_part

            total_val_loss_G_hr += val_loss_G_hr.item()
            total_val_loss_D_A_hr += val_loss_D_A_hr.item()
            total_val_loss_D_B_hr += val_loss_D_B_hr.item()
            total_val_loss_G_Identity_hr += loss_identity_A_hr + loss_identity_B_hr
            total_val_loss_G_GAN_hr += loss_GAN_A2B_hr + loss_GAN_B2A_hr
            total_val_loss_G_Cycle_hr += loss_cycle_ABA_hr + loss_cycle_BAB_hr

            total_val_loss_G_lr += val_loss_G_lr.item()
            total_val_loss_D_A_lr += val_loss_D_A_lr.item()
            total_val_loss_D_B_lr += val_loss_D_B_lr.item()
            total_val_loss_G_Identity_lr += loss_identity_A_lr + loss_identity_B_lr
            total_val_loss_G_GAN_lr += loss_GAN_A2B_lr + loss_GAN_B2A_lr
            total_val_loss_G_Cycle_lr += loss_cycle_ABA_lr + loss_cycle_BAB_lr

            total_val_loss_feature += val_loss_feature.item()
            total_val_loss_feature_intermediate += L_align_1.item()
            total_val_loss_feature_final += L_align_2.item()

            total_val_loss_em_consistency += val_loss_em_consistency

            total_val_loss_COMB += val_loss_COMB.item()

            # Progress report (TensorBoard)
            x_axis_val = epoch * vl_len
            writer.add_image('Maps/Val/LR (Real_A_lr, qt_A_hr_A)', real_A_lr[tb_img_idx], x_axis_val)
            writer.add_image('Maps/Val/HR (Real_A_hr, qt_A_hr_A)', real_A_hr[tb_img_idx], x_axis_val)
            with torch.no_grad():
                qt_A_hat_hr_hat_A = netG_B2A_hr(sr_net_B(netG_A2B_lr(real_A_lr)))  # <== OUR FINAL TASK !!!
            writer.add_image('Maps/Val/SR (B2A(SR(A2B(LR_A))), qt_A_hat_hr_hat_A)', qt_A_hat_hr_hat_A[tb_img_idx],
                             x_axis_val)
            if not args.ptr_sr:
                writer.add_image('Maps/Val/LR (Real_B_lr)', real_B_lr[tb_img_idx], x_axis_val)
                writer.add_image('Maps/Val/HR (Real_B_hr)', real_B_hr[tb_img_idx], x_axis_val)
                writer.add_image('Maps/Val/SR (Real_B_hr_hat)', real_B_hr[tb_img_idx], x_axis_val)

            writer.add_histogram('Hist/Val/HR (Real_A_hr, qt_A_hr_A)', real_A_hr[tb_img_idx], x_axis_val)
            writer.add_histogram('Hist/Val/SR (B2A(SR(A2B(LR_A))), qt_A_hat_hr_hat_A)', qt_A_hat_hr_hat_A[tb_img_idx],
                                 x_axis_val)

            # Short validation, when a certain percentage of val batches is processed
            if i == batches_before_break_valid:
                break  # exit from the inner for loop (VL BATCHES), pass to the next epoch

    # Epoch Averaged Losses
    avg_train_loss_DA_part, avg_train_loss_Ds_part, avg_train_loss_SR_part = total_train_loss_DA_part / tr_len, total_train_loss_Ds_part / tr_len, total_train_loss_SR_part / tr_len
    avg_val_loss_DA_part, avg_val_loss_Ds_part, avg_val_loss_SR_part = total_val_loss_DA_part / vl_len, total_val_loss_Ds_part / vl_len, total_val_loss_SR_part / vl_len

    avg_train_loss_G_hr, avg_train_loss_D_A_hr, avg_train_loss_D_B_hr = total_train_loss_G_hr / tr_len, total_train_loss_D_A_hr / tr_len, total_train_loss_D_B_hr / tr_len
    avg_train_loss_G_lr, avg_train_loss_D_A_lr, avg_train_loss_D_B_lr = total_train_loss_G_lr / tr_len, total_train_loss_D_A_lr / tr_len, total_train_loss_D_B_lr / tr_len
    avg_train_loss_G_Identity_hr, avg_train_loss_G_GAN_hr, avg_train_loss_G_Cycle_hr = total_train_loss_G_Identity_hr / tr_len, total_train_loss_G_GAN_hr / tr_len, total_train_loss_G_Cycle_hr / tr_len
    avg_train_loss_G_Identity_lr, avg_train_loss_G_GAN_lr, avg_train_loss_G_Cycle_lr = total_train_loss_G_Identity_lr / tr_len, total_train_loss_G_GAN_lr / tr_len, total_train_loss_G_Cycle_lr / tr_len

    avg_val_loss_G_hr, avg_val_loss_D_A_hr, avg_val_loss_D_B_hr = total_val_loss_G_hr / vl_len, total_val_loss_D_A_hr / vl_len, total_val_loss_D_B_hr / vl_len
    avg_val_loss_G_lr, avg_val_loss_D_A_lr, avg_val_loss_D_B_lr = total_val_loss_G_lr / vl_len, total_val_loss_D_A_lr / vl_len, total_val_loss_D_B_lr / vl_len
    avg_val_loss_G_Identity_hr, avg_val_loss_G_GAN_hr, avg_val_loss_G_Cycle_hr = total_val_loss_G_Identity_hr / vl_len, total_val_loss_G_GAN_hr / vl_len, total_val_loss_G_Cycle_hr / vl_len
    avg_val_loss_G_Identity_lr, avg_val_loss_G_GAN_lr, avg_val_loss_G_Cycle_lr = total_val_loss_G_Identity_lr / vl_len, total_val_loss_G_GAN_lr / vl_len, total_val_loss_G_Cycle_lr / vl_len

    avg_train_loss_feature, avg_val_loss_feature = total_train_loss_feature / tr_len, total_val_loss_feature / vl_len
    avg_train_loss_feature_intermediate, avg_val_loss_feature_intermediate = total_train_loss_feature_intermediate / tr_len, total_val_loss_feature_intermediate / vl_len
    avg_train_loss_feature_final, avg_val_loss_feature_final = total_train_loss_feature_final / tr_len, total_val_loss_feature_final / vl_len

    avg_train_loss_em_consistency, avg_val_loss_em_consistency = total_train_loss_em_consistency / tr_len, total_val_loss_em_consistency / vl_len

    avg_train_loss_COMB, avg_val_loss_COMB = total_train_loss_COMB / tr_len, total_val_loss_COMB / vl_len

    # Epoch Report (TensorBoard + Log)
    # Compute metrics on the LAST batch and for VAL ONLY, thus no mean
    # BE CAREFUL: here, metrics are computed in the QT domain, not BVOC domain! A
    # At test time, we will have to convert the SR maps to the BVOC domain
    results_A = compute_metrics(qt_A_hat_hr_hat_A, real_A_hr, metrics=sr_metrics, mean=True)  # A domain
    if not args.ptr_sr: results_B = compute_metrics(real_B_hr_hat, real_B_hr, metrics=sr_metrics, mean=True)  # B domain

    current_lr_COMB = optimizer_COMB.param_groups[0]['lr']
    current_lr_Ds = optimizer_Ds.param_groups[0]['lr']

    # Learning Rates
    writer.add_scalar('Learning Rate/COMB', current_lr_COMB, epoch)
    writer.add_scalar('Learning Rate/Ds', current_lr_Ds, epoch)

    # Training Losses
    writer.add_scalar('Loss/Train/DA_part', avg_train_loss_DA_part, epoch)
    writer.add_scalar('Loss/Train/Ds_part', avg_train_loss_Ds_part, epoch)
    writer.add_scalar('Loss/Train/SR_part', avg_train_loss_SR_part, epoch)

    writer.add_scalar('Loss/Train/G_hr', avg_train_loss_G_hr, epoch)
    writer.add_scalar('Loss/Train/D_A_hr', avg_train_loss_D_A_hr, epoch)
    writer.add_scalar('Loss/Train/D_B_hr', avg_train_loss_D_B_hr, epoch)
    writer.add_scalar('Loss/Train/G_lr', avg_train_loss_G_lr, epoch)
    writer.add_scalar('Loss/Train/D_A_lr', avg_train_loss_D_A_lr, epoch)
    writer.add_scalar('Loss/Train/D_B_lr', avg_train_loss_D_B_lr, epoch)

    writer.add_scalar('Loss/Train/G_Identity_hr', avg_train_loss_G_Identity_hr, epoch)
    writer.add_scalar('Loss/Train/G_GAN_hr', avg_train_loss_G_GAN_hr, epoch)
    writer.add_scalar('Loss/Train/G_Cycle_hr', avg_train_loss_G_Cycle_hr, epoch)
    writer.add_scalar('Loss/Train/G_Identity_lr', avg_train_loss_G_Identity_lr, epoch)
    writer.add_scalar('Loss/Train/G_GAN_lr', avg_train_loss_G_GAN_lr, epoch)
    writer.add_scalar('Loss/Train/G_Cycle_lr', avg_train_loss_G_Cycle_lr, epoch)

    writer.add_scalar('Loss/Train/Feature Alignment', avg_train_loss_feature, epoch)
    writer.add_scalar('Loss/Train/Feature Alignment Intermediate', avg_train_loss_feature_intermediate, epoch)
    writer.add_scalar('Loss/Train/Feature Alignment Final', avg_train_loss_feature_final, epoch)

    writer.add_scalar('Loss/Train/Em Consistency', avg_train_loss_em_consistency, epoch)

    writer.add_scalar('Loss/Train/COMB', avg_train_loss_COMB, epoch)

    # Validation Losses
    writer.add_scalar('Loss/Val/DA_part', avg_val_loss_DA_part, epoch)
    writer.add_scalar('Loss/Val/Ds_part', avg_val_loss_Ds_part, epoch)
    writer.add_scalar('Loss/Val/SR_part', avg_val_loss_SR_part, epoch)

    writer.add_scalar('Loss/Val/G_hr', avg_val_loss_G_hr, epoch)
    writer.add_scalar('Loss/Val/D_A_hr', avg_val_loss_D_A_hr, epoch)
    writer.add_scalar('Loss/Val/D_B_hr', avg_val_loss_D_B_hr, epoch)
    writer.add_scalar('Loss/Val/G_lr', avg_val_loss_G_lr, epoch)
    writer.add_scalar('Loss/Val/D_A_lr', avg_val_loss_D_A_lr, epoch)
    writer.add_scalar('Loss/Val/D_B_lr', avg_val_loss_D_B_lr, epoch)

    writer.add_scalar('Loss/Val/G_Identity_hr', avg_val_loss_G_Identity_hr, epoch)
    writer.add_scalar('Loss/Val/G_GAN_hr', avg_val_loss_G_GAN_hr, epoch)
    writer.add_scalar('Loss/Val/G_Cycle_hr', avg_val_loss_G_Cycle_hr, epoch)
    writer.add_scalar('Loss/Val/G_Identity_lr', avg_val_loss_G_Identity_lr, epoch)
    writer.add_scalar('Loss/Val/G_GAN_lr', avg_val_loss_G_GAN_lr, epoch)
    writer.add_scalar('Loss/Val/G_Cycle_lr', avg_val_loss_G_Cycle_lr, epoch)

    writer.add_scalar('Loss/Val/Feature Alignment', avg_val_loss_feature, epoch)
    writer.add_scalar('Loss/Val/Feature Alignment Intermediate', avg_val_loss_feature_intermediate, epoch)
    writer.add_scalar('Loss/Val/Feature Alignment Final', avg_val_loss_feature_final, epoch)

    writer.add_scalar('Loss/Val/Em Consistency', avg_val_loss_em_consistency, epoch)

    writer.add_scalar('Loss/Val/COMB', avg_val_loss_COMB, epoch)

    # Metrics
    writer.add_scalar('Metrics/Val/SSIM A', results_A['SSIM'], epoch)
    writer.add_scalar('Metrics/Val/PSNR A', results_A['PSNR'], epoch)
    writer.add_scalar('Metrics/Val/NMSE A', results_A['NMSE'], epoch)
    writer.add_scalar('Metrics/Val/UIQ A', results_A['UIQ'], epoch)
    writer.add_scalar('Metrics/Val/SCC A', results_A['SCC'], epoch)
    writer.add_scalar('Metrics/Val/MaxAE A', results_A['MaxAE'], epoch)

    if not args.ptr_sr:
        writer.add_scalar('Metrics/Val/SSIM B', results_B['SSIM'], epoch)
        writer.add_scalar('Metrics/Val/PSNR B', results_B['PSNR'], epoch)
        writer.add_scalar('Metrics/Val/NMSE B', results_B['NMSE'], epoch)
        writer.add_scalar('Metrics/Val/UIQ B', results_B['UIQ'], epoch)
        writer.add_scalar('Metrics/Val/SCC B', results_B['SCC'], epoch)
        writer.add_scalar('Metrics/Val/MaxAE B', results_B['MaxAE'], epoch)

    print(f'{datetime.now()} | Epoch {epoch}/{args.n_epochs} |')
    print(
        f'TR Loss | DA: {avg_train_loss_DA_part:.6f} | F_align: {avg_train_loss_feature:.6f} | EmC: {avg_train_loss_em_consistency:.6f} | COMB: {avg_train_loss_COMB:.6f}'
        f'\nVL Loss | DA: {avg_val_loss_DA_part:.6f} | F_align: {avg_val_loss_feature:.6f} | EmC: {avg_val_loss_em_consistency:.6f} | COMB: {avg_val_loss_COMB:.6f}')
    print(f'VL SR Metrics: SSIM {results_A["SSIM"]:.4f} | NMSE {results_A["NMSE"]:.4f}')
    print(f'LRs: COMB {current_lr_COMB:.8f} | Ds {current_lr_Ds:.8f}\n')

    # ———————————————— Epoch Ends ————————————————

    # Save the best models
    if avg_val_loss_COMB < COMB_best_val_loss:  # Improvement
        COMB_best_val_loss = avg_val_loss_COMB
        best_epoch = epoch
        # save BEST models checkpoints
        torch.save(netG_A2B_hr.state_dict(), os.path.join(output_dir_path, 'netG_A2B_hr_best.pth'))
        torch.save(netG_B2A_hr.state_dict(), os.path.join(output_dir_path, 'netG_B2A_hr_best.pth'))
        torch.save(netD_A_hr.state_dict(), os.path.join(output_dir_path, 'netD_A_hr_best.pth'))
        torch.save(netD_B_hr.state_dict(), os.path.join(output_dir_path, 'netD_B_hr_best.pth'))
        torch.save(netG_A2B_lr.state_dict(), os.path.join(output_dir_path, 'netG_A2B_lr_best.pth'))
        torch.save(netG_B2A_lr.state_dict(), os.path.join(output_dir_path, 'netG_B2A_lr_best.pth'))
        torch.save(netD_A_lr.state_dict(), os.path.join(output_dir_path, 'netD_A_lr_best.pth'))
        torch.save(netD_B_lr.state_dict(), os.path.join(output_dir_path, 'netD_B_lr_best.pth'))
        if not args.ptr_sr: torch.save(sr_net_B.state_dict(), os.path.join(output_dir_path, 'sr_net_B_best.pth'))
        best_SSIM_A = results_A['SSIM']
        best_NMSE_A = results_A['NMSE']
        print(f'Best models saved at epoch {epoch} with loss {COMB_best_val_loss:.8f} !')
        print(f'A Domain Metrics ==> SSIM {best_SSIM_A:.4f} | NMSE {best_NMSE_A:.4f}')

    else:  # Early stopping, after initial warmup
        if (epoch - best_epoch) > args.es_patience and epoch > args.warmup:
            COMB_last_loss = COMB_best_val_loss
            # Save LAST models checkpoints
            torch.save(netG_A2B_hr.state_dict(), os.path.join(output_dir_path, 'netG_A2B_hr_last.pth'))
            torch.save(netG_B2A_hr.state_dict(), os.path.join(output_dir_path, 'netG_B2A_hr_last.pth'))
            torch.save(netD_A_hr.state_dict(), os.path.join(output_dir_path, 'netD_A_hr_last.pth'))
            torch.save(netD_B_hr.state_dict(), os.path.join(output_dir_path, 'netD_B_hr_last.pth'))
            torch.save(netG_A2B_lr.state_dict(), os.path.join(output_dir_path, 'netG_A2B_lr_last.pth'))
            torch.save(netG_B2A_lr.state_dict(), os.path.join(output_dir_path, 'netG_B2A_lr_last.pth'))
            torch.save(netD_A_lr.state_dict(), os.path.join(output_dir_path, 'netD_A_lr_last.pth'))
            torch.save(netD_B_lr.state_dict(), os.path.join(output_dir_path, 'netD_B_lr_last.pth'))
            if not args.ptr_sr: torch.save(sr_net_B.state_dict(), os.path.join(output_dir_path, 'sr_net_B_last.pth'))

            print(f'DA Early stopping at epoch {epoch} with loss {COMB_last_loss:.8f}!'
                  f'\nBest models saved at epoch {best_epoch} with loss: {COMB_best_val_loss:.8f} !')
            print(f'A Domain Metrics ==> SSIM {best_SSIM_A:.4f} | NMSE {best_NMSE_A:.4f}')
            break  # to exit from the main for loop (EPOCHS)
        else:
            print(f'No improvement. Early stopping in: {epoch - best_epoch}/{args.es_patience}')

    # Update learning rates
    if args.scheduler_type == 'global':
        lr_scheduler_COMB.step()
        lr_scheduler_Ds.step()
    elif args.scheduler_type == 'on-validation':
        lr_scheduler_COMB.step(avg_val_loss_COMB)
        lr_scheduler_Ds.step(avg_val_loss_Ds_part)

###################################
writer.close()
end_time = datetime.now()
print(f'{end_time}\tTrain ends !\nElapsed time: {end_time - start_time}')
print(f'Best loss {COMB_best_val_loss:.8f} at epoch {best_epoch}')
print(f'==> {output_folder_name} <==')
