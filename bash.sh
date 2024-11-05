# ——————— Path Examples
# <user_directory>/runs/logs/<timestamp>_<flag>  <-- Tensorboard
# <user_directory>/runs/end2end/<source_domain_dataset>/<target_domain_dataset>/<timestamp>_<flag>  <-- Model and Data

# ——————— Run Examples
# python train.py --source_domain_dataset <source_domain_dataset> --target_domain_dataset <target_domain_dataset> --em_consistency_loss --feature_alignment_loss --gamma 0.5 --delta 0.0 --ptr_sr --ptr_sr_run_flag <pretrained_sr_net_run_flag>
# python test.py --train_run_timestamp <train_run_timestamp> --main_dataset <lr_dataset> --source_domain_dataset <source_domain_dataset> --target_domain_dataset <target_domain_dataset>  --ptr_sr --ptr_sr_run_flag <pretrained_sr_net_run_flag>
