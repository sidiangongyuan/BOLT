# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import random
import statistics
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path (resume). When set, the config is loaded from '
                             '`model_dir/config.yaml` and checkpoints are saved into `model_dir`.')
    parser.add_argument(
        '--stage1_model_dir',
        default='',
        help='Initialize from a different training directory (load checkpoint weights) while still '
             'using the current `-y` YAML config. This is designed for Stage-2 open-join training '
             'so that outputs are saved to a new log directory.',
    )
    parser.add_argument(
        '--output_dir',
        default='',
        help='Optional output directory for a new run (scratch or Stage-2). If empty, a new folder '
             'is created automatically (train_utils.setup_train).',
    )
    parser.add_argument(
        '--token_cache_dir',
        default='',
        help=(
            "Override `token.cache_dir` in the YAML for training. "
            "Useful for oracle-geometry diagnostics without duplicating YAMLs."
        ),
    )
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    opt = parser.parse_args()
    return opt


def main():
    seed_all() # seed all random functions to make the training reproducible
    opt = train_parser()
    if opt.model_dir and opt.stage1_model_dir:
        raise ValueError("Use either --model_dir (resume) or --stage1_model_dir (init), not both.")
    if opt.model_dir and opt.output_dir:
        raise ValueError("--output_dir is not supported together with --model_dir (resume in-place).")

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    if opt.token_cache_dir:
        if "token" not in hypes or not isinstance(hypes["token"], dict):
            raise ValueError(
                "--token_cache_dir is set but config has no `token` section. "
                "This override is intended for camera-token models."
            )
        hypes["token"]["enabled"] = True
        hypes["token"]["cache_dir"] = str(opt.token_cache_dir)
        print(f"[train] override token.cache_dir={hypes['token']['cache_dir']}")

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True, calibrate=False)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False,
                                              calibrate=False) 

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=4,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)

    print(model.get_memory_footprint())
    # exit()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    
    # half precision training
    if opt.half:
        print("Half precision training.")
        scaler = torch.cuda.amp.GradScaler()

    # Resume training in-place: load config.yaml from model_dir and keep saving into model_dir.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch (model_dir={saved_path}).")

    # New run: create a fresh output dir. Optionally initialize weights from stage1_model_dir.
    else:
        init_epoch = 0
        if opt.output_dir:
            saved_path = opt.output_dir
            os.makedirs(saved_path, exist_ok=True)
            try:
                train_utils.backup_script(saved_path)
            except Exception as e:
                print(f"[WARN] backup_script failed: {e}")
            save_name = os.path.join(saved_path, 'config.yaml')
            if not os.path.exists(save_name):
                with open(save_name, 'w') as outfile:
                    import yaml
                    yaml.dump(hypes, outfile)
        else:
            saved_path = train_utils.setup_train(hypes)

        if opt.stage1_model_dir:
            ckpt_epoch, model = train_utils.load_saved_model(opt.stage1_model_dir, model)
            print(
                f"init weights from stage1_model_dir={opt.stage1_model_dir} "
                f"(loaded ckpt epoch {ckpt_epoch}), start stage2 from epoch 0, save_dir={saved_path}."
            )

        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # record training
    writer = SummaryWriter(saved_path)

    ic(model)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        no_grad_steps = 0
        grad_steps = 0
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        # the model will be evaluation mode during validation
        model.train()
        try: # heter_model stage2
            model.model_train_init()
        except:
            print("No model_train_init function")
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            
            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                if not opt.half:
                    final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                else:
                    with torch.cuda.amp.autocast():
                        final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            if not isinstance(final_loss, torch.Tensor) or not final_loss.requires_grad:
                no_grad_steps += 1
                if no_grad_steps <= 3:
                    record_len = batch_data['ego'].get('record_len', None)
                    agent_modality_list = batch_data['ego'].get('agent_modality_list', None)
                    print(
                        "[WARN] Skip backward: loss has no grad_fn. "
                        "This usually happens when the trainable Stage-2 module is unused in this batch "
                        "(e.g., non-ego agent filtered by comm_range). "
                        f"epoch={epoch} iter={i} record_len={record_len} agent_modality_list={agent_modality_list}"
                    )
                continue
            grad_steps += 1

            # back-propagation
            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # torch.cuda.empty_cache()  # it will destroy memory buffer

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        scheduler.step(epoch)

        opencood_train_dataset.reinitialize()

        if grad_steps == 0:
            raise RuntimeError(
                "No valid training steps produced gradients in this epoch. "
                "For Stage-2 open-join, this typically means the target modality agent "
                "is always filtered out (e.g., comm_range too small) or the trainable module "
                "is not used by the current model/dataset wiring."
            )

    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = False
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
