# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import random
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
torch.multiprocessing.set_sharing_strategy('file_system')

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument(
        "--comm_range",
        type=float,
        default=None,
        help="Override comm_range in config.yaml for inference (e.g., 180).",
    )
    parser.add_argument(
        "--use_cav",
        type=int,
        default=0,
        help=(
            "For intermediateheter models only: use only the first K agents for feature fusion "
            "while keeping GT boxes from all agents (via intermediateheterinfer dataset). "
            "Set use_cav=1 to get an ego-only baseline comparable to collaborative inference."
        ),
    )
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="102.4,51.2",
                        help="detection range is [-102.4, +102.4, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--seed', default=42, type=int, 
                        help='random seed for results reproduction')
    parser.add_argument(
        '--token_ablation',
        default='normal',
        choices=['normal', 'drop', 'zero_feat', 'shuffle_feat'],
        help=(
            "Token ablation for camera-token models. "
            "'drop': set token_mask all-False; "
            "'zero_feat': keep xyz/mask but zero token_feat; "
            "'shuffle_feat': shuffle token_feat among valid tokens."
        ),
    )
    parser.add_argument(
        '--token_cache_dir',
        default='',
        help=(
            "Override `token.cache_dir` in `model_dir/config.yaml` for inference. "
            "Useful for oracle-geometry diagnostics without touching the trained config."
        ),
    )
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument(
        '--subset_start',
        type=int,
        default=0,
        help='Evaluate on a subset starting at this index (0 = from start).',
    )
    parser.add_argument(
        '--subset_end',
        type=int,
        default=0,
        help='Evaluate on a subset ending at this index (exclusive). 0 = full dataset.',
    )
    parser.add_argument('--disable_gate', action='store_true',
                        help="Force gate=1 (all ones) at inference for gate ablation.")
    parser.add_argument('--drop_camera', action='store_true',
                        help="Zero out camera features at inference (for HEAL model ablation).")
    parser.add_argument('--strict_n_car', type=int, default=0,
                        help='Strict N-car filtering: only evaluate scenes with exactly N agents (0=disabled)')
    parser.add_argument('--assignment_path', type=str, default=None,
                        help='Override modality assignment JSON file path')
    opt = parser.parse_args()
    return opt


def _apply_token_ablation(ego_dict, mode: str, seed: int, step: int) -> None:
    """
    Apply token ablations in-place on `ego_dict` (after `to_device`).

    This is used to produce quick negative controls for writing and debugging:
    - drop: remove all tokens (mask all False)
    - zero_feat: keep geometry + mask but remove semantics (feat -> 0)
    - shuffle_feat: destroy the xyz↔feature binding while preserving feature distribution
    """
    if mode == 'normal':
        return

    # Hetero protocol model stores camera inputs under keys like "inputs_m2".
    for key, inputs in list(ego_dict.items()):
        if not isinstance(key, str) or not key.startswith('inputs_'):
            continue
        if not isinstance(inputs, dict):
            continue
        if 'token_xyz' not in inputs or 'token_feat' not in inputs or 'token_mask' not in inputs:
            continue

        token_feat = inputs['token_feat']
        token_mask = inputs['token_mask']

        if not torch.is_tensor(token_feat) or not torch.is_tensor(token_mask):
            continue

        if mode == 'drop':
            inputs['token_mask'] = torch.zeros_like(token_mask, dtype=torch.bool)
            continue

        if mode == 'zero_feat':
            inputs['token_feat'] = torch.zeros_like(token_feat)
            continue

        if mode == 'shuffle_feat':
            # Deterministic per-step shuffle (so results are reproducible).
            gen = torch.Generator(device=token_feat.device)
            gen.manual_seed(int(seed) + int(step))

            feat = token_feat.clone()
            mask = token_mask.bool()

            # token_feat is expected to be [B,T,D] where B is #agents of this modality.
            if feat.ndim != 3 or mask.ndim != 2:
                raise ValueError(
                    f"Unexpected token shapes: token_feat={tuple(feat.shape)} token_mask={tuple(mask.shape)}"
                )

            for b in range(int(feat.shape[0])):
                valid_idx = torch.nonzero(mask[b], as_tuple=False).flatten()
                if valid_idx.numel() <= 1:
                    continue
                perm = valid_idx[torch.randperm(valid_idx.numel(), generator=gen, device=valid_idx.device)]
                feat[b, valid_idx] = feat[b, perm]

            inputs['token_feat'] = feat
            continue

        raise ValueError(f"Unknown token_ablation={mode}")


def main():
    opt = test_parser()
    seed_all(opt.seed)

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    if opt.token_cache_dir:
        if "token" not in hypes or not isinstance(hypes["token"], dict):
            raise ValueError(
                "--token_cache_dir is set but config has no `token` section. "
                "This override is intended for camera-token models."
            )
        hypes["token"]["enabled"] = True
        hypes["token"]["cache_dir"] = str(opt.token_cache_dir)
        print(f"[inference] override token.cache_dir={hypes['token']['cache_dir']}")

    if 'heter' in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir'] or "V2XREAL" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if opt.disable_gate:
        model._disable_gate = True
        print("[inference] Gate DISABLED (forced to all-ones)")

    if opt.drop_camera:
        model._drop_camera = True
        print("[inference] Camera features DROPPED (zeroed out)")

    if opt.comm_range is not None:
        hypes["comm_range"] = float(opt.comm_range)
        print(f"[inference] override comm_range={hypes['comm_range']}")

    if int(opt.use_cav) > 0:
        if "fusion" not in hypes or "core_method" not in hypes["fusion"]:
            raise ValueError("--use_cav requires hypes['fusion']['core_method'] in config.")
        fusion_core = str(hypes["fusion"]["core_method"])
        if fusion_core == "intermediateheter":
            hypes["fusion"]["core_method"] = "intermediateheterinfer"
        elif fusion_core != "intermediateheterinfer":
            raise ValueError(
                f"--use_cav is only supported for fusion.core_method=intermediateheter, got {fusion_core}."
            )
        hypes["use_cav"] = int(opt.use_cav)
        print(f"[inference] use_cav={hypes['use_cav']} (fusion.core_method={hypes['fusion']['core_method']})")

    if opt.strict_n_car > 0:
        hypes['strict_n_car'] = opt.strict_n_car
        print(f"[inference] strict_n_car={hypes['strict_n_car']} (only evaluate scenes with exactly {opt.strict_n_car} agents)")

    if opt.assignment_path:
        hypes['heter']['assignment_path'] = opt.assignment_path
        print(f"[inference] override assignment_path={opt.assignment_path}")

    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False, calibrate=False)

    dataset_for_loader = opencood_dataset
    if int(opt.subset_start) > 0 or int(opt.subset_end) > 0:
        start = max(int(opt.subset_start), 0)
        end = int(opt.subset_end) if int(opt.subset_end) > 0 else len(opencood_dataset)
        end = min(end, len(opencood_dataset))
        if end <= start:
            raise ValueError(f"Invalid subset range: start={start}, end={end}, len={len(opencood_dataset)}")
        dataset_for_loader = Subset(opencood_dataset, range(start, end))
        print(f"[inference] dataset subset: [{start}, {end}) / {len(opencood_dataset)}")

    data_loader = DataLoader(dataset_for_loader,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat_short = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat_middle = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat_long = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
                
    infer_info = opt.fusion_method + opt.note


    for i, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            _apply_token_ablation(
                batch_data['ego'],
                mode=opt.token_ablation,
                seed=opt.seed,
                step=i,
            )

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(
                    batch_data, model, opencood_dataset
                )
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(
                    batch_data, model, opencood_dataset
                )
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(
                    batch_data, model, opencood_dataset
                )
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(
                    batch_data, model, opencood_dataset
                )
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(
                    batch_data, model, opencood_dataset, single_gt=True
                )
            else:
                raise NotImplementedError(
                    'Only single, no, no_w_uncertainty, early, late and intermediate'
                    'fusion is supported.'
                )

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            # 3-class postprocessor returns (N,2) with [score, label]; extract score only
            if pred_score is not None and pred_score.dim() == 2:
                pred_score = pred_score[:, 0]
            
            for iou_threshold in [0.3, 0.5, 0.7]:
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        iou_threshold)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat_short,
                                        iou_threshold, 
                                        left_range=0,
                                        right_range=30)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat_middle,
                                        iou_threshold, 
                                        left_range=30,
                                        right_range=50)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat_long,
                                        iou_threshold,
                                        left_range=50,
                                        right_range=100)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
                
                # vis_feat_save_path = os.path.join(opt.model_dir, f'feat_vis_{infer_info}')
                # vis_utils.visualize_feature_distribution(infer_result, vis_feat_save_path, i)


        torch.cuda.empty_cache()
    eval_utils.eval_final_results(result_stat_short,
                                  opt.model_dir, infer_info=f"{infer_info}_short")
    eval_utils.eval_final_results(result_stat_middle,
                                  opt.model_dir, infer_info=f"{infer_info}_middle")
    eval_utils.eval_final_results(result_stat_long,
                                  opt.model_dir, infer_info=f"{infer_info}_long")
    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)

if __name__ == '__main__':
    main()
