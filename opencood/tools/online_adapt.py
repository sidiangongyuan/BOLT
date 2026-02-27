"""
Online Test-Time Training (TTT) for base-free heterogeneous cooperative perception.

v3.1: Ego-as-Teacher distillation with enhancement signal + multi-epoch.
    - Preservation loss: distill where teacher is confident (don't hurt ego detections)
    - Enhancement loss: boost student confidence where teacher is uncertain but has
      some signal (encourage fusion to leverage neighbor info)
    - Multi-epoch: warmup epochs (train only) + final epoch (train + evaluate)

All encoders / fusion / heads remain frozen; only plugin parameters are updated.

Typical usage:
  python -m opencood.tools.online_adapt \
    --model_dir /path/to/DirectHeter_base_free_lidar_camera \
    --output_dir /path/to/output \
    --lr 1e-4 --epochs 3 --teacher_conf_thresh 0.3 \
    --boost_weight 0.1 --boost_lo 0.1 --boost_hi 0.3 \
    --plugin_adain_alpha_init_logit -10
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils
from opencood.utils.common_utils import update_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online TTT with ego-as-teacher (base-free)")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    # Data
    p.add_argument("--comm_range", type=float, default=100.0)
    p.add_argument("--use_cav", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--fuse_method", type=str, default="weighted",
                    choices=["weighted", "max", "mean", "attn", "v2xvit"],
                    help="Fusion aggregation method inside PyramidFusion")

    # Plugin config
    p.add_argument("--src_modality", type=str, default="m2")
    p.add_argument("--plugin_hidden", type=int, default=128)
    p.add_argument("--plugin_blocks", type=int, default=3)
    p.add_argument("--plugin_gn_groups", type=int, default=16)
    p.add_argument("--plugin_adain_alpha_init_logit", type=float, default=10.0)
    p.add_argument("--plugin_gate_init_logit", type=float, default=0.0)

    # Optimization
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=1,
                    help="Total epochs. epochs-1 are warmup (train only), last epoch trains+evaluates.")

    # Distillation — preservation
    p.add_argument("--cls_weight", type=float, default=1.0)
    p.add_argument("--reg_weight", type=float, default=1.0)
    p.add_argument("--dir_weight", type=float, default=0.5)
    p.add_argument("--teacher_conf_thresh", type=float, default=0.0,
                    help="Only distill where teacher confidence > thresh (0 = distill everywhere)")

    # Distillation — enhancement
    p.add_argument("--boost_weight", type=float, default=0.0,
                    help="Weight for enhancement loss (0 = disabled)")
    p.add_argument("--boost_lo", type=float, default=0.1,
                    help="Lower bound of teacher confidence for boost region")
    p.add_argument("--boost_hi", type=float, default=0.3,
                    help="Upper bound of teacher confidence for boost region")

    # Logging
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_log", action="store_true")
    p.add_argument("--no_plugin", action="store_true",
                    help="Disable plugin entirely (for no-plugin baselines)")
    p.add_argument("--train_fusion", action="store_true",
                    help="Also unfreeze fusion module params (e.g. v2xvit_levels) during TTT")
    p.add_argument("--teacher_mode", type=str, default="ego",
                    choices=["ego", "neighbor"],
                    help="Teacher source: 'ego' (default) or 'neighbor' (use neighbor's single-agent prediction)")
    p.add_argument("--score_threshold", type=float, default=None,
                    help="Override postprocessor score_threshold (default: use config value)")
    p.add_argument("--convergence_interval", type=int, default=0,
                    help="Compute running AP every N samples during eval epoch (0 = disabled)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _inject_plugin_cfg(hypes: dict, opt: argparse.Namespace) -> dict:
    hypes = copy.deepcopy(hypes)
    hypes.setdefault("model", {}).setdefault("args", {})
    hypes["model"]["args"]["plugin"] = {
        "enable": True,
        "src_modality": opt.src_modality,
        "args": {
            "in_channels": 64,
            "hidden_channels": opt.plugin_hidden,
            "num_blocks": opt.plugin_blocks,
            "gn_groups": opt.plugin_gn_groups,
            "adain_alpha_init_logit": opt.plugin_adain_alpha_init_logit,
            "gate_init_logit": opt.plugin_gate_init_logit,
        },
    }
    return hypes


def _init_result_stats():
    """Create fresh result_stat dicts for AP evaluation."""
    def _make():
        return {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
        }
    return _make(), _make(), _make(), _make()  # overall, short, middle, long


def _expand_anchor_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expand per-anchor mask [B,A,H,W] to match target channels [B,C,H,W].

    When A does not divide C evenly (e.g. multi-class cls has A=18 but reg
    has C=42), reduce mask to GCD(A,C) groups first, then expand.
    """
    from math import gcd
    a, c = mask.shape[1], target.shape[1]
    if c % a == 0:
        return mask.repeat_interleave(c // a, dim=1)
    # Reduce mask to g = gcd(a, c) groups via max-pool, then expand
    g = gcd(a, c)
    b, _, h, w = mask.shape
    reduced = mask.view(b, g, a // g, h, w).amax(dim=2)  # [B, g, H, W]
    return reduced.repeat_interleave(c // g, dim=1)


def _distillation_loss(student_out, teacher_out, opt):
    """Preservation + Enhancement distillation loss."""
    s_cls = student_out["cls_preds"]
    t_cls = teacher_out["cls_preds"].detach()
    s_reg = student_out["reg_preds"]
    t_reg = teacher_out["reg_preds"].detach()
    s_dir = student_out["dir_preds"]
    t_dir = teacher_out["dir_preds"].detach()

    t_prob = torch.sigmoid(t_cls)
    s_prob = torch.sigmoid(s_cls)

    # ── Preservation loss: distill where teacher is confident ──────────
    if opt.teacher_conf_thresh > 0:
        conf_mask = (t_prob > opt.teacher_conf_thresh).float()
        cls_loss = F.binary_cross_entropy_with_logits(
            s_cls, t_prob, reduction="none"
        )
        cls_loss = (cls_loss * conf_mask).sum() / conf_mask.sum().clamp(min=1)

        reg_mask = _expand_anchor_mask(conf_mask, s_reg)
        reg_loss = F.smooth_l1_loss(s_reg, t_reg, reduction="none")
        reg_loss = (reg_loss * reg_mask).sum() / reg_mask.sum().clamp(min=1)

        dir_mask = _expand_anchor_mask(conf_mask, s_dir)
        dir_loss = F.mse_loss(s_dir, t_dir, reduction="none")
        dir_loss = (dir_loss * dir_mask).sum() / dir_mask.sum().clamp(min=1)
    else:
        cls_loss = F.mse_loss(s_prob, t_prob)
        reg_loss = F.smooth_l1_loss(s_reg, t_reg)
        dir_loss = F.mse_loss(s_dir, t_dir)

    preserve = (opt.cls_weight * cls_loss
                + opt.reg_weight * reg_loss
                + opt.dir_weight * dir_loss)

    # ── Enhancement loss: boost student in teacher's uncertain region ──
    boost_loss_val = 0.0
    if opt.boost_weight > 0:
        boost_mask = ((t_prob > opt.boost_lo) & (t_prob <= opt.boost_hi)).float()
        n_boost = boost_mask.sum().clamp(min=1)
        # Encourage student to be more confident than teacher in these regions
        # -log(s_prob) weighted by boost_mask: pushes s_prob toward 1
        boost_loss = (-torch.log(s_prob + 1e-7) * boost_mask).sum() / n_boost
        boost_loss_val = float(boost_loss)
        total = preserve + opt.boost_weight * boost_loss
    else:
        total = preserve

    return total, float(cls_loss), float(reg_loss), float(dir_loss), boost_loss_val


def _run_teacher_forward(model, cav_content, device):
    """Ego-only teacher forward (no plugin, record_len=1)."""
    with torch.no_grad():
        saved_plugin = model.plugin_enabled
        model.plugin_enabled = False

        saved_rl = cav_content["record_len"].clone()
        saved_ml = cav_content["agent_modality_list"]

        cav_content["record_len"] = torch.ones(1, dtype=saved_rl.dtype, device=device)
        cav_content["agent_modality_list"] = saved_ml[:1]

        teacher_out = model(cav_content)

        cav_content["record_len"] = saved_rl
        cav_content["agent_modality_list"] = saved_ml
        model.plugin_enabled = saved_plugin

    return teacher_out


def _run_neighbor_teacher_forward(model, cav_content, device):
    """Neighbor-only teacher forward (no plugin, record_len=1).

    Swaps agent order so the neighbor (index 1) becomes the sole agent,
    then runs single-agent inference.  Requires record_len sum >= 2.
    """
    with torch.no_grad():
        saved_plugin = model.plugin_enabled
        model.plugin_enabled = False

        saved_rl = cav_content["record_len"].clone()
        saved_ml = cav_content["agent_modality_list"]
        total_agents = int(saved_rl.sum())

        if total_agents < 2:
            # Fallback: only ego available, use ego as teacher
            return _run_teacher_forward(model, cav_content, device)

        # Save and reorder spatial tensors: move neighbor (idx 1) to position 0
        saved_tensors = {}
        for key in list(cav_content.keys()):
            val = cav_content[key]
            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == total_agents:
                saved_tensors[key] = val
                cav_content[key] = val[1:2]  # neighbor slice only

        cav_content["record_len"] = torch.ones(1, dtype=saved_rl.dtype, device=device)
        cav_content["agent_modality_list"] = saved_ml[1:2]

        teacher_out = model(cav_content)

        # Restore
        for key, val in saved_tensors.items():
            cav_content[key] = val
        cav_content["record_len"] = saved_rl
        cav_content["agent_modality_list"] = saved_ml
        model.plugin_enabled = saved_plugin

    return teacher_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    opt = _parse_args()
    _set_seed(opt.seed)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    out_dir = opt.output_dir or (opt.model_dir.rstrip("/") + "_online_ttt")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load config ────────────────────────────────────────────────────────
    class _Opt:
        model_dir = opt.model_dir
        fusion_method = "intermediate"

    hypes = yaml_utils.load_yaml(None, _Opt())
    if not opt.no_plugin:
        hypes = _inject_plugin_cfg(hypes, opt)

    hypes["validate_dir"] = hypes["test_dir"]
    hypes["comm_range"] = float(opt.comm_range)
    hypes["use_cav"] = int(opt.use_cav)
    fusion_core = str(hypes["fusion"]["core_method"])
    if fusion_core == "intermediateheter":
        hypes["fusion"]["core_method"] = "intermediateheterinfer"
    elif fusion_core != "intermediateheterinfer":
        hypes["fusion"]["core_method"] = fusion_core + "infer"
    hypes = update_dict(hypes, {"ego_modality": "m1"})

    # Fix mapping_dict: training may save 'none' for unused modalities in v2v mode
    md = hypes.get("heter", {}).get("mapping_dict", {})
    ego_mod = hypes.get("heter", {}).get("ego_modality", "m1")
    if any(v == "none" for v in md.values()):
        hypes["heter"]["mapping_dict"] = {k: (ego_mod if v == "none" else v) for k, v in md.items()}

    # Ensure add_data_extension includes bev_visibility.png when camera modality is present
    ms = hypes.get("heter", {}).get("modality_setting", {})
    has_camera = any(
        isinstance(v, dict) and v.get("sensor_type") == "camera"
        for v in ms.values()
    )
    if has_camera:
        ext = hypes.get("add_data_extension", [])
        if "bev_visibility.png" not in ext:
            hypes["add_data_extension"] = ext + ["bev_visibility.png"]

    # Inject fuse_method into fusion backbone config
    if "model" in hypes and "args" in hypes["model"] and "fusion_backbone" in hypes["model"]["args"]:
        hypes["model"]["args"]["fusion_backbone"]["fuse_method"] = opt.fuse_method

    yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    parser_func = None
    for name, func in yaml_utils_lib.__dict__.items():
        if name == hypes["yaml_parser"]:
            parser_func = func
            break
    if parser_func is None:
        raise RuntimeError(f"Cannot find yaml_parser={hypes.get('yaml_parser')}")
    hypes = parser_func(hypes)

    # ── Build model ────────────────────────────────────────────────────────
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.to(device)

    if opt.no_plugin:
        # Disable plugin entirely for baseline evaluation
        if hasattr(model, "plugin_enabled"):
            model.plugin_enabled = False
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        print(f"[online-ttt] no_plugin mode — evaluation only (fuse_method={opt.fuse_method})")
        optimizer = None
    else:
        for name, p in model.named_parameters():
            trainable = name.startswith("plugin")
            if opt.train_fusion and "v2xvit_levels" in name:
                trainable = True
            p.requires_grad_(trainable)

        train_params = [p for p in model.parameters() if p.requires_grad]
        if len(train_params) == 0:
            raise RuntimeError("No trainable plugin parameters found.")

        n_params = sum(p.numel() for p in train_params)
        print(f"[online-ttt] trainable params: {n_params:,} (train_fusion={opt.train_fusion})")
        print(f"[online-ttt] epochs={opt.epochs} (warmup={max(opt.epochs-1,0)}, eval=last)")
        print(f"[online-ttt] teacher_conf_thresh={opt.teacher_conf_thresh}")
        print(f"[online-ttt] boost_weight={opt.boost_weight} boost_range=[{opt.boost_lo}, {opt.boost_hi}]")
        print(f"[online-ttt] teacher_mode={opt.teacher_mode}")

        optimizer = torch.optim.Adam(train_params, lr=opt.lr, weight_decay=opt.weight_decay)

        model.eval()
        if hasattr(model, "plugin"):
            model.plugin.train()

    # ── Build dataset / loader ─────────────────────────────────────────────
    if opt.score_threshold is not None:
        hypes["postprocess"]["target_args"]["score_threshold"] = opt.score_threshold
        print(f"[online-ttt] override score_threshold={opt.score_threshold}")
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    total_samples = len(dataset)
    total_epochs = 1 if opt.no_plugin else max(opt.epochs, 1)
    convergence_log = []  # [(sample_idx, ap30, ap50, ap70), ...]

    # ── Multi-epoch loop ───────────────────────────────────────────────────
    for epoch in range(total_epochs):
        is_eval_epoch = (epoch == total_epochs - 1)
        tag = "eval" if is_eval_epoch else "warmup"

        if is_eval_epoch:
            result_stat, result_stat_short, result_stat_middle, result_stat_long = _init_result_stats()

        running_loss = 0.0
        print(f"\n[online-ttt] epoch {epoch+1}/{total_epochs} ({tag}) — {total_samples} samples")

        for i, batch_data in enumerate(loader):
            if batch_data is None:
                continue

            batch_data = train_utils.to_device(batch_data, device)
            cav_content = batch_data["ego"]

            if opt.no_plugin:
                # ── No-plugin baseline: inference only ────────────────
                with torch.no_grad():
                    student_out = model(cav_content)
            else:
                # ── Teacher forward ────────────────────────────────────────
                if opt.teacher_mode == "neighbor":
                    teacher_out = _run_neighbor_teacher_forward(model, cav_content, device)
                else:
                    teacher_out = _run_teacher_forward(model, cav_content, device)

                # ── Student forward: full fusion with plugin ───────────────
                student_out = model(cav_content)

                # ── Loss + update ──────────────────────────────────────────
                loss, cls_l, reg_l, dir_l, boost_l = _distillation_loss(
                    student_out, teacher_out, opt
                )

                if loss.requires_grad:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if opt.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(train_params, opt.grad_clip)
                    optimizer.step()

                running_loss += float(loss)

            # ── Evaluate on last epoch only ────────────────────────────
            if is_eval_epoch:
                with torch.no_grad():
                    output_dict = {"ego": student_out}
                    post_ret = dataset.post_process(
                        batch_data, output_dict
                    )
                    pred_box_tensor, pred_score, gt_box_tensor = post_ret[0], post_ret[1], post_ret[2]
                    # 3-class post_process returns score_labels [N,2]; extract scores only
                    if pred_score is not None and pred_score.dim() == 2:
                        pred_score = pred_score[:, 0]
                    for iou_thresh in [0.3, 0.5, 0.7]:
                        eval_utils.caluclate_tp_fp(
                            pred_box_tensor, pred_score, gt_box_tensor,
                            result_stat, iou_thresh
                        )
                        eval_utils.caluclate_tp_fp(
                            pred_box_tensor, pred_score, gt_box_tensor,
                            result_stat_short, iou_thresh,
                            left_range=0, right_range=30
                        )
                        eval_utils.caluclate_tp_fp(
                            pred_box_tensor, pred_score, gt_box_tensor,
                            result_stat_middle, iou_thresh,
                            left_range=30, right_range=50
                        )
                        eval_utils.caluclate_tp_fp(
                            pred_box_tensor, pred_score, gt_box_tensor,
                            result_stat_long, iou_thresh,
                            left_range=50, right_range=100
                        )

                # ── Convergence checkpoint ─────────────────────────────
                if opt.convergence_interval > 0 and (i + 1) % opt.convergence_interval == 0:
                    import copy as _copy
                    _snap = _copy.deepcopy(result_stat)
                    _a30, _, _ = eval_utils.calculate_ap(_snap, 0.3)
                    _a50, _, _ = eval_utils.calculate_ap(_snap, 0.5)
                    _a70, _, _ = eval_utils.calculate_ap(_snap, 0.7)
                    convergence_log.append((i + 1, float(_a30), float(_a50), float(_a70)))
                    print(f"[convergence] sample {i+1}: AP@30={_a30:.4f} AP@50={_a50:.4f} AP@70={_a70:.4f}")

            # ── Logging ────────────────────────────────────────────────
            if not opt.no_plugin and opt.log_interval > 0 and (i + 1) % opt.log_interval == 0:
                avg_loss = running_loss / opt.log_interval
                boost_str = f" boost={boost_l:.4f}" if opt.boost_weight > 0 else ""
                print(
                    f"[online-ttt] e{epoch+1} sample {i+1}/{total_samples} "
                    f"loss={avg_loss:.4f} (cls={cls_l:.4f} reg={reg_l:.4f} "
                    f"dir={dir_l:.4f}{boost_str})"
                )
                running_loss = 0.0

    # ── Final evaluation ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    for tag, stat in [
        ("short", result_stat_short),
        ("middle", result_stat_middle),
        ("long", result_stat_long),
        ("overall", result_stat),
    ]:
        ap30, _, _ = eval_utils.calculate_ap(stat, 0.3)
        ap50, _, _ = eval_utils.calculate_ap(stat, 0.5)
        ap70, _, _ = eval_utils.calculate_ap(stat, 0.7)
        print(
            f"{tag:>8s} The Average Precision at IOU 0.3 is {ap30:.4f}, "
            f"The Average Precision at IOU 0.5 is {ap50:.4f}, "
            f"The Average Precision at IOU 0.7 is {ap70:.4f}"
        )

    eval_utils.eval_final_results(result_stat, out_dir, "online_ttt")

    # ── Save convergence log ──────────────────────────────────────────────
    if convergence_log:
        conv_path = os.path.join(out_dir, "convergence.json")
        with open(conv_path, "w") as f:
            json.dump(convergence_log, f)
        print(f"[convergence] saved {len(convergence_log)} checkpoints to {conv_path}")

    # ── Save checkpoint ────────────────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, "net_epoch_bestval_at0.pth")
    torch.save(model.state_dict(), ckpt_path)

    cfg_path = os.path.join(out_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        import yaml
        yaml.dump(hypes, f, default_flow_style=False, allow_unicode=True)

    if opt.save_log:
        with open(os.path.join(out_dir, "adapt_log.json"), "w") as f:
            json.dump(
                {
                    "model_dir": opt.model_dir,
                    "out_dir": out_dir,
                    "loss_type": "distillation_ego_teacher_v3.1",
                    "total_samples": total_samples,
                    "epochs": opt.epochs,
                    "lr": opt.lr,
                    "teacher_conf_thresh": opt.teacher_conf_thresh,
                    "boost_weight": opt.boost_weight,
                    "boost_lo": opt.boost_lo,
                    "boost_hi": opt.boost_hi,
                    "plugin_adain_alpha_init_logit": opt.plugin_adain_alpha_init_logit,
                    "teacher_mode": opt.teacher_mode,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(f"\nSaved checkpoint → {ckpt_path}")
    print(f"Saved config     → {cfg_path}")
    print(f"Saved eval       → {out_dir}/eval.yaml")


if __name__ == "__main__":
    main()
