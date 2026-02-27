'''
-*- coding: utf-8 -*-
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
License: TDG-Attribution-NonCommercial-NoDistrib

intermediate heter fusion dataset

Note that for DAIR-V2X dataset,
Each agent should retrieve the objects itself, and merge them by iou, 
instead of using the cooperative label.
''' 

import random
import math
from collections import OrderedDict
from typing import Tuple
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.common_utils import merge_features_to_dict, compute_iou, convert_format
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.common_utils import read_json
from opencood.utils.heter_utils import Adaptor


def _dinov2_patch_uv(
    orig_w: int,
    orig_h: int,
    num_tokens: int,
    img_size: int,
) -> Tuple[np.ndarray, int]:
    """
    Reproduce `build_token_cache_dairv2x.py:DINOv2PatchTokenizer.tokens` uv ordering.
    """
    grid = int(round(math.sqrt(float(num_tokens))))
    if grid * grid != int(num_tokens):
        raise ValueError(f"num_tokens must be a square for dinov2_patches uv: {num_tokens}")

    patch_size = float(img_size) / float(grid)
    xs = (np.arange(grid, dtype=np.float32) + 0.5) * patch_size
    ys = (np.arange(grid, dtype=np.float32) + 0.5) * patch_size
    xv, yv = np.meshgrid(xs, ys)
    uv_resized = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.float32)

    scale_u = float(orig_w) / float(img_size)
    scale_v = float(orig_h) / float(img_size)
    uv = np.stack([uv_resized[:, 0] * scale_u, uv_resized[:, 1] * scale_v], axis=1).astype(np.float32)
    return uv, grid


def _token_depth_from_lidar(
    lidar_xyz: np.ndarray,
    intrinsic: np.ndarray,
    lidar_to_camera: np.ndarray,
    grid: int,
    orig_w: int,
    orig_h: int,
    depth_mode: str,
    min_z: float,
    max_z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-token depth (camera Z) from LiDAR points as privileged supervision.
    The token grid is square (grid x grid) and aligned to a (img_size x img_size) resize
    with per-axis scaling back to (orig_w x orig_h).
    """
    pts = np.asarray(lidar_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"lidar_xyz must be [N,3], got {pts.shape}")

    num_tokens = int(grid) * int(grid)
    depth = np.zeros((num_tokens,), dtype=np.float32)
    found = np.zeros((num_tokens,), dtype=bool)
    if pts.shape[0] == 0:
        return depth, found

    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)  # [N,4]
    cam = (lidar_to_camera.astype(np.float32) @ pts_h.T).T[:, :3]  # [N,3]
    z = cam[:, 2].astype(np.float32)
    valid = (z > float(min_z)) & (z < float(max_z))
    cam = cam[valid]
    z = z[valid]
    if cam.shape[0] == 0:
        return depth, found

    pix = (intrinsic.astype(np.float32) @ cam.T).T  # [N,3]
    u = (pix[:, 0] / pix[:, 2]).astype(np.float32)
    v = (pix[:, 1] / pix[:, 2]).astype(np.float32)

    patch_w = float(orig_w) / float(grid)
    patch_h = float(orig_h) / float(grid)
    col = np.floor(u / patch_w).astype(np.int32)
    row = np.floor(v / patch_h).astype(np.int32)
    in_img = (col >= 0) & (col < grid) & (row >= 0) & (row < grid)
    if not np.any(in_img):
        return depth, found

    u = u[in_img]
    v = v[in_img]
    z = z[in_img]
    col = col[in_img]
    row = row[in_img]
    patch_id = (row * grid + col).astype(np.int32)

    depth_mode = str(depth_mode).strip().lower()
    if depth_mode == "min":
        buf = np.full((num_tokens,), np.inf, dtype=np.float32)
        np.minimum.at(buf, patch_id, z.astype(np.float32))
        found = np.isfinite(buf) & (buf < np.inf)
        depth[found] = buf[found].astype(np.float32)
        return depth, found

    if depth_mode == "center_nearest":
        u_c = (col.astype(np.float32) + 0.5) * patch_w
        v_c = (row.astype(np.float32) + 0.5) * patch_h
        d2 = (u - u_c) ** 2 + (v - v_c) ** 2
        order = np.lexsort((d2, patch_id))
        patch_sorted = patch_id[order]
        z_sorted = z[order]
        uniq, first_idx = np.unique(patch_sorted, return_index=True)
        depth[uniq.astype(np.int64)] = z_sorted[first_idx].astype(np.float32)
        found[uniq.astype(np.int64)] = True
        return depth, found

    raise ValueError(f"Unknown depth_mode={depth_mode}. Supported: center_nearest|min.")


def getIntermediateheterFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class IntermediateheterFusionDataset(cls):
        def __init__(self, params, visualize, train=True, calibrate=False):
            super().__init__(params, visualize, train, calibrate)
            # intermediate and supervise single
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            self.proj_first = False if 'proj_first' not in params['fusion']['args']\
                                         else params['fusion']['args']['proj_first']

            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.heterogeneous = True
            self.modality_assignment = None if ('assignment_path' not in params['heter'] or params['heter']['assignment_path'] is None) \
                                            else read_json(params['heter']['assignment_path'])
            
            self.ego_modality = params['heter']['ego_modality'] # "m1" or "m1&m2" or "m3"

            self.modality_name_list = list(params['heter']['modality_setting'].keys())
            self.sensor_type_dict = OrderedDict()

            lidar_channels_dict = params['heter'].get('lidar_channels_dict', OrderedDict())
            mapping_dict = params['heter']['mapping_dict']
            cav_preference = params['heter'].get("cav_preference", None)

            self.adaptor = Adaptor(self.ego_modality,
                                   self.modality_name_list,
                                   self.modality_assignment,
                                   lidar_channels_dict,
                                   mapping_dict,
                                   cav_preference,
                                   train,
                                   calibrate)

            for modality_name, modal_setting in params['heter']['modality_setting'].items():
                self.sensor_type_dict[modality_name] = modal_setting['sensor_type']
                if modal_setting['sensor_type'] == 'lidar':
                    setattr(self, f"pre_processor_{modality_name}", build_preprocessor(modal_setting['preprocess'], train))

                elif modal_setting['sensor_type'] == 'camera':
                    setattr(self, f"data_aug_conf_{modality_name}", modal_setting['data_aug_conf'])

                else:
                    raise("Not support this type of sensor")

            self.reinitialize()
                

            self.kd_flag = params.get('kd_flag', False)

            # Protocol distillation (privileged teacher LiDAR for camera modality).
            proto_cfg = params.get("model", {}).get("args", {}).get("protocol", {})
            proto_cfg = proto_cfg if isinstance(proto_cfg, dict) else {}
            distill_cfg = proto_cfg.get("distill", {})
            distill_cfg = distill_cfg if isinstance(distill_cfg, dict) else {}

            self.protocol_distill_enabled = bool(distill_cfg.get("enabled", False))
            apply_to = distill_cfg.get("apply_to", proto_cfg.get("apply_to", []))
            if apply_to is None:
                apply_to = []
            if not isinstance(apply_to, (list, tuple)):
                apply_to = [apply_to]
            self.protocol_distill_apply_to = set(str(v) for v in apply_to)

            # Which lidar modality's voxelizer we use to preprocess teacher point clouds (default: ego lidar modality).
            self.protocol_teacher_modality = str(distill_cfg.get("teacher_modality", self.ego_modality))

            # Online depth supervision for camera tokens (privileged LiDAR, training only).
            depth_cfg = proto_cfg.get("depth_adaptor", {})
            depth_cfg = depth_cfg if isinstance(depth_cfg, dict) else {}

            self.token_depth_enabled = bool(depth_cfg.get("enabled", False))
            depth_apply_to = depth_cfg.get("apply_to", proto_cfg.get("apply_to", []))
            if depth_apply_to is None:
                depth_apply_to = []
            if not isinstance(depth_apply_to, (list, tuple)):
                depth_apply_to = [depth_apply_to]
            self.token_depth_apply_to = set(str(v) for v in depth_apply_to)

            self.token_depth_mode = str(depth_cfg.get("label", depth_cfg.get("depth_mode", "center_nearest")))
            self.token_depth_mode = self.token_depth_mode.strip().lower()
            self.token_depth_img_size = int(depth_cfg.get("dinov2_img_size", 224))
            self.token_depth_min_z = float(depth_cfg.get("min_depth", 0.1))
            self.token_depth_max_z = float(depth_cfg.get("max_depth", 80.0))

            self.box_align = False
            if "box_align" in params:
                self.box_align = True
                self.stage1_result_path = params['box_align']['train_result'] if train else params['box_align']['val_result']
                self.stage1_result = read_json(self.stage1_result_path)
                self.box_align_args = params['box_align']['args']
                


        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_pose : list, length 6
                The ego vehicle lidar pose under world coordinate.
            ego_pose_clean : list, length 6
                only used for gt box generation

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

            # calculate the transformation matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],
                        ego_pose) # T_ego_cav
            transformation_matrix_clean = \
                x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                        ego_pose_clean)
            
            modality_name = selected_cav_base['modality_name']
            sensor_type = self.sensor_type_dict[modality_name]
            need_teacher_lidar = (
                sensor_type == "camera"
                and self.train
                and self.protocol_distill_enabled
                and modality_name in self.protocol_distill_apply_to
            )
            need_token_depth = (
                sensor_type == "camera"
                and self.train
                and self.token_depth_enabled
                and modality_name in self.token_depth_apply_to
            )
            need_token_uv = (
                sensor_type == "camera"
                and self.token_depth_enabled
                and modality_name in self.token_depth_apply_to
            )
            need_privileged_lidar = need_teacher_lidar or need_token_depth
            lidar_np_local = None

            # lidar
            if sensor_type == "lidar" or self.visualize or need_privileged_lidar:
                # process lidar
                if need_privileged_lidar and self.proj_first:
                    raise ValueError(
                        "protocol.distill/depth_adaptor expects fusion.args.proj_first=false so that privileged LiDAR "
                        "and camera tokens stay in the agent local frame."
                    )
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)
                if need_privileged_lidar and sensor_type == "camera":
                    lidar_np_local = lidar_np
                # project the lidar to ego space
                # x,y,z in ego space
                projected_lidar = \
                    box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                                transformation_matrix)
                if self.proj_first:
                    lidar_np[:, :3] = projected_lidar

                if self.visualize:
                    # filter lidar
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

                if self.kd_flag:
                    lidar_proj_np = copy.deepcopy(lidar_np)
                    lidar_proj_np[:,:3] = projected_lidar

                    selected_cav_processed.update({'projected_lidar': lidar_proj_np})

                    # 2023.8.31, to correct discretization errors. Just replace one point to avoid empty voxels. need fix later.
                    lidar_proj_np[np.random.randint(0, lidar_proj_np.shape[0]),:3] = np.array([0,0,0]) 
                    processed_lidar_proj = eval(f"self.pre_processor_{modality_name}").preprocess(lidar_proj_np)
                    selected_cav_processed.update({f'processed_features_{modality_name}_proj': processed_lidar_proj})

                if sensor_type == "lidar":
                    processed_lidar = eval(f"self.pre_processor_{modality_name}").preprocess(lidar_np)
                    selected_cav_processed.update({f'processed_features_{modality_name}': processed_lidar})
                elif need_teacher_lidar:
                    teacher_modality = self.protocol_teacher_modality
                    if teacher_modality not in self.modality_name_list:
                        raise ValueError(
                            f"protocol.distill.teacher_modality={teacher_modality} is not in heter.modality_setting"
                        )
                    if self.sensor_type_dict.get(teacher_modality, None) != "lidar":
                        raise ValueError(
                            f"protocol.distill.teacher_modality={teacher_modality} must be a lidar modality, "
                            f"but got sensor_type={self.sensor_type_dict.get(teacher_modality, None)}"
                        )
                    teacher_processed_lidar = eval(f"self.pre_processor_{teacher_modality}").preprocess(lidar_np)
                    selected_cav_processed.update(
                        {f"processed_features_{modality_name}_teacher_lidar": teacher_processed_lidar}
                    )

            # generate targets label single GT, note the reference pose is itself.
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], selected_cav_base['params']['lidar_pose']
            )
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
            )
            selected_cav_processed.update({
                                "single_label_dict": label_dict,
                                "single_object_bbx_center": object_bbx_center,
                                "single_object_bbx_mask": object_bbx_mask})

            # camera
            if sensor_type == "camera":
                camera_data_list = selected_cav_base["camera_data"]
                params = selected_cav_base["params"]
                imgs = []
                rots = []
                trans = []
                intrins = []
                extrinsics = []
                post_rots = []
                post_trans = []

                for idx, img in enumerate(camera_data_list):
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(
                        camera_to_lidar[:3, :3]
                    )  # R_wc, we consider world-coord is the lidar-coord
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # depth
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None

                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        eval(f"self.data_aug_conf_{modality_name}"), self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src,
                        post_rot,
                        post_tran,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # decouple RGB and Depth

                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)
                    

                selected_cav_processed.update(
                    {
                    f"image_inputs_{modality_name}": 
                        {
                            "imgs": torch.stack(imgs), # [Ncam, 3or4, H, W]
                            "intrins": torch.stack(intrins),
                            "extrinsics": torch.stack(extrinsics),
                            "rots": torch.stack(rots),
                            "trans": torch.stack(trans),
                            "post_rots": torch.stack(post_rots),
                            "post_trans": torch.stack(post_trans),
                        }
                    }
                )
                if "token_data" in selected_cav_base:
                    token_data = selected_cav_base["token_data"]
                    selected_cav_processed[f"image_inputs_{modality_name}"].update(
                        {
                            "token_xyz": torch.from_numpy(token_data["xyz"]).float(),
                            "token_feat": torch.from_numpy(token_data["feat"]).float(),
                            "token_mask": torch.from_numpy(token_data["mask"]).bool(),
                        }
                    )
                    if need_token_uv:
                        num_tokens = int(token_data["feat"].shape[0])
                        if len(camera_data_list) > 0 and hasattr(camera_data_list[0], "size"):
                            orig_w, orig_h = camera_data_list[0].size
                        else:
                            aug_conf = eval(f"self.data_aug_conf_{modality_name}")
                            orig_h = int(aug_conf.get("H", 1080))
                            orig_w = int(aug_conf.get("W", 1920))

                        token_uv, grid = _dinov2_patch_uv(
                            orig_w=int(orig_w),
                            orig_h=int(orig_h),
                            num_tokens=num_tokens,
                            img_size=int(self.token_depth_img_size),
                        )
                        selected_cav_processed[f"image_inputs_{modality_name}"].update(
                            {"token_uv": torch.from_numpy(token_uv).float()}
                        )

                        if need_token_depth:
                            if lidar_np_local is None:
                                raise KeyError(
                                    "token depth supervision is enabled, but lidar_np is missing for this camera CAV. "
                                    "Make sure input_source includes 'lidar' and DAIR-V2X provides infra LiDAR in "
                                    "training."
                                )
                            cam_to_lidar, camera_intrinsic = self.get_ext_int(params, 0)
                            lidar_to_camera = np.linalg.inv(cam_to_lidar.astype(np.float32))

                            depth_gt, depth_found = _token_depth_from_lidar(
                                lidar_xyz=lidar_np_local[:, :3].astype(np.float32),
                                intrinsic=camera_intrinsic.astype(np.float32),
                                lidar_to_camera=lidar_to_camera,
                                grid=int(grid),
                                orig_w=int(orig_w),
                                orig_h=int(orig_h),
                                depth_mode=str(self.token_depth_mode),
                                min_z=float(self.token_depth_min_z),
                                max_z=float(self.token_depth_max_z),
                            )
                            selected_cav_processed[f"image_inputs_{modality_name}"].update(
                                {
                                    "token_depth_gt": torch.from_numpy(depth_gt).float(),
                                    "token_depth_mask": torch.from_numpy(depth_found).bool(),
                                }
                            )

            # anchor box
            selected_cav_processed.update({"anchor_box": self.anchor_box})

            # note the reference pose ego
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                        ego_pose_clean)

            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                    "object_bbx_mask": object_bbx_mask,
                    "object_ids": object_ids,
                    'transformation_matrix': transformation_matrix,
                    'transformation_matrix_clean': transformation_matrix_clean
                }
            )


            return selected_cav_processed

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break
                
            assert cav_id == list(base_data_dict.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            
            input_list_m1 = [] # can contain lidar or camera
            input_list_m2 = []
            input_list_m3 = []
            input_list_m4 = []
            teacher_input_list_m1 = []
            teacher_input_list_m2 = []
            teacher_input_list_m3 = []
            teacher_input_list_m4 = []

            agent_modality_list = []
            object_stack = []
            object_id_stack = []
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            exclude_agent = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []
            projected_lidar_clean_list = [] # disconet

            if self.visualize or self.kd_flag:
                projected_lidar_stack = []
                input_list_m1_proj = [] # 2023.8.31 to correct discretization errors with kd flag
                input_list_m2_proj = []
                input_list_m3_proj = []
                input_list_m4_proj = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)

                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    exclude_agent.append(cav_id)
                    continue
                
                # if modality not match
                if self.adaptor.unmatched_modality(selected_cav_base['modality_name']):
                    exclude_agent.append(cav_id)
                    continue

                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
                cav_id_list.append(cav_id)   

            if len(cav_id_list) == 0:
                return None

            for cav_id in exclude_agent:
                base_data_dict.pop(cav_id)

            ########## Updated by Yifan Lu 2022.1.26 ############
            # box align to correct pose.
            # stage1_content contains all agent. Even out of comm range.
            if self.box_align and str(idx) in self.stage1_result.keys():
                from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
                stage1_content = self.stage1_result[str(idx)]
                if stage1_content is not None:
                    all_agent_id_list = stage1_content['cav_id_list'] # include those out of range
                    all_agent_corners_list = stage1_content['pred_corner3d_np_list']
                    all_agent_uncertainty_list = stage1_content['uncertainty_np_list']

                    cur_agent_id_list = cav_id_list
                    cur_agent_pose = [base_data_dict[cav_id]['params']['lidar_pose'] for cav_id in cav_id_list]
                    cur_agnet_pose = np.array(cur_agent_pose)
                    cur_agent_in_all_agent = [all_agent_id_list.index(cur_agent) for cur_agent in cur_agent_id_list] # indexing current agent in `all_agent_id_list`

                    pred_corners_list = [np.array(all_agent_corners_list[cur_in_all_ind], dtype=np.float64) 
                                            for cur_in_all_ind in cur_agent_in_all_agent]
                    uncertainty_list = [np.array(all_agent_uncertainty_list[cur_in_all_ind], dtype=np.float64) 
                                            for cur_in_all_ind in cur_agent_in_all_agent]

                    if sum([len(pred_corners) for pred_corners in pred_corners_list]) != 0:
                        refined_pose = box_alignment_relative_sample_np(pred_corners_list,
                                                                        cur_agnet_pose, 
                                                                        uncertainty_list=uncertainty_list, 
                                                                        **self.box_align_args)
                        cur_agnet_pose[:,[0,1,4]] = refined_pose 

                        for i, cav_id in enumerate(cav_id_list):
                            lidar_pose_list[i] = cur_agnet_pose[i].tolist()
                            base_data_dict[cav_id]['params']['lidar_pose'] = cur_agnet_pose[i].tolist()



            pairwise_t_matrix = \
                get_pairwise_transformation(base_data_dict,
                                                self.max_cav,
                                                self.proj_first)

            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
            
            # merge preprocessed features from different cavs into the same dict
            cav_num = len(cav_id_list)
            
            for _i, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]
                modality_name = selected_cav_base['modality_name']
                sensor_type = self.sensor_type_dict[selected_cav_base['modality_name']]

                # dynamic object center generator! for heterogeneous input
                if not self.visualize:
                    self.generate_object_center = eval(f"self.generate_object_center_{sensor_type}")
                # need discussion. In test phase, use lidar label.
                else: 
                    self.generate_object_center = self.generate_object_center_lidar

                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_cav_base)
                
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']


                if sensor_type == "lidar":
                    eval(f"input_list_{modality_name}").append(selected_cav_processed[f"processed_features_{modality_name}"])
                elif sensor_type == "camera":
                    eval(f"input_list_{modality_name}").append(selected_cav_processed[f"image_inputs_{modality_name}"])
                    teacher_key = f"processed_features_{modality_name}_teacher_lidar"
                    if teacher_key in selected_cav_processed:
                        eval(f"teacher_input_list_{modality_name}").append(selected_cav_processed[teacher_key])
                else:
                    raise
                
                agent_modality_list.append(modality_name)

                if self.visualize or self.kd_flag:
                    # heterogeneous setting do not support disconet' kd
                    projected_lidar_stack.append(
                        selected_cav_processed['projected_lidar'])
                    if sensor_type == "lidar" and self.kd_flag:
                        eval(f"input_list_{modality_name}_proj").append(selected_cav_processed[f"processed_features_{modality_name}_proj"])
                        
                
                if self.supervise_single or self.heterogeneous:
                    single_label_list.append(selected_cav_processed['single_label_dict'])
                    single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                    single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])

            # generate single view GT label
            if self.supervise_single or self.heterogeneous:
                single_label_dicts = self.post_processor.collate_batch(single_label_list)
                single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
                single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                processed_data_dict['ego'].update({
                    "single_label_dict_torch": single_label_dicts,
                    "single_object_bbx_center_torch": single_object_bbx_center,
                    "single_object_bbx_mask_torch": single_object_bbx_mask,
                    })
            
            # exculude all repetitve objects, DAIR-V2X
            if self.params['fusion']['dataset'] == 'dairv2x':
                if len(object_stack) == 1:
                    object_stack = object_stack[0]
                else:
                    ego_boxes_np = object_stack[0]
                    cav_boxes_np = object_stack[1]
                    order = self.params['postprocess']['order']
                    ego_corners_np = box_utils.boxes_to_corners_3d(ego_boxes_np, order)
                    cav_corners_np = box_utils.boxes_to_corners_3d(cav_boxes_np, order)
                    ego_polygon_list = list(convert_format(ego_corners_np))
                    cav_polygon_list = list(convert_format(cav_corners_np))
                    iou_thresh = 0.05 


                    gt_boxes_from_cav = []
                    for i in range(len(cav_polygon_list)):
                        cav_polygon = cav_polygon_list[i]
                        ious = compute_iou(cav_polygon, ego_polygon_list)
                        if (ious > iou_thresh).any():
                            continue
                        gt_boxes_from_cav.append(cav_boxes_np[i])
                    
                    if len(gt_boxes_from_cav):
                        object_stack_from_cav = np.stack(gt_boxes_from_cav)
                        object_stack = np.vstack([ego_boxes_np, object_stack_from_cav])
                    else:
                        object_stack = ego_boxes_np

                unique_indices = np.arange(object_stack.shape[0])
                object_id_stack = np.arange(object_stack.shape[0])
            else:
                # exclude all repetitive objects, OPV2V-H
                unique_indices = \
                    [object_id_stack.index(x) for x in set(object_id_stack)]
                object_stack = np.vstack(object_stack)
                object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            
            for modality_name in self.modality_name_list:
                if self.sensor_type_dict[modality_name] == "lidar":
                    merged_feature_dict = merge_features_to_dict(eval(f"input_list_{modality_name}")) 
                    processed_data_dict['ego'].update({f'input_{modality_name}': merged_feature_dict}) # maybe None
                elif self.sensor_type_dict[modality_name] == "camera":
                    merged_image_inputs_dict = merge_features_to_dict(eval(f"input_list_{modality_name}"), merge='stack')
                    processed_data_dict['ego'].update({f'input_{modality_name}': merged_image_inputs_dict}) # maybe None
                    if (
                        self.train
                        and self.protocol_distill_enabled
                        and modality_name in self.protocol_distill_apply_to
                    ):
                        merged_teacher_lidar_dict = merge_features_to_dict(eval(f"teacher_input_list_{modality_name}"))
                        processed_data_dict['ego'].update(
                            {f"input_{modality_name}_teacher_lidar": merged_teacher_lidar_dict}
                        )

            if self.kd_flag:
                # heterogenous setting do not support DiscoNet's kd
                # stack_lidar_np = np.vstack(projected_lidar_stack)
                # stack_lidar_np = mask_points_by_range(stack_lidar_np,
                #                             self.params['preprocess'][
                #                                 'cav_lidar_range'])
                # stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                for modality_name in self.modality_name_list:
                    processed_data_dict['ego'].update({
                        f'input_{modality_name}_proj': merge_features_to_dict(eval(f"input_list_{modality_name}_proj")) # maybe None
                        })


            processed_data_dict['ego'].update({'agent_modality_list': agent_modality_list})

            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=self.anchor_box,
                    mask=mask)

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': self.anchor_box,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_poses_clean': lidar_poses_clean,
                'lidar_poses': lidar_poses})


            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                    np.vstack(
                        projected_lidar_stack)})


            processed_data_dict['ego'].update({'sample_idx': idx,
                                                'cav_id_list': cav_id_list})

            return processed_data_dict


        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            inputs_list_m1 = [] 
            inputs_list_m2 = []
            inputs_list_m3 = []
            inputs_list_m4 = []
            teacher_inputs_list_m1 = []
            teacher_inputs_list_m2 = []
            teacher_inputs_list_m3 = []
            teacher_inputs_list_m4 = []

            inputs_list_m1_proj = [] 
            inputs_list_m2_proj = []
            inputs_list_m3_proj = []
            inputs_list_m4_proj = []

            agent_modality_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []
            lidar_pose_list = []
            origin_lidar = []
            lidar_pose_clean_list = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # disconet
            teacher_processed_lidar_list = []
            
            ### 2022.10.10 single gt ####
            if self.supervise_single or self.heterogeneous:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []
                object_bbx_center_single = []
                object_bbx_mask_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
                lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])

                for modality_name in self.modality_name_list:
                    if ego_dict[f'input_{modality_name}'] is not None:
                        eval(f"inputs_list_{modality_name}").append(ego_dict[f'input_{modality_name}']) # OrderedDict() if empty?
                    teacher_key = f"input_{modality_name}_teacher_lidar"
                    if ego_dict.get(teacher_key, None) is not None:
                        eval(f"teacher_inputs_list_{modality_name}").append(ego_dict[teacher_key])

                agent_modality_list.extend(ego_dict['agent_modality_list'])
                
                record_len.append(ego_dict['cav_num'])
                label_dict_list.append(ego_dict['label_dict'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                if self.kd_flag:
                    # hetero setting do not support disconet' kd
                    # teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])
                    for modality_name in self.modality_name_list:
                        if ego_dict[f'input_{modality_name}_proj'] is not None:
                            eval(f"inputs_list_{modality_name}_proj").append(ego_dict[f"input_{modality_name}_proj"])

                ### 2022.10.10 single gt ####
                if self.supervise_single or self.heterogeneous:
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])
                    object_bbx_center_single.append(ego_dict['single_object_bbx_center_torch'])
                    object_bbx_mask_single.append(ego_dict['single_object_bbx_mask_torch'])


            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))


            # 2023.2.5
            for modality_name in self.modality_name_list:
                if len(eval(f"inputs_list_{modality_name}")) != 0:
                    if self.sensor_type_dict[modality_name] == "lidar":
                        merged_feature_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}"))
                        processed_lidar_torch_dict = eval(f"self.pre_processor_{modality_name}").collate_batch(merged_feature_dict)
                        output_dict['ego'].update({f'inputs_{modality_name}': processed_lidar_torch_dict})

                    elif self.sensor_type_dict[modality_name] == "camera":
                        merged_image_inputs_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}"), merge='cat')
                        output_dict['ego'].update({f'inputs_{modality_name}': merged_image_inputs_dict})

            if self.protocol_distill_enabled:
                for modality_name in self.modality_name_list:
                    if len(eval(f"teacher_inputs_list_{modality_name}")) == 0:
                        continue
                    teacher_modality = self.protocol_teacher_modality
                    merged_teacher_dict = merge_features_to_dict(eval(f"teacher_inputs_list_{modality_name}"))
                    teacher_lidar_torch_dict = eval(f"self.pre_processor_{teacher_modality}").collate_batch(
                        merged_teacher_dict
                    )
                    output_dict['ego'].update({f"inputs_{modality_name}_teacher_lidar": teacher_lidar_torch_dict})

            output_dict['ego'].update({"agent_modality_list": agent_modality_list})
            
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                     'object_bbx_mask': object_bbx_mask})

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # add pairwise_t_matrix to label dict
            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict['record_len'] = record_len
            

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'record_len': record_len,
                                    'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0],
                                    'pairwise_t_matrix': pairwise_t_matrix,
                                    'lidar_pose_clean': lidar_pose_clean,
                                    'lidar_pose': lidar_pose,
                                    'anchor_box': self.anchor_box_torch})


            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.kd_flag:
                # teacher_processed_lidar_torch_dict = \
                #     self.pre_processor.collate_batch(teacher_processed_lidar_list)
                # output_dict['ego'].update({'teacher_processed_lidar':teacher_processed_lidar_torch_dict})
                for modality_name in self.modality_name_list:
                    if len(eval(f"inputs_list_{modality_name}_proj")) != 0 and self.sensor_type_dict[modality_name] == "lidar":
                        merged_feature_proj_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}_proj"))
                        processed_lidar_torch_proj_dict = eval(f"self.pre_processor_{modality_name}").collate_batch(merged_feature_proj_dict)
                        output_dict['ego'].update({f'inputs_{modality_name}_proj': processed_lidar_torch_proj_dict})

            if self.supervise_single  or self.heterogeneous:
                output_dict['ego'].update({
                    "label_dict_single":{
                            "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                            "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                            "targets": torch.cat(targets_single, dim=0),
                            # for centerpoint
                            "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                            "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                        },
                    "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                    "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                })

            return output_dict

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            if batch[0] is None:
                return None
            output_dict = self.collate_batch_train(batch)
            if output_dict is None:
                return None

            # check if anchor box in the batch
            if batch[0]['ego']['anchor_box'] is not None:
                output_dict['ego'].update({'anchor_box':
                    self.anchor_box_torch})

            # save the transformation matrix (4, 4) to ego vehicle
            # transformation is only used in post process (no use.)
            # we all predict boxes in ego coord.
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()
            transformation_matrix_clean_torch = \
                torch.from_numpy(np.identity(4)).float()

            output_dict['ego'].update({'transformation_matrix':
                                        transformation_matrix_torch,
                                        'transformation_matrix_clean':
                                        transformation_matrix_clean_torch,})

            output_dict['ego'].update({
                "sample_idx": batch[0]['ego']['sample_idx'],
                "cav_id_list": batch[0]['ego']['cav_id_list'],
                "agent_modality_list": batch[0]['ego']['agent_modality_list']
            })

            return output_dict

        def collate_batch(self, batch):
            if self.train:
                return self.collate_batch_train(batch)
            else:
                return self.collate_batch_test(batch) # just added this so collate_batch exists

        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict, output_dict)
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor


    return IntermediateheterFusionDataset
