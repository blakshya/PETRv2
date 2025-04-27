# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import os

@DATASETS.register_module()
class AdaptiveNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset with multi-frame temporal support.
    
    This implementation extends NuScenesDataset to support loading multiple consecutive frames
    with temporal consistency tracking for adaptive weighting between frames.
    """

    def __init__(self, num_frames=3, **kwargs):
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.dummy_stats = {'total': 0, 'per_offset': {i:0 for i in range(1, num_frames)}}
    
    def prepare_train_data(self, index):
        """Load data with multiple previous frames for training"""
        input_dict = self.get_data_info(index)

        if 'bbox3d_fields' not in input_dict:
            input_dict['bbox3d_fields'] = []
        
        # Load previous frames
        prev_frames = []
        for offset in range(1, self.num_frames):
            prev_idx = index - offset
            valid_frame = False
            
            if prev_idx >= 0:
                try:
                    prev_info = self.get_data_info(prev_idx)
                    time_diff = abs(input_dict['timestamp'] - prev_info['timestamp'])
                    # Check if frames are temporally consistent
                    if time_diff <= 0.5:
                        valid_frame = True
                        prev_frames.append(prev_info)
                except:
                    valid_frame = False
            
            if not valid_frame:
                # Create dummy frame
                dummy = copy.deepcopy(input_dict)
                dummy['timestamp'] -= offset * 0.5
                # Clear sweeps for dummy frames
                dummy['sweeps'] = []
                prev_frames.append(dummy)
                self.dummy_stats['total'] += 1
                self.dummy_stats['per_offset'][offset] += 1
        
        input_dict['prev_frames'] = prev_frames
        return self.pipeline(input_dict)
    
    def prepare_test_data(self, index):
        """Load data for testing with previous frames"""
        input_dict = self.get_data_info(index)

        if 'bbox3d_fields' not in input_dict:
            input_dict['bbox3d_fields'] = []
        
        # Also load previous frames for testing
        prev_frames = []
        for offset in range(1, self.num_frames):
            prev_idx = index - offset
            valid_frame = False
            
            if prev_idx >= 0:
                try:
                    prev_info = self.get_data_info(prev_idx)
                    time_diff = abs(input_dict['timestamp'] - prev_info['timestamp'])
                    # Check temporal consistency
                    if time_diff <= 0.5:
                        valid_frame = True
                        prev_frames.append(prev_info)
                except:
                    valid_frame = False
            
            if not valid_frame:
                # Create dummy frame
                dummy = copy.deepcopy(input_dict)
                dummy['timestamp'] -= offset * 0.5
                # Clear sweeps for dummy frames
                dummy['sweeps'] = []
                prev_frames.append(dummy)
        
        input_dict['prev_frames'] = prev_frames
        return self.pipeline(input_dict)

    def get_data_info(self, index):
        """Get data info according to the given index.
        
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data preprocessing pipelines.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],  # Keep original sweep handling
            timestamp=info['timestamp'] / 1e6,
        )

        # Add scene token if available in original data
        if 'scene_token' in info:
            input_dict['scene_token'] = info['scene_token']

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)  ###The extrinsics mean the tranformation from lidar to camera. If anyone want to use the extrinsics as sensor to lidar, please use np.linalg.inv(lidar2cam_rt.T) and modify the ResizeCropFlipImage and LoadMultiViewImageFromMultiSweepsFiles.
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics 
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        return input_dict
