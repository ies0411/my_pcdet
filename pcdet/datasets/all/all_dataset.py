import copy
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import SharedArray
import torch.utils.data as torch_data

from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..argo2.argo2_dataset import Argo2Dataset
from ..custom.custom_dataset import CustomDataset
from ..dataset import DatasetTemplate
from ..kitti import kitti_utils
from ..kitti.kitti_dataset import KittiDataset
from ..nuscenes.nuscenes_dataset import NuScenesDataset
from ..waymo.waymo_dataset import WaymoDataset
from ..once.once_dataset import ONCEDataset
from ..lyft.lyft_dataset import LyftDataset

class AllinOneDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
    ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        print(class_names['class_names'])
        super().__init__(
            dataset_cfg=dataset_cfg[class_names['class_names']],
            class_names=class_names[class_names['class_names']],
            training=training,
            root_path=root_path,
            logger=logger,
        )


        self.kitti_dataset = (
            None
            if class_names['KITTI'] is False
            else KittiDataset(
                dataset_cfg=dataset_cfg.KITTI,
                class_names=class_names['KITTI'],
                training=training,
                root_path=root_path,
                logger=logger,
            )
        )
        # self.kitti_dataset = KittiDataset(
        #     dataset_cfg=dataset_cfg.KITTI,
        #     class_names=class_names['kitti'],
        #     training=training,
        #     root_path=root_path,
        #     logger=logger,
        # )
        self.waymo_dataset = (
            None
            if class_names['WAYMO'] is False
            else WaymoDataset(
                dataset_cfg=dataset_cfg.WAYMO,
                class_names=class_names['WAYMO'],
                training=training,
                root_path=root_path,
                logger=logger,
            )
        )

        self.nuscenes_dataset = (
            None
            if class_names['NUSCENES'] is False
            else NuScenesDataset(
                dataset_cfg=dataset_cfg.NUSCENES,
                class_names=class_names['NUSCENES'],
                training=training,
                root_path=root_path,
                logger=logger,
            )
        )

        self.nia_dataset = (
            None
            if class_names['NIA'] is False
            else CustomDataset(
                dataset_cfg=dataset_cfg.NIA,
                class_names=class_names['NIA'],
                training=training,
                root_path=root_path,
                logger=logger,
            )
        )

        self.argo2_dataset = (
            None
            if class_names['ARGO2'] is False
            else Argo2Dataset(
                dataset_cfg=dataset_cfg.ARGO2,
                class_names=class_names['ARGO2'],
                training=training,
                root_path=root_path,
                logger=logger,
            )
        )

        self.once_dataset = (
            None
            if class_names['ONCE'] is False
            else ONCEDataset(
                dataset_cfg=dataset_cfg.ONCE,
                class_names=class_names['ONCE'],
                training=training,
                root_path=root_path,
                logger=logger,
            )
        )

        self.lyft_dataset = (
            None
            if class_names['LYFT'] is False
            else LyftDataset(
                dataset_cfg=dataset_cfg.LYFT,
                class_names=class_names['LYFT'],
                training=training,
                root_path=root_path,
                logger=logger,
            )
        )

        self.kitti_dataset_cfg = (
            dataset_cfg.KITTI if self.kitti_dataset is not None else None
        )
        self.waymo_dataset_cfg = (
            dataset_cfg.WAYMO if self.waymo_dataset is not None else None
        )
        self.nuscenes_dataset_cfg = (
            dataset_cfg.NUSCENES if self.nuscenes_dataset is not None else None
        )
        self.nia_dataset_cfg = dataset_cfg.NIA if self.nia_dataset is not None else None
        self.argo2_dataset_cfg = (
            dataset_cfg.ARGO2 if self.argo2_dataset is not None else None
        )
        self.once_dataset_cfg = (
            dataset_cfg.ONCE if self.once_dataset is not None else None
        )
        self.lyft_dataset_cfg = (
            dataset_cfg.LYFT if self.lyft_dataset is not None else None
        )

        # self.__all__ = [
        #     self.kitti_dataset,
        #     self.waymo_dataset,
        #     self.nuscenes_dataset,
        #     self.nia_dataset,
        #     self.argo2_dataset,
        #     self.once_dataset,
        #     self.lyft_dataset,
        # ]

        self.kitti_data_cnt = (
            len(self.kitti_dataset.kitti_infos) if self.kitti_dataset is not None else 0
        )
        self.waymo_data_cnt = (
            len(self.waymo_dataset.infos) if self.waymo_dataset is not None else 0
        )
        self.nuscenes_data_cnt = (
            len(self.nuscenes_dataset.infos) if self.nuscenes_dataset is not None else 0
        )
        self.nia_dataset_cnt = (
            len(self.nia_dataset.custom_infos) if self.nia_dataset is not None else 0
        )
        self.argo2_dataset_cnt = (
            len(self.argo2_dataset.argo2_infos) if self.argo2_dataset is not None else 0
        )
        self.once_dataset_cnt = (
            len(self.once_dataset.once_infos) if self.once_dataset is not None else 0
        )
        self.lyft_dataset_cnt = (
            len(self.lyft_dataset.infos) if self.lyft_dataset is not None else 0
        )

    def lyft_getitem(self, index):


        info = copy.deepcopy(self.lyft_dataset.infos[index])
        points = self.lyft_dataset.get_lidar_with_sweeps(
            index, max_sweeps=self.lyft_dataset_cfg.MAX_SWEEPS
        )

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']},
        }

        if 'gt_boxes' in info:
            for k in range(info['gt_names'].shape[0]):
                info['gt_names'][k] = (
                    self.lyft_dataset_cfg.MAP_LYFT_TO_CLASS[info['gt_names'][k]]
                    if info['gt_names'][k]
                    in self.lyft_dataset_cfg.MAP_LYFT_TO_CLASS.keys()
                    else info['gt_names'][k]
                )
            input_dict.update(
                {'gt_boxes': info['gt_boxes'], 'gt_names': info['gt_names']}
            )

        data_dict = self.lyft_dataset.prepare_data(data_dict=input_dict)

        return data_dict

    def nuscenes_getitem(self, index):
        if self.nuscenes_dataset._merge_all_iters_to_one_epoch:
            index = index % len(self.nuscenes_dataset.infos)

        info = copy.deepcopy(self.nuscenes_dataset.infos[index])
        points = self.nuscenes_dataset.get_lidar_with_sweeps(
            index, max_sweeps=self.nuscenes_dataset_cfg.MAX_SWEEPS
        )

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']},
        }

        if 'gt_boxes' in info:
            if self.nuscenes_dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (
                    info['num_lidar_pts'] > self.nuscenes_dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1
                )
            else:
                mask = None
            for k in range(info['gt_names'].shape[0]):
                info['gt_names'][k] = (
                    self.nuscenes_dataset_cfg.MAP_NUSCENES_TO_CLASS[
                        info['gt_names'][k]
                    ]
                    if info['gt_names'][k]
                    in self.nuscenes_dataset_cfg.MAP_NUSCENES_TO_CLASS.keys()
                    else info['gt_names'][k]
                )
            input_dict.update(
                {
                    'gt_names': info['gt_names']
                    if mask is None
                    else info['gt_names'][mask],
                    'gt_boxes': info['gt_boxes']
                    if mask is None
                    else info['gt_boxes'][mask],
                }
            )
        if self.nuscenes_dataset.use_camera:
            input_dict = self.nuscenes_dataset.load_camera_info(input_dict, info)

        data_dict = self.nuscenes_dataset.prepare_data(data_dict=input_dict)

        if (
            self.nuscenes_dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False)
            and 'gt_boxes' in info
        ):
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.nuscenes_dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict


    def nia_getitem(self, index):
        if self.nia_dataset._merge_all_iters_to_one_epoch:
            index = index % len(self.nia_dataset.custom_infos)

        info = copy.deepcopy(self.nia_dataset.custom_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.nia_dataset.get_lidar(sample_idx)
        input_dict = {
            'frame_id': self.nia_dataset.sample_id_list[index],
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            for k in range(annos['name'].shape[0]):
                annos['name'][k] = (
                    self.nia_dataset_cfg.MAP_NIA_TO_CLASS[annos['name'][k]]
                    if annos['name'][k]
                    in self.nia_dataset_cfg.MAP_NIA_TO_CLASS.keys()
                    else annos['name'][k]
                )
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.nia_dataset.prepare_data(data_dict=input_dict)

        return data_dict


    def once_getitem(self, index):
        if self.once_dataset._merge_all_iters_to_one_epoch:
            index = index % len(self.once_dataset.once_infos)

        info = copy.deepcopy(self.once_dataset.once_infos[index])
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = self.once_dataset.get_lidar(seq_id, frame_id)

        if self.once_dataset_cfg.get('POINT_PAINTING', False):
            points = self.once_dataset.point_painting(points, info)

        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        if 'annos' in info:
            annos = info['annos']

            for k in range(annos['name'].shape[0]):
                annos['name'][k] = (
                    self.once_dataset_cfg.MAP_ONCE_TO_CLASS[annos['name'][k]]
                    if annos['name'][k]
                    in self.once_dataset_cfg.MAP_ONCE_TO_CLASS.keys()
                    else annos['name'][k]
            )


            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['boxes_3d'],
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.once_dataset.prepare_data(data_dict=input_dict)
        data_dict.pop('num_points_in_gt', None)
        return data_dict


    def argo2_getitem(self, index):
        # index = 4
        if self.argo2_dataset._merge_all_iters_to_one_epoch:
            index = index % len(self.argo2_dataset.argo2_infos)

        info = copy.deepcopy(self.argo2_dataset.argo2_infos[index])

        sample_idx = info['point_cloud']['velodyne_path'].split('/')[-1].rstrip('.bin')
        calib = None
        get_item_list = self.argo2_dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']

            for k in range(annos['name'].shape[0]):
                annos['name'][k] = (
                    self.argo2_dataset_cfg.MAP_ARGO2_TO_CLASS[annos['name'][k]]
                    if annos['name'][k]
                    in self.argo2_dataset_cfg.MAP_ARGO2_TO_CLASS.keys()
                    else annos['name'][k]
                )

            gt_names = annos['name']
            gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_bboxes_3d
            })

        if "points" in get_item_list:
            points = self.argo2_dataset.get_lidar(sample_idx)
            input_dict['points'] = points

        input_dict['calib'] = calib
        data_dict = self.argo2_dataset.prepare_data(data_dict=input_dict)

        return data_dict


    def waymo_getitem(self, index):
        if self.waymo_dataset._merge_all_iters_to_one_epoch:
            index = index % len(self.waymo_dataset.infos)

        info = copy.deepcopy(self.waymo_dataset.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        input_dict = {
            'sample_idx': sample_idx
        }
        if self.waymo_dataset.use_shared_memory and index < self.waymo_dataset.shared_memory_file_limit:
            sa_key = f'{sequence_name}___{sample_idx}'
            points = SharedArray.attach(f"shm://{sa_key}").copy()
        else:
            points = self.waymo_dataset.get_lidar(sequence_name, sample_idx)

        if self.waymo_dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.waymo_dataset_cfg.SEQUENCE_CONFIG.ENABLED:
            points, num_points_all, sample_idx_pre_list, poses, pred_boxes, pred_scores, pred_labels = self.get_sequence_data(
                info, points, sequence_name, sample_idx, self.waymo_dataset_cfg.SEQUENCE_CONFIG,
                load_pred_boxes=self.waymo_dataset_cfg.get('USE_PREDBOX', False)
            )
            input_dict['poses'] = poses
            if self.waymo_dataset_cfg.get('USE_PREDBOX', False):
                input_dict.update({
                    'roi_boxes': pred_boxes,
                    'roi_scores': pred_scores,
                    'roi_labels': pred_labels,
                })

        input_dict.update({
            'points': points,
            'frame_id': info['frame_id'],
        })

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            for k in range(annos['name'].shape[0]):
                annos['name'][k] = (
                    self.waymo_dataset_cfg.MAP_WAYMO_TO_CLASS[annos['name'][k]]
                    if annos['name'][k]
                    in self.waymo_dataset_cfg.MAP_WAYMO_TO_CLASS.keys()
                    else annos['name'][k]
                )

            if self.waymo_dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.waymo_dataset_cfg.get('TRAIN_WITH_SPEED', False):
                assert gt_boxes_lidar.shape[-1] == 9
            else:
                gt_boxes_lidar = gt_boxes_lidar[:, 0:7]

            if self.training and self.waymo_dataset_cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN', False):
                mask = (annos['num_points_in_gt'] > 0)  # filter empty boxes
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.waymo_dataset.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)
        return data_dict

    def kitti_getitem(self, index):
        # index = 4
        if self.kitti_dataset._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_dataset.kitti_infos)

        info = copy.deepcopy(self.kitti_dataset.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.kitti_dataset.get_calib(sample_idx)
        get_item_list = self.kitti_dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.kitti_dataset.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.kitti_dataset.get_lidar(sample_idx)
            if self.kitti_dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.kitti_dataset.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.kitti_dataset.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.kitti_dataset.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        input_dict['calib'] = calib
        data_dict = self.kitti_dataset.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict


    def __len__(self):
        return (
            self.kitti_data_cnt
            + self.waymo_data_cnt
            + self.nuscenes_data_cnt
            + self.nia_dataset_cnt
            + self.argo2_dataset_cnt
            + self.once_dataset_cnt
            + self.lyft_dataset_cnt
        )
    def __getitem__(self, index):
        if (
            index
            >= self.kitti_data_cnt
            + self.waymo_data_cnt
            + self.nuscenes_data_cnt
            + self.nia_dataset_cnt
            + self.argo2_dataset_cnt
            + self.once_dataset_cnt
        ):
            if self.lyft_dataset is None:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            else:
                lyft_index = (
                    index
                    - self.kitti_data_cnt
                    - self.waymo_data_cnt
                    - self.nuscenes_data_cnt
                    - self.nia_dataset_cnt
                    - self.argo2_dataset_cnt
                    - self.once_dataset_cnt
                )
                data_dict = self.lyft_getitem(lyft_index)
        elif (
            index
            >= self.kitti_data_cnt
            + self.waymo_data_cnt
            + self.nuscenes_data_cnt
            + self.nia_dataset_cnt
            + self.argo2_dataset_cnt
        ):
            if self.once_dataset is None:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            else:
                once_index = (
                    index
                    - self.kitti_data_cnt
                    - self.waymo_data_cnt
                    - self.nuscenes_data_cnt
                    - self.nia_dataset_cnt
                    - self.argo2_dataset_cnt
                )
                data_dict = self.once_getitem(once_index)
        elif (
            index
            >= self.kitti_data_cnt
            + self.waymo_data_cnt
            + self.nuscenes_data_cnt
            + self.nia_dataset_cnt
        ):
            if self.argo2_dataset is None:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            else:
                argo2_index = (
                    index
                    - self.kitti_data_cnt
                    - self.waymo_data_cnt
                    - self.nuscenes_data_cnt
                    - self.nia_dataset_cnt
                )
                data_dict = self.argo2_getitem(argo2_index)
        elif (
            index >= self.kitti_data_cnt + self.waymo_data_cnt + self.nuscenes_data_cnt
        ):
            if self.nia_dataset is None:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            else:
                nia_index = (
                    index
                    - self.kitti_data_cnt
                    - self.waymo_data_cnt
                    - self.nuscenes_data_cnt
                )
                data_dict = self.nia_getitem(nia_index)
        elif index >= self.kitti_data_cnt + self.waymo_data_cnt:
            if self.nuscenes_dataset is None:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            else:
                nuscenes_index = index - self.kitti_data_cnt - self.waymo_data_cnt
                data_dict = self.nuscenes_getitem(nuscenes_index)
        elif index >= self.kitti_data_cnt:
            if self.waymo_dataset is None:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            else:
                waymo_index = index - self.kitti_data_cnt
                data_dict = self.waymo_getitem(waymo_index)
        else:
            if self.kitti_dataset is None:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            else:
                data_dict = self.kitti_getitem(index)

        return data_dict
    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(
                            coor, ((0, 0), (1, 0)), mode='constant', constant_values=i
                        )
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, max_gt, val[0].shape[-1]), dtype=np.float32
                    )
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, : val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_boxes']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, val[0].shape[0], max_gt, val[0].shape[-1]),
                        dtype=np.float32,
                    )
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :, : val[k].shape[1], :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_scores', 'roi_labels']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, val[0].shape[0], max_gt), dtype=np.float32
                    )
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :, : val[k].shape[1]] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros(
                        (batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32
                    )
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, : val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(
                            desired_size=max_h, cur_size=image.shape[0]
                        )
                        pad_w = common_utils.get_pad_params(
                            desired_size=max_w, cur_size=image.shape[1]
                        )
                        pad_width = (pad_h, pad_w)
                        pad_value = 0

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(
                            image,
                            pad_width=pad_width,
                            mode='constant',
                            constant_values=pad_value,
                        )

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                elif key in ['calib']:
                    ret[key] = val
                elif key in ["points_2d"]:
                    max_len = max([len(_val) for _val in val])
                    pad_value = 0
                    points = []
                    for _points in val:
                        pad_width = ((0, max_len - len(_points)), (0, 0))
                        points_pad = np.pad(
                            _points,
                            pad_width=pad_width,
                            mode='constant',
                            constant_values=pad_value,
                        )
                        points.append(points_pad)
                    ret[key] = np.stack(points, axis=0)
                elif key in ['object_scale_noise', 'object_rotate_noise']:
                    max_noise = max([len(x) for x in val])
                    batch_noise = np.zeros((batch_size, max_noise), dtype=np.float32)
                    for k in range(batch_size):
                        batch_noise[k, : val[k].__len__()] = val[k]
                    ret[key] = batch_noise
                elif key in ['beam_labels']:
                    continue
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
