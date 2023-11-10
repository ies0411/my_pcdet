import torch

from .vfe_template import VFETemplate


class NormalVFE(VFETemplate):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_point_count = model_cfg.USE_POINT_COUNT
        self.use_plane_likeness = model_cfg.USE_PLANE_LIKENESS
        self.min_pca_point = model_cfg.MIN_POINT_PLANE

        self.num_point_features = 6 + self.use_point_count + self.use_plane_likeness

        self.idx_plane_likeness = 5 + self.use_plane_likeness
        self.idx_point_number = 5 + self.use_point_count + self.use_plane_likeness

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_pointsper_voxel, C)
                voxel_num_points: (num_voxels)
        Returns:
            voxel_features: (num_voxels, 8)
            [x,y,z,normal_x,normal_y,normal_z,(plane_likeness),(num_points)]
        """
        voxels, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        voxel_features = torch.zeros(
            voxels.shape[0], self.num_point_features, device=voxels.device
        )

        # point number features
        # TODO_datatype
        voxel_num_points_float = voxel_num_points.type_as(voxels)
        voxel_num_points_integer = voxel_num_points.type(torch.int32)
        if self.use_point_count:
            voxel_features[:, self.idx_point_number] = voxel_num_points_float

        # mean
        voxel_features[:, :3] = voxels[:, :, :3].sum(
            dim=1, keepdim=False
        ) / torch.clamp_min(voxel_num_points_float.view(-1, 1), min=1.0)

        # cov & plane_likeness
        for i, n_point in enumerate(voxel_num_points_integer):
            if n_point < self.min_pca_point:
                continue

            # # using torch api version =====
            # _,s,v = torch.pca_lowrank(voxels[1,:n_point,:3])
            # # pca is descending order
            # if (voxel_features[i,:3].dot(v[2,:]))<0.0:
            #     voxel_features[i,3:6] = -v[2,:]
            # else:
            #     voxel_features[i,3:6] = v[2,:]

            # if self.use_plane_likeness:
            #     voxel_features[i,self.idx_plane_likeness] = 2.0*(s[1]-s[2])/s.sum()

            # implemented version =========

            bias = voxels[i, :n_point, :3] - voxel_features[i, :3]
            cov = bias.t().mm(bias) / (n_point.type_as(voxels) - 1.0)
            # eigh returns ascending order
            eig_val, eig_vec = torch.linalg.eigh(cov)

            if (voxel_features[i, :3].dot(eig_vec[0, :])) < 0.0:
                voxel_features[i, 3:6] = -eig_vec[0, :]
            else:
                voxel_features[i, 3:6] = eig_vec[0, :]

            if self.use_plane_likeness:
                voxel_features[i, self.idx_plane_likeness] = (
                    2.0 * (eig_val[1] - eig_val[0]) / eig_val.sum()
                )

        batch_dict['voxel_features'] = voxel_features.contiguous()

        # TODO:
        # 1. 요상한 plane_likeness같은거 쓰지말고, eigenvalue 3종 주기
        # 2. voxel direction을 신경쓰는것이 차이가 날지?
        # 3. 절대 좌표가 아니라, voxel내부의 상대좌표로 하면 안될까? -> 그러면 셀 차이가 별로 안날거같긴한데...
        # 4. multi scale voxel은 안되니?

        return batch_dict
