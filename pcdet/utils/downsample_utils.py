import pdb
import time

import numpy as np
import torch
from kmeans_pytorch import kmeans as pytorch_kmeans
from sklearn import cluster
from sklearn.cluster import KMeans
from torch.nn import functional as F


def compute_angles(pc_np):
    tan_theta = pc_np[:, 2] / (pc_np[:, 0] ** 2 + pc_np[:, 1] ** 2) ** (0.5)
    theta = np.arctan(tan_theta)

    sin_phi = pc_np[:, 1] / (pc_np[:, 0] ** 2 + pc_np[:, 1] ** 2) ** (0.5)
    phi_ = np.arcsin(sin_phi)

    cos_phi = pc_np[:, 0] / (pc_np[:, 0] ** 2 + pc_np[:, 1] ** 2) ** (0.5)
    phi = np.arccos(cos_phi)

    phi[phi_ < 0] = 2 * np.pi - phi[phi_ < 0]
    phi[phi == 2 * np.pi] = 0

    return theta, phi


def beam_label(theta, beam):
    # start_time = time.time()
    # estimator = KMeans(n_clusters=beam)
    # res = estimator.fit_predict(theta.reshape(-1, 1))
    # print(f'kmean : {time.time()-start_time}')
    # label = estimator.labels_
    # centroids = estimator.cluster_centers_
    # print(f'label : {label}')
    # data
    # data_size, dims, num_clusters = 1000, 2, 3
    # x = np.random.randn(data_size, dims) / 6
    # x = torch.from_numpy(x)
    # start_time = time.time()
    x = torch.from_numpy(theta.reshape(-1, 1))
    # torch.multiprocessing.set_start_method('spawn')
    # torch.cuda.current_device()
    cluster_ids_x, cluster_centers = pytorch_kmeans(
        X=x,
        num_clusters=beam,
        distance="euclidean",
        tqdm_flag=False,
        iter_limit=1000,
        # device=torch.device(f'cuda:{torch.cuda.current_device()}'),
        # tqdm_flag=False,
    )
    cluster_ids_x = cluster_ids_x.detach().cpu().numpy()
    cluster_centers = cluster_centers.detach().cpu().numpy()
    # print(f'cluster_ids_x :{cluster_ids_x}')
    # print(f'kmean-gpu : {time.time()-start_time}')

    # estimator = pytorch_kmeans
    # # kmeans
    # cluster_ids_x, cluster_centers = kmeans(
    #     X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
    # )
    return cluster_ids_x, cluster_centers[:, 0]
    # return label, centroids[:, 0]


def generate_mask(phi, beam, label, idxs, beam_ratio, bin_ratio):
    mask = np.zeros((phi.shape[0])).astype(np.bool)

    for i in range(0, beam, beam_ratio):
        phi_i = phi[label == idxs[i]]
        idxs_phi = np.argsort(phi_i)
        mask_i = label == idxs[i]
        mask_temp = np.zeros((phi_i.shape[0])).astype(np.bool)
        mask_temp[idxs_phi[::bin_ratio]] = True
        mask[mask_i] = mask_temp

    return mask
