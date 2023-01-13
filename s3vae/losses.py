import torch.distributions as dist
import torch
from torch.nn.functional import max_pool3d
import numpy as np


def calculate_vae_loss(x, x_hat, z_f , z_t, z, z_t_prior):
    z_f_prior = dist.Normal(loc=z.new_zeros(z_f.loc.shape), scale=z.new_ones(z_f.loc.shape))

    # 맨 앞은 정규 분포를 따르게
    loc = z_t_prior.loc[:,:-1,:]
    scale = z_t_prior.scale[:,:-1,:]
    z_t_prior.loc = torch.cat([z.new_zeros(loc[:,:1,:].shape), loc], 1)
    z_t_prior.scale = torch.cat([z.new_ones(scale[:,:1,:].shape), scale], 1)

    img_loss = torch.sum(x_hat.log_prob(x), axis=(4,3,2)).mean()
    kl_td = torch.sum(dist.kl_divergence(z_t, z_t_prior), axis=(-1, 0)).mean() # kl-div time-dependant
    kl_ti = torch.sum(dist.kl_divergence(z_f, z_f_prior), axis=0).mean() # kl-div time invariant
    return -img_loss + kl_td + kl_ti


def calculate_dfp_loss(x, pred, criterion, device, topk=3, patch_size=3):
    dense_map = max_pool3d(x.float(), (1, int(x.shape[-2]//patch_size), int(x.shape[-1]//patch_size)))  # maxpool 3x3
    dense_map = dense_map.reshape(*dense_map.shape[:2], patch_size*patch_size)

    indices = torch.topk(dense_map, k=3, dim=-1).indices
    dense_label = torch.zeros(dense_map.shape).to(device)
    dense_label = dense_label.scatter_(-1, indices, 1.)
    return criterion(pred, dense_label)


def calculate_mi_loss(z_t_dist, z_f_dist):
    z_t1 = dist.Normal(loc=z_t_dist.loc.unsqueeze(2), scale=z_t_dist.scale.unsqueeze(2))
    z_t2 = dist.Normal(loc=z_t_dist.loc.unsqueeze(3), scale=z_t_dist.scale.unsqueeze(3))
    log_q_t = z_t1.log_prob(z_t2.rsample()).sum(-1)
    H_t = log_q_t.logsumexp(2).mean(1) - np.log(log_q_t.shape[2]) # torch.Size([16])

    z_f1 = dist.Normal(loc=z_f_dist.loc.unsqueeze(1), scale=z_f_dist.scale.unsqueeze(1))
    z_f2 = dist.Normal(loc=z_f_dist.loc.unsqueeze(2), scale=z_f_dist.scale.unsqueeze(2))
    log_q_f = z_f1.log_prob(z_f2.rsample()).sum(-1)
    H_f = log_q_f.logsumexp(1).mean(0) - np.log(log_q_f.shape[1]) # torch.Size([])
    H_ft = (log_q_f.unsqueeze(1) + log_q_t).logsumexp(1).mean(1) # torch.Size([16])

    mi_loss = (H_f + H_t.mean() - H_ft.mean()) 
    return mi_loss
