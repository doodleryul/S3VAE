from s3vae.datasets import SeqMNISTDataset
from torch.utils.data import DataLoader
from s3vae.s3vae import S3VAE
import wandb
import torch
import os
from tqdm import tqdm

def run(config, data_dir):
    wandb.init(project=config['wandb_project'], group=config['wandb_group'])


    train_dataset = SeqMNISTDataset(os.path.join(data_dir, 'train.npy'))
    valid_dataset = SeqMNISTDataset(os.path.join(data_dir, 'valid.npy'))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'])
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config['batch_size'])

    s3vae = S3VAE(config)


    for epoch in tqdm(range(config['num_epochs'])):
        for image in train_dataloader:
            shuffle_idx = torch.randperm(image.shape[0]).contiguous()
            permuted = image[shuffle_idx, :]
            loss, vae, scc, dfp, mi = s3vae.train_step(image, permuted)

            wandb.log({'train_loss/loss': loss, 
            'train_loss/vae': vae, 
            'train_loss/scc': scc, 
            'train_loss/dfp': dfp, 
            'train_loss/mi': mi})

        for image in valid_dataloader:
            shuffle_idx = torch.randperm(image.shape[0]).contiguous()
            permuted = image[shuffle_idx, :]
            loss, vae, scc, dfp, mi, x_hat = s3vae.validate_step(image, permuted)

            wandb.log({'valid_loss/loss': loss, 
            'valid_loss/vae': vae, 
            'valid_loss/scc': scc, 
            'valid_loss/dfp': dfp, 
            'valid_loss/mi': mi})

    os.makedirs(config['model_save_path'], exist_ok=True)
    s3vae.save(config['model_save_path'])