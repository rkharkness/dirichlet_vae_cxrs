import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import torchvision.utils as vutils

def show_recons(model, phase, batch_img, recon, epoch, input_type: str):
    if input_type == 'z':
        img_grid = vutils.make_grid(torch.cat((batch_img[:4], recon[:4])), nrow=4, padding=12, pad_value=-1)

        fig = plt.figure(figsize=(10,5))
        plt.imshow(img_grid[0].detach().cpu(), cmap='gray')
        plt.axis('off')
        fig.savefig(f"/MULTIX/DATA/mm_vae/large_input_recon_0_{phase}_{epoch}")
        return fig

    elif input_type == 'noise':
        recon = model.generate_noise_example(4)
        img_grid = vutils.make_grid(recon, nrow=4, padding=12, pad_value=-1)

        fig = plt.figure(figsize=(10,5))
        plt.imshow(img_grid[0].detach().cpu(), cmap='gray')
        plt.axis('off')
        fig.savefig(f"/MULTIX/DATA/mm_vae/large_noise_recon_0_{phase}_{epoch}")
        return fig