from torch.distributions.dirichlet import Dirichlet 
import torch
import torch.nn as nn
import torch.nn.functional as F
from stochastic import ResampleDir

from blocks import Encoder, Decoder

class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.beta = config['beta']
        self.z_dim = config['z_dim']
        self.warm_up_period = config['warm_up_period'] # 100
        self.batch_size = config['batchsize']
        self.mode = config['mode']
        self.k = config['k']

        self.kl_cost_annealing = True

        self.encoder = Encoder(z_size=self.z_dim)
        self.decoder = Decoder(self.z_dim, size=self.encoder.size)

        # two linear to get mu and logvar
        self.l_var = nn.Linear(in_features=1024, out_features=self.z_dim)
        self.mu  = nn.Linear(in_features=1024, out_features=self.z_dim)

        self.device='cuda'

    @property
    def qdist_nparams(self):
        return 2
        
    def generate_noise_example(self, num_examples):
        mu = torch.zeros(num_examples, self.z_dim).to(self.device)
        logvar = torch.ones(num_examples, self.z_dim).to(self.device)

        z = self.reparameterise(mu, logvar)
        return self.decoder(z)
    
    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        e = torch.randn_like(std)
        return  mu + (std * e)
    
    def forward(self, x):
        x = self.encoder(x)

        logvar = self.l_var(x)
        mu = self.mu(x)
        z = self.reparameterise(mu, logvar)
        recon_x = self.decoder(z)

        return z, recon_x, mu, logvar    

    def loss_fn(self, x_tilde, x, mu, logvar, epoch):
        bce = F.l1_loss(x_tilde.view(-1, 256*256), x.view(-1, 256*256), reduction='mean')   
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        kld_weightings = [0.01,0.02,0.03,0.04,0.05]
        if self.kl_cost_annealing == True:
            idx = epoch//5
            if idx > 4:
                idx = 4
            self.beta = kld_weightings[idx]
        else:
            self.beta = 0.05

        if self.get_mode(epoch) == 'mse':
            objective = bce
        elif self.get_mode(epoch) == 'kld' or 'clf':
            objective = bce + (self.beta * kld)
        else:
            objective = 0

        # increase beta from 0. to 1. - KL cost annealing
        return torch.mean(bce), torch.mean(kld), torch.mean(objective)

    def get_mode(self, epoch):
        if self.mode == None:
            if epoch < self.warm_up_period:
                self.mode = 'mse'
            else:
                self.mode = "kld"

        return self.mode

    def save_model(self, model, epoch):
        mode = self.get_mode(epoch)
        path = f"/MULTIX/DATA/vae_k{self.k}_{mode}.pth"
        torch.save(model.state_dict(), path)


class dirVAE(nn.Module):
    def __init__(self, config):
        super(dirVAE, self).__init__()
        print('initialising dirvae ...')

         # config
        self.beta = config['beta']
        self.z_dim = config['z_dim']
        self.warm_up_period = config['warm_up_period'] # 100
        self.batch_size = config['batchsize']
        self.k = config['k']

        self.device = 'cuda'

        self.mode = config['mode']
        self.alpha = config['alpha']

        self.encoder = Encoder(z_size=self.z_dim)
        self.alpha_fc = nn.Linear(in_features=1024, out_features=self.z_dim)

        self.resampler = ResampleDir(self.z_dim, self.batch_size, self.alpha)

        self.decoder = Decoder(self.z_dim, size=self.encoder.size)
        self.kl_cost_annealing = True

    @property
    def qdist_nparams(self):
        return self.z_dim

    def forward(self, x):
        x = self.encoder(x)
        alpha = self.alpha_fc(x)

        dir_sample = self.resampler.sample(alpha)
        x_hat = self.decoder(dir_sample)       

        return x_hat, alpha, dir_sample

    def loss_fn(self, x_tilde, x, alpha, epoch):
        bce = F.l1_loss(x_tilde.view(-1, 256*256), x.view(-1, 256*256), reduction='mean')
        analytical_kld = self.resampler.prior_forward(alpha)

        kld_weightings = [0.00001,0.00002,0.00003,0.00004,0.00005]
        if self.kl_cost_annealing == True:
            idx = epoch//5
            if idx > 4:
                idx = 4
            self.beta = kld_weightings[idx]
        else:
            self.beta = 0.00005

        if self.get_mode(epoch) == 'mse':
            objective = bce*1000

        elif self.get_mode(epoch) == 'kld' or 'clf':
            objective = (bce*100) + (self.beta * analytical_kld)

        return torch.mean(bce), torch.mean(analytical_kld), torch.mean(objective)

    def generate_noise_example(self, num_examples):
        alpha = torch.full((num_examples, self.z_dim), fill_value=self.alpha, dtype=torch.float, device='cuda')
        z = self.resampler.sample(alpha)
        return self.decoder(z)

    def get_mode(self, epoch):
        if self.mode == None:
            if epoch < self.warm_up_period:
                self.mode = 'mse'
            else:
                self.mode = "kld"
        return self.mode

    def save_model(self, model, epoch):
        mode = self.get_mode(epoch)
        path = f"/MULTIX/DATA/dir_vae_k{self.k}_{mode}_stable_conc05.pth"
        torch.save(model.state_dict(), path)