import torch
import torch.nn as nn
from torch.distributions import Dirichlet

class ResampleDir(nn.Module):
    def __init__(self, latent_dim, batch_size, alpha):
        super(ResampleDir, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.alpha = alpha
        self.alpha_target = torch.full((batch_size, latent_dim), fill_value=self.alpha, dtype=torch.float, device='cuda')

    def concentrations_from_logits(self, logits):
        alpha_c = torch.log(1.+ torch.exp(logits))
        alpha_c = torch.unsqueeze(alpha_c,-1)
        alpha_c = torch.logsumexp(alpha_c,-1,keepdim=True)
        alpha_c = torch.squeeze(alpha_c)
        return alpha_c

    def dirichlet_kl_divergence(self, logits, eps=10e-10):
        alpha_c_pred = self.concentrations_from_logits(logits)
        
        alpha_0_target = torch.sum(self.alpha_target, axis=-1, keepdims=True)
        alpha_0_pred = torch.sum(alpha_c_pred, axis=-1, keepdims=True)

        term1 = torch.lgamma(alpha_0_target) - torch.lgamma(alpha_0_pred)
        term2 = torch.lgamma(alpha_c_pred + eps) - torch.lgamma(self.alpha_target + eps)

        term3_tmp = torch.digamma(self.alpha_target + eps) - torch.digamma(alpha_0_target + eps)
        term3 = (self.alpha_target - alpha_c_pred) * term3_tmp

        result = torch.squeeze(term1 + torch.sum(term2 + term3, keepdims=True, axis=-1))
        return result
 
    def prior_forward(self, logits): # analytical kld loss
        return self.dirichlet_kl_divergence(logits)

    def sample(self, logits):
        alpha_pred = self.concentrations_from_logits(logits)  
        dir_sample = torch.squeeze(Dirichlet(alpha_pred).rsample()) #1 # output to decoder 
        return dir_sample