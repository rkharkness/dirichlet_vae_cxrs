
from dirichlet_vae_cxrs.model.vae import dirVAE, VAE
from dataloader import make_dataloaders
from training import train, LogisticRegression
import argparse
import pandas as pd
import numpy as np
import torch
import mlflow
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop,Adam,SGD

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

import scienceplots
plt.style.use(['science'])

class LatentViz():
    def __init__(self, vae_config, dir):
        super(LatentViz, self).__init__()
        self.dir = dir
        self.labels = ['No Finding', 'Lung Opacity', 'Pleural Effusion', 'Support Devices']
       # self.tsne = PCA(n_components=2)
        self.tsne = TSNE(n_components=2, verbose=1, perplexity=60, n_iter=1000)

        if self.dir==True:
            self.vae = dirVAE(vae_config)
            self.vae.load_state_dict(torch.load(f"/MULTIX/DATA/dir_vae_k2_clf_stable_conc05_3.pth"))
        else:
            self.vae = VAE(vae_config)
            self.vae.load_state_dict(torch.load("/MULTIX/DATA/vae_clf.pth"))

        self.vae.cuda()
        self.vae.eval()

    def reduce_latent(self, dataloader,fit,epoch):
        tsne_results_list = []
        batch_y_list = []
        z_list = []
        self.epoch = epoch
        for batch_idx, (batch_x, batch_y) in enumerate(tqdm(dataloader)):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_y = torch.squeeze(batch_y)
 
            classes = np.argmax(batch_y.cpu().detach().numpy(),axis=1)
            print(batch_y)
            print(classes)

            if self.dir:
                recon_x, alpha, z = self.vae(batch_x)
                z = torch.log(z)
            else:
                z, recon_x, mu, logvar = self.vae(batch_x)

            z_list.append(z.cpu().detach().numpy())

            #tsne_res = self.tsne_analysis(z.cpu().detach().numpy(), batch_y.cpu().detach().numpy(), fit=fit)
#            tsne_results_list.append(tsne_res)
           # if fit == False:
            batch_y_list.append(classes) #[np.argmax(batch_y.cpu().detach().numpy(),axis=1)])
                #tsne_results_list.append(tsne_res)
      
#        if fit==False:
        z_list = [x for y in z_list for x in y]

        print(z_list,'z')
        batch_y_list = [x for y in batch_y_list for x in y]
        tsne_res = self.tsne_fn(z_list, fit=fit)
#        print(tsne_res, 'ts')
#            print(batch_y_list)
#            tsne_results_list = [x for y in tsne_results_list for x in y]
 #           batch_y_list = [x for y in batch_y_list for x in y]
        self.tsne_vis(np.array(tsne_res), np.array(batch_y_list), fit)
    
    def tsne_fn(self, latent_space, fit):
        if fit==True:
            tsne_results = self.tsne.fit_transform(latent_space)
        else:
            tsne_results = self.tsne.fit_transform(latent_space)

        return tsne_results

    def tsne_analysis(self, latent_space, gt_labels, fit=True):
        tsne_results = self.tsne_fn(latent_space, fit=fit)
  #      return tsne_results
#        if fit == True:
 #           self.tsne_vis(tsne_results, gt_labels, fit)
 #       else:
        return tsne_results
       
    def tsne_vis(self, tsne_results, gt_labels, fit):

        plt.figure(figsize=(10,10))
        plt.title("t-SNE Results")
        print(gt_labels)
        for cl in range(4):
            indices = np.where(gt_labels==cl)
            plt.scatter(tsne_results[indices,0], tsne_results[indices, 1],label=self.labels[cl], alpha=0.5)
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")

        if fit == True:
            title = f"/MULTIX/DATA/cxr_vae/tsne_plot_train_{self.epoch}.pdf"
        else:
            title = f"/MULTIX/DATA/cxr_vae/tsne_plot_test_{self.epoch}.pdf"

        plt.legend()
        plt.savefig(title, dpi=300)


def create_optimizers(model, config: dict):
    if config['optimizer'] == 'adam':
        optimizer  = Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer']:
        optimizer = RMSprop(model.parameters(), lr=config['lr'])
    elif config['optimizer']:
        optimizer = SGD(model.parameters(), lr=config['lr'])

    return optimizer

def prep(config, df):
    k = config['k']
    #df = df[['Path','split','No Finding','Lung Opacity', 'Pleural Effusion', 'Support Devices']]
    df = df.replace(np.nan, 0) #['No Finding', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'])
    df = df.replace(-1, 0)
    
    #df_no_finding = df[df['No Finding']==1.0]
    #print('nf',len(df_no_finding))
    
    #df_lung_opacity = df[df['Lung Opacity']==1.0]
    #print('lo',len(df_lung_opacity))

    #idx1 = df.index[df['Pleural Other'] == 1.0].tolist()
    #idx2 = df.index[df['Pneumothorax'] == 1.0].tolist()
    #idx = idx1 + idx2
    
    df_no_finding = df[df['No Finding']==1.0]
    print('nf',len(df_no_finding))
    
    df_lung_opacity = df[df['Lung Opacity']==1.0]
    print('lo',len(df_lung_opacity))

    idx1 = df.index[df['Pleural Other'] == 1.0].tolist()
    idx2 = df.index[df['Pneumothorax'] == 1.0].tolist()
    idx = idx1 + idx2
    
    df.loc[:, 'Pleural Effusion'] = np.where(df.index.isin(idx), 0.0, 1.0)
    df_pleural_effusion = df[df['Pleural Effusion']==1.0]
    print('pe',len(df_pleural_effusion))

    df_support_devices = df[df['Support Devices']==1.0]
    print('sd',len(df_support_devices))
    
    sample_size = [len(df_no_finding), len(df_lung_opacity),len(df_pleural_effusion),len(df_support_devices)]
    max_size = min(sample_size)
    print(max_size)
    df = pd.concat([df_no_finding.sample(max_size, random_state=1),df_lung_opacity.sample(max_size,random_state=1),df_pleural_effusion.sample(max_size,random_state=1),df_support_devices.sample(max_size,random_state=1)])
    df = df.sample(frac=1, random_state=1)

#    df = df[['Path','split','No Finding','Lung Opacity', 'Pleural Effusion', 'Support Devices']]
    df_lab = df[['No Finding','Lung Opacity', 'Pleural Effusion', 'Support Devices']]
    drop_idx = df_lab.index[(df_lab.sum(axis=1) > 1 )]
    #drop_idx = df.index[df[['No Finding','Lung Opacity', 'Pleural Effusion', 'Support Devices']].sum() > 1]
    print(len(drop_idx))
    df = df.drop(drop_idx)
#    df.iloc['split'] = 'train'
    df.split[:10000] = 'test'

    start_idx = [10000,15000,20000][k-1]
    end_idx = [15000,20000,25000][k-1]

    df.split[start_idx: end_idx] = 'val'
    df.split[end_idx:] = 'train'
    
   # df.iloc[:10000]['split'] = 'test'

#    start_idx=[10000,15000,20000][k]
#    end_idx= [15000,20000,25000][k]

 #   df.iloc[start_idx:end_idx]['split'] = 'val'
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == 'test']
    df = test_df    


    #df = pd.concat([df_no_finding.sample(max_size, random_state=1),df_lung_opacity.sample(max_size,random_state=1),df_pleural_effusion.sample(max_size,random_state=1),df_support_devices.sample(max_size,r
    df_no_finding = df[df['No Finding']==1.0]
    print('nf',len(df_no_finding))
    
    df_lung_opacity = df[df['Lung Opacity']==1.0]
    print('lo',len(df_lung_opacity))

    idx1 = df.index[df['Pleural Other'] == 1.0].tolist()
    idx2 = df.index[df['Pneumothorax'] == 1.0].tolist()
    idx = idx1 + idx2
    
    df.loc[:, 'Pleural Effusion'] = np.where(df.index.isin(idx), 0.0, 1.0)
    df_pleural_effusion = df[df['Pleural Effusion']==1.0]
    print('pe',len(df_pleural_effusion))

    df_support_devices = df[df['Support Devices']==1.0]
    print('sd',len(df_support_devices))
    
    sample_size = [len(df_no_finding), len(df_lung_opacity),len(df_pleural_effusion),len(df_support_devices)]
    max_size = min(sample_size)
    print(max_size)
    df = pd.concat([df_no_finding.sample(max_size, random_state=1),df_lung_opacity.sample(max_size,random_state=1),df_pleural_effusion.sample(max_size,random_state=1),df_support_devices.sample(max_size,random_state=1)])
    df = df.sample(frac=1, random_state=1)
    test_df = df
    for key,val in config.items():
        mlflow.log_param(key, val)

     #make generators
    dataloaders = make_dataloaders(train_df, val_df, test_df, config)  # create dict of dataloaders
    return dataloaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCA Script')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/chexpert_df.csv', type=str, help='Path to data file')
    parser.add_argument('--prior', default='gaussian', type=str, help='Model prior: gaussian or dirichlet')
    parser.add_argument('--batchsize', default=48, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--mode', default=None, type=str)
    parser.add_argument('--beta', default=5, type=int, help='ENC LOSS: KLD + BETA')
    parser.add_argument('--alpha', default=0.9, type=float, help='Controls DIRICHLET sparsity')
    parser.add_argument('--warm_up_period', default=10, type=int, help="Epochs without KLD")
    parser.add_argument('--z_dim', default=1024, type=int, help='Size of latent space')
    parser.add_argument('--lr', default=1e-4, type=float, help='Optimizer learning rate')
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer')
    parser.add_argument('--experiment_name', default="Rachael-mm-dVAE-GAN", type=str, help='Experiment name')
    parser.add_argument('--artifact_location', default="s3://users-rharkness-london/mlruns/", type=str, help='Where to save artefacts')
    parser.add_argument('--tracking_uri', default="https://k3s.multi-x.org:5000", type=str)
    parser.add_argument('--k', type=int)
    args = parser.parse_args()

    config = {"mode":args.mode, "k":args.k, "batchsize":args.batchsize, "alpha":args.alpha, "warm_up_period":args.warm_up_period, "num_workers":args.num_workers, "optimizer":args.optimizer,"lr":args.lr, "dataset":'chexpert',"k":1, "beta":args.beta, 'z_dim':args.z_dim}
    df = pd.read_csv(args.data_csv)

    if args.prior == 'gaussian':
        dir = False
    else:
        dir = True  

    dataloaders = prep(config, df)
    latent_viz = LatentViz(config, dir=dir)

    #train
#    latent_viz.reduce_latent(dataloaders['train'], fit=True)
    #test
    for i in range(20):
        latent_viz.reduce_latent(dataloaders['test'], fit=True, epoch=i)