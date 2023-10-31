from imp import source_from_cache
from locale import normalize
import pandas as pd
import numpy as np
import cv2

import matplotlib.pyplot as plt
from training import LogisticRegression
from dataloader import make_dataloaders
from dirichlet_vae_cxrs.model.vae import VAE, dirVAE
import torch
import argparse
from tqdm import tqdm
import torchvision.utils as vutils
import imageio
from  matplotlib import animation
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from scipy.interpolate import interp1d

torch.manual_seed(0)

plt.style.use('science')

from skbio.stats.composition import ilr, ilr_inv
    

def remove_confounders(df):
    # chexpert - ap / pa
    df = df[df['AP/PA'] == 'AP']
    # chexpert - frontal / lateral
    df = df[df['Frontal/Lateral'] == 'Frontal']
    # chexpert - gender
    df = df[df['Sex'] == 'Male']
    # chexpert - age
    df = df[df['Age'] >= 18]
    df = df[df['Age'] < 50]
    return df


def save_grid(img_list, path):
    img_grid = vutils.make_grid(img_list, nrow=4, padding=12, pad_value=-1)

    fig = plt.figure(figsize=(15,10))
    plt.imshow(img_grid[0].detach().cpu(), cmap='gray')
    plt.axis('off')

    fig.savefig(path, dpi=300)

def one_class(df):
    df = df.replace(np.nan, 0)
    df = df.replace(-1,1)
    Y_cat = df[['Lung Opacity', 'Pleural Effusion', 'Support Devices']].values
    cat_sum = np.sum(Y_cat,axis=1)
    print(cat_sum)
    nonselect_id = []
    for i in range(len(cat_sum)):
        if cat_sum[i]>1:
            nonselect_id.append(i) 
    df = df.drop(df.index[nonselect_id])
    return df
            
def create_data(df):
    df = df.replace(np.nan, 0)
    df = df.replace(-1, 1)

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

    df = df[['Path','split','AP/PA','Sex','Frontal/Lateral','Age','No Finding','Lung Opacity', 'Pleural Effusion', 'Support Devices']]

    df.iloc[:10000]['split'] = 'test'
    df.iloc[10000:15000]['split'] = 'val'

    return df

class LatentEval(torch.nn.Module):
    def __init__(self, vae, clf, dirichlet, label, dataloader, n=1):
        super(LatentEval, self).__init__()
        self.vae = vae.eval()
        self.clf = clf
        self.label=label

        self.dirichlet = dirichlet
        self.labels = ['No Finding', 'Lung Opacity', 'Pleural Effusion', 'Support Devices']

        self.label_idx = [i for i in range(len(self.labels)) if self.labels[i] == label][0]        

        self.n = n

        self.dataloader = dataloader
        self.params = self.get_clf_params()

    def get_clf_grad(self, z):
        z = torch.tensor(z, requires_grad=True)
        pred, probs = self.clf(z)

        probs[self.label_idx].backward()
        z_grad = z.grad
        return z_grad

    def get_clf_params(self):
        params = []
        for name, param in self.clf.named_parameters():
            if name == f'models.{self.label_idx}.linear.weight':
                params.append(param.detach().cpu().numpy().flatten())
        return params

    def predict(self, batch_x): # batch size = 1
        if self.dirichlet:
            recon_x, alpha, z = self.vae(batch_x)
        else:
            z, recon_x, mu, logvar = self.vae(batch_x)

        multilabel_pred, multilabel_prob = self.clf(z)
        return z, multilabel_pred, multilabel_prob 

    def check_samples(self, weights, latent_vector):

        for idx, vec in enumerate([weights, latent_vector]):
            mean = np.mean(vec, axis=0)
            var = np.var(vec, axis=0)

            plt.figure(figsize=(15,8))
            ind = np.arange(len(vec[0]))
            width = 8
            plt.bar(ind, mean, width, yerr=var, error_kw=dict(ecolor='red', lw=2, capsize=2, capthick=2))
            if idx == 0:
                plt.savefig(f'/MULTIX/DATA/cxr_vae/clf_weights_{self.label}.pdf', dpi=300)
            else:
                plt.savefig(f'/MULTIX/DATA/cxr_vae/latent_var_{self.label}.pdf', dpi=300)


    def select_factors(self, z, pred_class):
        z_grads = self.get_clf_grad(z)
        z_idxs = np.argsort(-z_grads.detach().cpu().numpy().flatten())[:self.n]
        # z = z[0].detach().cpu().numpy().flatten() 
        # z = z/np.max(z)

        # clf_params = np.abs(np.array(self.params))
        # # clf_params = clf_params/np.max(clf_params)

        # # clf_params = np.argsort(clf_params)
        # # latent_z = np.argsort(z[0].detach().cpu().numpy().flatten())

        # print(clf_params)
        # print(z)
        # comb_z_params  = clf_params * z # +latent_z 

        # print(comb_z_params)

        # z_idxs  = np.argsort(-comb_z_params).flatten()[:self.n]
        return z_idxs

    # def calc_var(self):
    #     train_z = []
    #     params_list = []

    #     for idx, (batch_x, batch_y) in enumerate(self.train_loaders[0]):
    #         batch_x = batch_x.cuda()
    #         batch_y = batch_y.cuda()
    #         batch_y = torch.squeeze(batch_y)

    #         results = self.predict(batch_x, batch_y, correct_only=True)

    #         if results != None:
    #             z, multilabel_pred, multilabel_prob, params = results
    #             train_z.append(z[0].detach().cpu().numpy().flatten())
    #             params_list.append(np.abs(params))

    #             if idx < 30:
    #                 self.plot_weights(params, idx)
    #             else:
    #                 pass

    #     if self.var == False:
    #         params_avg = np.mean(params_list[0], axis=0)
    #         z_avg = np.mean(train_z, axis=0)
    #     else:
    #         params_avg = np.mean(params_list[0], axis=0)
    #         z_avg = np.mean(train_z, axis=0)

    #     clf_z_idxs = (-params_avg).argsort()[:self.n]
    #     latent_z_idxs = (-z_avg).argsort()[:self.n]

    #     return clf_z_idxs, latent_z_idxs


    def plot_weights(self, weights, idx, train=True):
        plt.figure()
        assert len(weights[0]) == 1024, "Dimensionality error - weights don't match latent dimensions"
        ind = np.arange(len(weights[0]))
        width = 5
        plt.bar(ind, weights[0], width=width)

        if self.dirichlet:
            prefix = 'dirichlet'
        else:
            prefix = 'gaussian'

        if train == True:
            PATH = f'/MULTIX/DATA/cxr_vae/{prefix}_clf_weights_train_{self.label}_{idx}.pdf'
        else:
            PATH = f'/MULTIX/DATA/cxr_vae/{prefix}_clf_weights_test_{self.label}_{idx}.pdf'

        plt.savefig(PATH, dpi=300)

    
    def decode(self, z):
        if self.dirichlet:
            recon_x = self.vae.decoder(z)
        else:
            recon_x = self.vae.decoder(z)
        return recon_x

        
    def plot_latents(self, z, z_idx, i):
        z = z.detach().cpu().numpy()

        fig=plt.figure(figsize=(6,3))
        fig.show()
        ax=fig.add_subplot(111)
        ax.bar(np.arange(1024), z, width=5)
        ax.set_xlabel("Latent Factor")

        ax.set_ylabel("Value")
        ax.set_title("Latent Representation")

        if self.dirichlet:
            PATH = f'/MULTIX/DATA/cxr_vae/dir_latent_factors_{self.label}_{i}_z_{z_idx}.pdf'
        else:
            PATH = f'/MULTIX/DATA/cxr_vae/gau_latent_factors_{self.label}_{i}_z_{z_idx}.pdf'

        plt.savefig(PATH, dpi=300)

    # def plot_changing_latents(self, z_traversal, z_idx_list, interval=2):
    #     num_traversals = len(z_traversal)

    #     print(z_traversal[0][0])

    #     plt.figure()
    #     plt.bar(np.arange(1024), z_traversal[0], width=4)
    #     # plt.xticks(np.arange(1024))
    #     plt.tight_layout()
    #     plt.savefig(f'/MULTIX/DATA/cxr_vae/example_latent_space_{z_idx_list}.pdf', dpi=300)

    #     # z_idx_list.sort()
    #     fig, ax = plt.subplots(num_traversals, len(z_idx_list), figsize=(15,5))
    #     fig.subplots_adjust(hspace=0.05)  # adjust space between axes

    #     colours = ['r','g','b']

    #     for j in range(num_traversals):
    #         for i, z_idx in enumerate(z_idx_list):
    #             clipped_z = [z[z_idx-interval:z_idx+interval] for z in z_traversal]
    #             ind = np.arange(interval*2)
            
    #             ax[j,i].bar(ind, clipped_z[j], color=colours[i], width=0.2)
    #             ax[j,i].set_ylim(np.min(np.array(clipped_z).flatten()), np.min(np.array(clipped_z).flatten())+1e-3) #np.max(np.array(clipped_z).flatten()))
    #             # hide the spines between ax and ax2
    #             ax[j,i].spines['bottom'].set_visible(False)
    #             ax[j,i].spines['top'].set_visible(False)
    #             ax[j,i].xaxis.tick_top()
    #             ax[j,i].tick_params(labeltop=False)  # don't put tick labels at the top
    #             ax[j,i].xaxis.tick_bottom()
    #             d = .5  # proportion of vertical to horizontal extent of the slanted line
    #             kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
    #                         linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    #             ax[j,i].plot([0, 1], [0, 0], transform=ax[j,i].transAxes, **kwargs)
    #             ax[j,i].plot([0, 1], [1, 1], transform=ax[j,i].transAxes, **kwargs)
    #             # ax[j,i].set_xlim(z_idx-interval,z_idx+interval)
    #             # ax[j,i].spines['right'].set_visible(False)
    #             # ax[j,i].yaxis.tick_left()
    #             # ax[j,i].tick_params(labelright='off')
    #             ax[j,i].set_xticklabels(np.arange(z_idx-interval,z_idx+interval))

    #         plt.tight_layout()
    #         plt.savefig(f'/MULTIX/DATA/cxr_vae/changing_latent_zidx_{z_idx}.pdf', dpi=300)


    def latent_trav(self, z, batch_x, batch_y, num_traversals, idx, z_i=2, independent=False, gif=True):
        if gif==True:
            num_traversals=50 #tested max for 0.002 /n*2.5 for LO | 0.0025 / n *2 for SD

        z, multilabel_pred, multilabel_prob = self.predict(batch_x)

        traverse_recon_x = []
        traverse_recon_x.append(batch_x[0])
    
        z_vec=torch.unsqueeze(z[0],dim=0)
        z_idxs = self.select_factors(z_vec, self.label_idx)
        print(z_idxs)

        increment_list = torch.tensor(np.array([0.002]*len(z_idxs))/(self.n*2.5)).cuda() #dir 0.002 - gau 0.5

        # if len(z_idxs) == self.n and independent == False:
            # self.plot_latents(z[0], z_idxs, idx)

        probs = []
        preds = []

        z_traversal = []
        if len(z_idxs) == self.n and independent == True: # assess individual latent factors as well
            increment = increment_list[z_i]
            z_idx = z_idxs[z_i]
        else:
            z_idx = z_idxs
            increment = increment_list

        for i in np.arange(num_traversals):

            recon_x = self.decode(z_vec)
            multilabel_pred, multilabel_prob = self.clf(z_vec)

            traverse_recon_x.append(recon_x[0])

            probs.append(multilabel_prob.detach().cpu().numpy())
            preds.append(multilabel_pred)

            z_vec[0][z_idx] += increment #(var*4.7)

            z_traversal.append(z_vec[0].detach().cpu().numpy())
        
        # self.plot_changing_latents(z_traversal, z_idx)
        var = self.visualise_variance(traverse_recon_x)
        heatmap = self.heatmap_plot(traverse_recon_x, idx, z_idx)

        var = torch.unsqueeze(torch.tensor(var, device='cuda'),dim=0)

        traverse_recon_x_list = [traverse_recon_x[0], traverse_recon_x[1], \
            traverse_recon_x[10], traverse_recon_x[20], \
            traverse_recon_x[30], traverse_recon_x[40], traverse_recon_x[-1], var]

        # traverse_recon_x.append(torch.unsqueeze(torch.tensor(heatmap, device='cuda'), dim=0))

        if self.dirichlet:
            if self.n >= 10:
                PATH = f'/MULTIX/DATA/cxr_vae/grids/traversals/{self.label}/{self.n}z/dir_{self.label}_{idx}.pdf'
            else:
                PATH = f"/MULTIX/DATA/cxr_vae/grids/traversals/{self.label}/dir_{self.label}_{idx}_z_{z_idx}.pdf"
        else:
            PATH = f"/MULTIX/DATA/cxr_vae/grids/traversals/{self.label}/gau_{self.label}_{idx}_z_{z_idx}.pdf"

        save_grid(traverse_recon_x_list, PATH)

        gif_list = [np.squeeze(i.detach().cpu().numpy()) for i in traverse_recon_x[1:]]
        if self.dirichlet:
            if self.n >= 10:
                PATH = f'/MULTIX/DATA/cxr_vae/gifs/traversals/{self.label}/{self.n}z/dir_{self.label}_{idx}.gif'
            else:
                PATH = f"/MULTIX/DATA/cxr_vae/gifs/traversals/{self.label}/dir_traversal_{self.label}_{idx}_z_{z_idx}.gif"
        else:
            PATH = f"/MULTIX/DATA/cxr_vae/gifs/traversals/{self.label}/gau_traversal_{self.label}_{idx}_z_{z_idx}.gif"

        imageio.mimsave(PATH, gif_list)
        
        return traverse_recon_x

    def visualise_variance(self, recon_samples):
        recons = [i.detach().cpu().numpy() for i in recon_samples]
        img_var = np.var(np.squeeze(np.array(recons[1:])), axis=0) #pixel wise variance - ont including original recon
        img = img_var * np.array(np.squeeze(recons[0]))
        return img/np.max(img.flatten())

    def heatmap_plot(self, recon_samples, idx, z):
        recons = [i.detach().cpu().numpy() for i in recon_samples]
        img_var = np.var(np.squeeze(np.array(recons[1:-1])), axis=0) #pixel wise variance - ont including original econ
        
        # img_var = img_var/np.max(img_var.flatten())
        img_var = (img_var-np.min(img_var.flatten()))/(np.max(img_var.flatten())-np.min(img_var.flatten()))
        img_var[img_var<=0.1] = 0
        img_var = np.expand_dims(img_var, 2)

        heatmapimg = (img_var*255.).astype(np.uint8)
        heatmapimg = np.repeat(heatmapimg, 3, axis=2)
        print(heatmapimg.shape)

        heatmapimg = cv2.cvtColor(heatmapimg, cv2.COLOR_RGB2BGR)
        heatmap = cv2.applyColorMap(heatmapimg, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        orig = cv2.cvtColor((recons[0].transpose(1,2,0)*255.).astype(np.uint8), cv2.COLOR_GRAY2RGB)

       # orig = cv2.cvtColor(recons[0].transpose(1,2,0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        super_imposed_img = cv2.addWeighted(heatmap, 0.3, orig, 0.7, 0) #, dtype = cv2.CV_32F)
        super_imposed_img = super_imposed_img/255.

        plt.figure()
        plt.imshow(super_imposed_img, cmap='jet')
        plt.colorbar()
        plt.axis('off')

        if self.dirichlet:
            if self.n >= 10:
                PATH = f'/MULTIX/DATA/cxr_vae/heatmaps/{self.label}/{self.n}z/dir_heatmap_{self.label}_{idx}.pdf'
            else:
                PATH = f"/MULTIX/DATA/cxr_vae/heatmaps/{self.label}/dir_heatmap_{self.label}_{idx}_z_{z}.pdf"
        else:
            PATH = f"/MULTIX/DATA/cxr_vae/heatmaps/{self.label}/gau_heatmap_{self.label}_{idx}_z_{z}.pdf"

        plt.savefig(PATH, dpi=300)
        return super_imposed_img

    def evaluate(self, ind):
        count = 0
        for batch_x, batch_y in tqdm(self.dataloader):
            count +=1
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_y = torch.squeeze(batch_y)

            z, multilabel_pred, multilabel_prob = self.predict(batch_x)
            if batch_y.detach().cpu().numpy()[0][self.label_idx] == multilabel_pred[0][self.label_idx]:
                print('tp')
                # self.plot_weights(params, count, train=False)
                if ind == True:
                    for i in range(3):
                        traverse_recon = self.latent_trav(z, batch_x, batch_y,num_traversals=25, idx=count, independent=ind, z_i=i)
                else:
                    traverse_recon = self.latent_trav(z, batch_x, batch_y, num_traversals=25, idx=count, independent=ind)

            # heatmap = self.heatmap_plot(traverse_recon, count)
            if count == 300:
                break


    # min_probs = np.min(np.array(probs[0][self.label1_idx]).flatten())
    # max_probs = np.max(np.array(probs[0][self.label1_idx]).flatten())

    # fig, ax = plt.subplots(figsize=(5,8))
    # bars = ax.bar(range(len(self.labels)), probs[0])
    # plt.xticks(np.arange(0,4), labels=self.labels, rotation = 45)
    # ax.set_ylabel('Probability')
    # ax.set_title(self.label1)
    # # ax.set_ylim(min_probs-0.05, max_probs+0.05)

    # def update(i):
    #     for bar, height in zip(bars, probs[i]):
    #         bar.set_height(height)

    #     return bars

    # anim = animation.FuncAnimation(fig, update, frames=len(probs))
    # anim.save(f'/MULTIX/DATA/cxr_vae/latent_animation_{self.label1}_{z_idx}_{idx}.gif', dpi=300)

def latent_interpolate(evaluator1, evaluator2, num_traversals=150, dirichlet=True):

    label = evaluator1.label
    label2 = evaluator2.label

    counter  = 0
    for (batch_x1, batch_y1), (batch_x2, batch_y2) in zip(evaluator1.dataloader, evaluator2.dataloader):
        counter +=1

        print(counter)

        batch_x1 = batch_x1.cuda()
        batch_x2 = batch_x2.cuda()
        
        batch_y1 = batch_y1.cuda()
        batch_y2 = batch_y2.cuda()

        z1, pred1, prob1 = evaluator1.predict(batch_x1)
        z2, pred2, prob2 = evaluator2.predict(batch_x2)

        z1i = z1[0]
        z2i = z2[0]

        # Convert these points to Euclidean space:
        start_euc = ilr_inv(z1i.detach().cpu().numpy()).reshape(1,-1)
        end_euc = ilr_inv(z2i.detach().cpu().numpy()).reshape(1,-1)
        delta_euc = end_euc - start_euc

        interpolations = np.linspace(0.+1e-5, 1.-1e-5, 150).reshape(-1, 1)
        # Now we can create a Euclidean line between two euclidean points
        # path_euc = np.array([i + j * k for i,j,k in zip(start_euc,interpolations,delta_euc)])
        path_euc = start_euc + interpolations * delta_euc
        # And we map the path back to the Simplex
        path = ilr(path_euc)

        print(path)
        z_inter = torch.tensor(path, dtype=torch.float).cuda()
        print(z_inter.shape)
        # z_inter = torch.tensor([np.linspace(x1, x2, num_traversals, dtype=np.float) for x1, x2 in zip(z1i.detach().cpu().numpy(), z2i.detach().cpu().numpy())], dtype=torch.float)
        # z_inter = z_inter.permute(1,0).cuda()

        multilabel_pred, multilabel_prob = evaluator1.clf(z_inter)
        multilabel_prob = multilabel_prob.detach().cpu().numpy()


        print(multilabel_prob)
        inter_recon = evaluator1.decode(z_inter)

        img_gif_list = [np.squeeze(i.detach().cpu().numpy()) for i in inter_recon]

        if dirichlet:
            PATH = f"/MULTIX/DATA/cxr_vae/gifs/interpolations/{label}/dir_interpolate_{label}_{label2}_{counter}.gif"
        else:
            PATH = f"/MULTIX/DATA/cxr_vae/gifs/interpolations/{label}/gau_interpolate_{label}_{label2}_{counter}.gif"

        imageio.mimsave(PATH, img_gif_list)
            
        if dirichlet:
            PATH = f"/MULTIX/DATA/cxr_vae/gifs/probabilities/{label}/dir_interpolate_probs_{label}_{label2}_{counter}.gif"
        else:
            PATH = f"/MULTIX/DATA/cxr_vae/gifs/probabilities/{label}/gau_interpolate_probs_{label}_{label2}_{counter}.gif"

        fig, ax = plt.subplots(figsize=(5,8))
        bars = ax.bar(range(len(evaluator1.labels)), multilabel_prob[0])
        plt.xticks(np.arange(0,4), labels=evaluator1.labels, rotation = 45)
        ax.set_ylabel('Probability')
        ax.set_title(f"Interpolation from {label} to {label2}")
        # ax.set_ylim(min_probs-0.05, max_probs+0.05)

        def update(j):
            for bar, height in zip(bars, multilabel_prob[j]):
                bar.set_height(height)
            return bars

        anim = animation.FuncAnimation(fig, update, frames=num_traversals)
        anim.save(PATH, dpi=100)

        # traverse_recon_x.append(torch.unsqueeze(torch.tensor(heatmap, device='cuda'), dim=0))
        # if dirichlet:
            # PATH = f"/MULTIX/DATA/cxr_vae/grids/interpolations/{label}/dir_interpolate_{label}_{label2}_{counter}.pdf"
        # else:
            # PATH = f"/MULTIX/DATA/cxr_vae/grids/interpolations/{label}/gau_interpolate_{label}_{label2}_{counter}.pdf"

        z_inter = z_inter.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(5,8))
        bars = ax.bar(range(1024), z_inter[0]) #.detach().cpu().numpy())
        ax.set_ylabel('Values')
        ax.set_title(f"Interpolation from {label} to {label2}")
        # ax.set_ylim(min_probs-0.05, max_probs+0.05)

        def update(j):
            for bar, height in zip(bars, z_inter[j]):
                bar.set_height(height)
            return bars

        if dirichlet:
            PATH = f"/MULTIX/DATA/cxr_vae/gifs/probabilities/{label}/dir_interpolate_probs_full_z_{label}_{label2}_{counter}.gif"
        else:
            PATH = f"/MULTIX/DATA/cxr_vae/gifs/probabilities/{label}/gau_interpolate_probs_full_z_{label}_{label2}_{counter}.gif"

        anim = animation.FuncAnimation(fig, update, frames=num_traversals)
        anim.save(PATH, dpi=100)

        # z_inter = [z.detach().cpu().numpy() for z in z_inter]
        changing_z = np.argsort(-np.abs(z_inter[0]-z_inter[-1]))[:100]

        z1i = z1i.detach().cpu().numpy()
        z1i = np.array([z1i]*len(z_inter))
        z_inter = np.array(z_inter)

        z1i[:,changing_z] = z_inter[:,changing_z]
        z1i = torch.tensor(z1i).cuda()

        multilabel_pred, multilabel_prob = evaluator1.clf(z1i)
        multilabel_prob = multilabel_prob.detach().cpu().numpy()
        inter_recon = evaluator1.decode(z1i)

        img_gif_list = [np.squeeze(i.detach().cpu().numpy()) for i in inter_recon]

        if dirichlet:
            PATH = f"/MULTIX/DATA/cxr_vae/gifs/interpolations/{label}/dir_interpolate_100z_{label}_{label2}_{counter}.gif"
        else:
            PATH = f"/MULTIX/DATA/cxr_vae/gifs/interpolations/{label}/gau_interpolate_100z_{label}_{label2}_{counter}.gif"

        imageio.mimsave(PATH, img_gif_list)

        if dirichlet:
            PATH = f"/MULTIX/DATA/cxr_vae/grids/interpolations/{label}/dir_interpolate_100z_{label}_{label2}_{counter}.pdf"
        else:
            PATH = f"/MULTIX/DATA/cxr_vae/grids/interpolations/{label}/gau_interpolate_100z_{label}_{label2}_{counter}.pdf"

        var = evaluator1.visualise_variance(inter_recon)
        var = torch.unsqueeze(torch.tensor(var, device='cuda'),dim=0)

        inter_recon_list = [inter_recon[0], inter_recon[1], inter_recon[25], inter_recon[50], inter_recon[75],  \
            inter_recon[100], inter_recon[-1], var]

        save_grid(inter_recon_list, PATH)

        fig, ax = plt.subplots(figsize=(5,8))
        bars = ax.bar(range(len(evaluator1.labels)), multilabel_prob[0])
        plt.xticks(np.arange(0,4), labels=evaluator1.labels, rotation = 45)
        ax.set_ylabel('Probability')
        ax.set_title(f"Interpolation from {label} to {label2}")
        # ax.set_ylim(min_probs-0.05, max_probs+0.05)

        if dirichlet:
            PATH = f"/MULTIX/DATA/cxr_vae/gifs/probabilities/{label}/dir_interpolate_100z_probs_{label}_{label2}_{counter}.gif"
        else:
            PATH = f"/MULTIX/DATA/cxr_vae/gifs/probabilities/{label}/gau_interpolate_100z_probs_{label}_{label2}_{counter}.gif"

        def update(j):
            for bar, height in zip(bars, multilabel_prob[j]):
                bar.set_height(height)
            return bars

        anim = animation.FuncAnimation(fig, update, frames=num_traversals)
        anim.save(PATH, dpi=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/chexpert_df.csv', type=str, help='Path to data file')
    parser.add_argument('--prior', default='gaussian', type=str, help='Model prior: gaussian or dirichlet')
    parser.add_argument('--batchsize', default=2, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--mode', default=None, type=str)
    parser.add_argument('--beta', default=5, type=int, help='ENC LOSS: KLD + BETA')
    parser.add_argument('--alpha', default=0.3, type=float, help='Controls DIRICHLET sparsity')
    parser.add_argument('--warm_up_period', default=10, type=int, help="Epochs without KLD")
    parser.add_argument('--z_dim', default=1024, type=int, help='Size of latent space')
    parser.add_argument('--lr', default=1e-4, type=float, help='Optimizer learning rate')
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer')

    parser.add_argument('--label1', default='Lung Opacity', type=str, help='Primary img class for evaluation')
    parser.add_argument('--label2', default='No Finding', type=str, help='Destination img class for interpolation')


    args = parser.parse_args()
    config = {"mode":args.mode, "batchsize":args.batchsize, "warm_up_period":args.warm_up_period, "num_workers":args.num_workers, "optimizer":args.optimizer,"lr":args.lr, "dataset":'chexpert',"k":1, "beta":args.beta, 'z_dim':args.z_dim}

    if args.prior ==  'gaussian':
        model = VAE(config).cuda()
        model.load_state_dict(torch.load("/MULTIX/DATA/vae_clf.pth"))
        model.eval()
        dir = False

    else:
        model= dirVAE(config).cuda()
        model.load_state_dict(torch.load("/MULTIX/DATA/dir_vae_clf_stable_conc05_lr.pth"))
        model.eval()
        dir = True

    clf = LogisticRegression(args.z_dim).cuda()
    clf.load_weights(dir) # dir_clf_with_vae / gau_clf_final
    clf.eval()

    df = pd.read_csv(args.data_csv)

    # correct_df = pd.read_csv('/MULTIX/DATA/cxr_vae/chexpert_correct_results.csv')
    # correct_df = correct_df[correct_df['correct']==1]
    # df = pd.merge(correct_df, df, left_on='path', right_on='Path')

    df = create_data(df)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == 'test']

    # find 1 class cases
    train_df = one_class(train_df)
    test_df = one_class(test_df)

    print(test_df['Lung Opacity'].value_counts())

    label1 = args.label1
    test_df1 = test_df[test_df[label1]==1.0]
    val_df1 = test_df1
    train_df1 = test_df1

    #make generators
    dataloaders = make_dataloaders(train_df1, val_df1, test_df1, config)  # create dict of dataloaders

    evaluator1 = LatentEval(model, clf, dir, label1, dataloaders['test'], n=3)
    # evaluator1.evaluate(False)
    # evaluator1.evaluate(True)

    test_df1 = remove_confounders(test_df1)
    print(len(test_df1))
    train_df1 = test_df1
    val_df1 = test_df1

    dataloaders = make_dataloaders(train_df1, val_df1, test_df1, config)  # create dict of dataloaders

    evaluator1 = LatentEval(model, clf, dir, label1, dataloaders['test'], n=3)
    
    label2 = args.label2
    test_df2 = test_df[test_df[label2]==1.0]

    test_df2 = remove_confounders(test_df2)
    val_df2 = test_df2
    train_df2 = test_df2
    print(len(test_df2))

    dataloaders = make_dataloaders(train_df2, val_df2, test_df2, config)  # create dict of dataloaders
    evaluator2 = LatentEval(model, clf, dir, label2, dataloaders['test'], n=3)
    latent_interpolate(evaluator1, evaluator2)

