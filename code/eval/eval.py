import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from dataloader import make_dataloaders
from utils import show_recons
import seaborn as sns
import scienceplots
from matplotlib import markers
from dirichlet_vae_cxrs.model.vae import VAE, dirVAE
from training import LogisticRegression
import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score, hamming_loss
import argparse
# from pca import PCA

plt.style.use(['science'])

class Eval(torch.nn.Module):
    def __init__(self, vae, clf, dirichlet, k):
        super(Eval, self).__init__()
        self.vae = vae.eval()
        self.clf = clf
        self.k = k

        self.dirichlet = dirichlet
        self.labels = ['No Finding', 'Lung Opacity', 'Pleural Effusion', 'Support Devices']

    def loss(self, batch_x, recon):
        loss_fn = torch.nn.L1Loss()
        loss = loss_fn(recon, batch_x)
        return loss

    def vae_predict(self, batch_x, recon, idx):
            if self.dirichlet:
                recon_x, alpha, z = self.vae(batch_x)
            else:
                z, recon_x, mu, logvar = self.vae(batch_x)
            
            if recon == True:
                if idx < 100:
                    show_recons(model, f'gau_test', batch_x, recon_x, idx, 'z')
                    show_recons(model, f'gau_test', batch_x, recon_x, idx, 'noise')
            
            l1loss = self.loss(batch_x, recon_x)

            return z, l1loss

    def prediction_labels(self, pred, batch_y, path, pred_results_dict):
            for i in range(len(pred)):
                print(pred[i])
                if all(np.array(pred[i]) == batch_y[i].detach().cpu().numpy()):
                    pred_results_dict[path[i]]=1
                else:
                    pred_results_dict[path[i]]=0
            return pred_results_dict
            
    def clf_predict(self, z):
            multilabel_pred, multilabel_prob = self.clf(z)
            return multilabel_pred, multilabel_prob

    def roc(self, y_score, y_test):
           # y_score = torch.flatten(torch.tensor(y_score), start_dim=0, end_dim=1)
           # y_test = torch.flatten(torch.tensor(y_test), start_dim=0, end_dim=1)
            plt.figure(figsize=(7,6))
            fpr = {i:[] for i in self.labels}
            tpr = {i:[] for i in self.labels}
            roc_auc = {i:[] for i in self.labels}
            
            lw=2
            y_test = np.array([np.array(i) for i in y_test])
            y_score = np.array([np.array(i) for i in y_score])

            for i in range(4):
                fpr[self.labels[i]], tpr[self.labels[i]], _ = roc_curve(y_test[:,:, i].flatten(),y_score[:,:,i].flatten())
                roc_auc[self.labels[i]] = auc(fpr[self.labels[i]], tpr[self.labels[i]])
           # colors = cycle(['blue', 'red', 'green', 'purple'])
            # get all possible shapes

            plt.figure(figsize=(7,5))

            #colors = sns.color_palette()
            #for i, color in zip(range(4), colors):
            for i in range(4):
                l = self.labels[i]
                x = fpr[l]
                y = tpr[l]
                plt.plot(x, y, lw=2,
                        label="{0}: (area = {1:0.2f})".format(self.labels[i], roc_auc[self.labels[i]]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([-0.05, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.legend(loc="lower right", prop={'size': 12})

            if self.dirichlet:
                plt.title('Dirichlet VAE', fontsize= 15)
                # plt.savefig('/MULTIX/DATA/cxr_vae/dir_test_clf_together_roc.pdf', dpi=300)
            else:
                plt.title('Gaussian VAE', fontsize= 15)
                # plt.savefig('/MULTIX/DATA/cxr_vae/gau_test_clf_together_roc.pdf', dpi=300)

            return fpr, tpr, roc_auc
    
    def collect_class_wise_metrics(self, multilabel_pred, y):
        # y=y.detach().cpu().numpy()
        class_metrics = {'acc':{}, 'prec':{}, 'recall':{},'emr':{}, 'case_prec':{}, 'case_recall':{},'hamming':{}}
        multilabel_pred = torch.flatten(torch.tensor(multilabel_pred),start_dim=0, end_dim=1)
        y = torch.flatten(torch.tensor(y),start_dim=0, end_dim=1)
        for i in range(4):
                class_pred = multilabel_pred[...,i]
 #               print(class_pred)
                class_y = y[...,i]
#                print(class_y)
                class_metrics['acc'][self.labels[i]] = accuracy_score(class_pred, class_y)
                class_metrics['prec'][self.labels[i]] = precision_score(class_pred, class_y)
                class_metrics['recall'][self.labels[i]] = recall_score(class_pred, class_y)
        return class_metrics


    def multilabel_cm(self, y_true, y_pred):

        multilabel_pred = torch.flatten(torch.tensor(y_pred),start_dim=0, end_dim=1)
        y = torch.flatten(torch.tensor(y_true),start_dim=0, end_dim=1)

        cm_dict = {i:[] for i in self.labels}
        fig, axs = plt.subplots(ncols=4, figsize=(10,5))
        fig.tight_layout()

        for i in range(4):
                class_pred = multilabel_pred[...,i]
 #               print(class_pred)
                class_y = y[...,i]
                cm = confusion_matrix(class_y, class_pred)

                classes = ["Negative","Positive"]
                sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False, ax=axs[i])
                title = f"{self.labels[i]}"
                axs[i].set(title=title, xlabel="Predicted Label", ylabel="True Label")

                cm_dict[self.labels[i]]=cm
        
        if self.dirichlet:
            PATH = f"/MULTIX/DATA/cxr_vae/dir_multilabel_cm_k{self.k}.pdf"
        else:
            PATH = f"/MULTIX/DATA/cxr_vae/gau_multilabel_cm_k{self.k}.pdf"

        # plt.savefig(PATH, dpi=300)
        return cm_dict

    def example_case(self, y, multilabel_pred, class_metrics):
        class_pred = torch.flatten(torch.tensor(multilabel_pred),start_dim=0, end_dim=1)
        class_y = torch.flatten(torch.tensor(y), start_dim=0, end_dim=1)

        class_metrics['emr'] = accuracy_score(class_pred, class_y)
        class_metrics['hamming'] = hamming_loss(class_y, class_pred)
        class_metrics['case_prec'] = precision_score(class_pred, class_y, average='samples')
        class_metrics['case_recall'] = recall_score(class_pred, class_y, average='samples')

        return class_metrics


    def call(self, test_loader):
        prob_list = []
        pred_list = []
        gt_list = []

        loss_list = []

        pred_dict = {}

        for batch_idx, (batch_x, batch_y, path) in enumerate(test_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_y = torch.squeeze(batch_y)

            z, loss = self.vae_predict(batch_x, True, batch_idx)
            pred, prob = self.clf_predict(z)

            prob_list.append(prob.detach().cpu())
            pred_list.append(pred)
            gt_list.append(batch_y.detach().cpu().numpy())

            loss_list.append(loss.item())

            pred_dict = self.prediction_labels(pred, batch_y, path,  pred_dict)

        pred_df = pd.DataFrame(pred_dict.items(), columns=['path','correct'])
      
        print(pred_df)
        print(pred_df['correct'].value_counts())

        pred_df.to_csv('/MULTIX/DATA/cxr_vae/chexpert_correct_results.csv')
        
        #flatten list but preserve multilabel ordering
        class_metrics = self.collect_class_wise_metrics(pred_list, gt_list)
        #cm
        cm = self.multilabel_cm(gt_list, pred_list)
        case_class_metrics = self.example_case(gt_list, pred_list, class_metrics)

        #roc
        fpr, tpr, roc_auc = self.roc(prob_list, gt_list)

        class_metrics['pred'] = [i.tolist() for i in pred_list]

        class_metrics['fpr'] = {k: v.tolist() for k, v in fpr.items()}
        class_metrics['tpr'] = {k: v.tolist() for k, v in tpr.items()}
        class_metrics['auc'] = {k: v.tolist() for k, v in roc_auc.items()}
        class_metrics['l1'] = np.mean(np.array(loss_list))

        print(class_metrics['auc'])
        print(class_metrics['recall'])
        print(class_metrics['prec'])

        if self.dirichlet:
            PATH = f'dirichlet_{self.k}.json'
        else:
            PATH = f'gaussian_{self.k}.json'
        # with open(PATH, 'w') as fp:
            # json.dump(class_metrics, fp)



def main(model, clf, config, df, dir):
    #df = df[['Path','split','No Finding','Lung Opacity', 'Pleural Effusion', 'Support Devices']]
    df = df.replace(np.nan, 0) #['No Finding', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'])
    df = df.replace(-1, 0)
    
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

    df = df[['Path','split','No Finding','Lung Opacity', 'Pleural Effusion', 'Support Devices']]

#    df.iloc['split'] = 'train'
    df.split[:10000] = 'test'

    start_idx = [10000,15000,20000][config['k']-1]
    end_idx = [15000,20000,25000][config['k']-1]

    df.split[start_idx: end_idx] = 'val'
    df.split[end_idx:] = 'train'
 
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == 'test']

    print(test_df)
    
    #make generators
    dataloaders = make_dataloaders(train_df, val_df, test_df, config)  # create dict of dataloaders

    evaluator = Eval(model, clf, dir, config['k'])
    evaluator.call(dataloaders['test'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/chexpert_df.csv', type=str, help='Path to data file')
    parser.add_argument('--prior', default='gaussian', type=str, help='Model prior: gaussian or dirichlet')
    parser.add_argument('--batchsize', default=48, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--mode', default=None, type=str)
    parser.add_argument('--beta', default=5, type=int, help='ENC LOSS: KLD + BETA')
    parser.add_argument('--alpha', default=0.5, type=float, help='Controls DIRICHLET sparsity')
    parser.add_argument('--warm_up_period', default=10, type=int, help="Epochs without KLD")
    parser.add_argument('--z_dim', default=1024, type=int, help='Size of latent space')
    parser.add_argument('--lr', default=1e-4, type=float, help='Optimizer learning rate')
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer')
    parser.add_argument('--k', type=int, help='Cross val iteration')
    parser.add_argument('--root', type=int, help='File root')


    args = parser.parse_args()

    config = {"mode":args.mode, "alpha":args.alpha, "k":args.k, "batchsize":args.batchsize, "warm_up_period":args.warm_up_period, "num_workers":args.num_workers, "optimizer":args.optimizer,"lr":args.lr, "dataset":'chexpert',"k":1, "beta":args.beta, 'z_dim':args.z_dim}

    if args.prior ==  'gaussian':
        model = VAE(config).cuda()
        model.load_state_dict(torch.load(f"{args.root}/vae_clf.pth"))
        model.eval()
        dir = False

    else:
        model= dirVAE(config).cuda()
        model.load_state_dict(torch.load(f"/{args.root}/dir_vae_k{args.k}_clf_stable_conc05.pth"))
        model.eval()
        dir = True

    clf = LogisticRegression(args.z_dim, args.k).cuda()
    clf.load_weights(dir) # dir_clf_with_vae / gau_clf_final
    clf.eval()

    df = pd.read_csv(args.data_csv)
    main(model, clf, config, df, dir)

