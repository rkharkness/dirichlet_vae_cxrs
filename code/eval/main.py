from dataloader import make_dataloaders
from training import train, LogisticRegression
import argparse
from dirichlet_vae_cxrs.model.vae  import VAE, dirVAE
import pandas as pd

from torch.optim import RMSprop,Adam,SGD
import mlflow
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)

def create_optimizers(model, config: dict):
    if config['optimizer'] == 'adam':
        optimizer  = Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer']:
        optimizer = RMSprop(model.parameters(), lr=config['lr'])
    elif config['optimizer']:
        optimizer = SGD(model.parameters(), lr=config['lr'])

    return optimizer

def main(model, clf, clf_optimizer, config, df, dir, k):
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
    
    df.iloc[:10000]['split'] = 'test'

    start_idx=[10000,15000,20000][k]
    end_idx= [15000,20000,25000][k]

    df.iloc[start_idx:end_idx]['split'] = 'val'
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == 'test']
 #   print(df.columns)
    print(df['split'].value_counts())
    #print(test_df)
    optimizer = create_optimizers(model, config)

    for key,val in config.items():
        mlflow.log_param(key, val)

     #make generators
    dataloaders = make_dataloaders(train_df, val_df, test_df, config)  # create dict of dataloaders
    train(model=model, clf=clf, clf_optimizer=clf_optimizer, dataloaders=dataloaders, optimizers=optimizer, dir=dir, EPOCHS=500)

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
    parser.add_argument('--k', default='cross val iter', type=int, help='k-model to train')
    parser.add_argument('--experiment_name', default="Rachael-mm-dVAE-GAN", type=str, help='Experiment name')
    parser.add_argument('--artifact_location', default="s3://users-rharkness-london/mlruns/", type=str, help='Where to save artefacts')
    parser.add_argument('--tracking_uri', default="https://k3s.multi-x.org:5000", type=str)

    args = parser.parse_args()
    uri = "https://k3s.multi-x.org:5000"
    artifact_location = "s3://users-rharkness-london/mlruns"
    exp_name = "Rachael-mm-dVAE-GAN"
#    mlflow.set_tracking_uri(uri)

    try:
    # If the experiment doesn't exist, we create it.
        exp_id = mlflow.create_experiment(exp_name, artifact_location=artifact_location)
    except:
    # If the experiment already exists, we can just retrieve its ID
        exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id

    
    config = {"mode":args.mode, "batchsize":args.batchsize, "alpha":args.alpha, "warm_up_period":args.warm_up_period, "num_workers":args.num_workers, "optimizer":args.optimizer,"lr":args.lr, "dataset":'chexpert',"k":1, "beta":args.beta, 'z_dim':args.z_dim}
    with mlflow.start_run():
        if args.prior ==  'gaussian':
            model = VAE(config).cuda()
            dir = False
        else:
            model= dirVAE(config).cuda()
            dir = True

        clf = LogisticRegression(args.z_dim).cuda()
        clf_optimizer = Adam(clf.parameters(), lr=1e-4)

        model.load_state_dict(torch.load("/MULTIX/DATA/dir_weights09_no_ilr/dir_vae_mse_stable.pth")) # "/MULTIX/DATA/dir_vae_mse_stable_conc099_no_ilr.pth"
            # torch.load("/MULTIX/DATA/dir_vae_kld_stable_conc05.pth")) #dir_vae_kld_stable_conc5.pth"))
      #  model.load_state_dict(torch.load("/MULTIX/DATA/dir_vae_kld_stable.pth"))
        if args.mode == 'init_clf':
            model.eval()
           # clf.load_weights(dir)

        if args.mode == 'clf':
            clf.load_weights(dir)
            model.train()

        df = pd.read_csv(args.data_csv)

        main(model, clf, clf_optimizer, config, df, dir, args.k)
#        mlflow.end_run()



