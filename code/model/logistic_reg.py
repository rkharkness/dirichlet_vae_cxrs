import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

class LogisticRegressionUnit(torch.nn.Module):
    def __init__(self, z_dim):
        super(LogisticRegressionUnit, self).__init__()
        self.linear = torch.nn.Linear(z_dim,1)

    def forward(self, z):
        z = torch.log(z)
        z[z!=z]=0
        outputs = self.linear(z)
        return torch.sigmoid(outputs).clamp(min=1e-8, max=1-1e-8)

    def loss_fn(self, y, y_hat):
        bce = F.binary_cross_entropy(y, y_hat)
        bce[bce!=bce] = 0    
        return bce


class LogisticRegression(torch.nn.Module):
    def __init__(self, z_dim, k, num_classes=4):
        super(LogisticRegression, self).__init__()

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.k = k

        self.labels = ['No Finding', 'Lung Opacity', 'Pleural Effusion', 'Support Devices']

        self.models = torch.nn.ModuleList()
        for i in range(num_classes):
            self.models.append(LogisticRegressionUnit(self.z_dim))

    def one_vs_rest_mode(self, training_class):
        self.set_eval()
        self.models[training_class].train()

    def set_eval(self):
        for model in self.models:
            model.eval()   

    def set_train(self):
        for model in self.models:
            model.train()

    def forward(self, x):
        multilabel_pred = []
        multilabel_prob = []
        for idx, model in enumerate(self.models):
            pred = model(x)
            multilabel_prob.append(pred)
            multilabel_pred.append(np.round(pred.detach().cpu().numpy()))

        return np.squeeze(np.stack(multilabel_pred, axis=-1)), torch.squeeze(torch.stack(multilabel_prob, axis=-1))

    def accuracy(self, y, y_hat):
        return accuracy_score(y, y_hat)

    def loss_fn(self, y, y_hat, training_class):
        if training_class > 3:
            loss = F.binary_cross_entropy(y, y_hat, reduction="mean")

        else:
            loss = F.binary_cross_entropy(y[training_class], y_hat[training_class], reduction="mean")

        return loss

    def save_model(self, model, dir):
        if dir:
            prefix = 'dir'
            path = f"/MULTIX/DATA/{prefix}_init_lr_k{self.k}_conc05.pth"
        else:
            prefix = 'gau'
            path = f"/MULTIX/DATA/{prefix}_init_lr_k{self.k}.pth"
        torch.save(model.state_dict(), path)

    def load_weights(self, dir):
        if dir:
            prefix = 'dir' #dir_clf_no_vae1
 #           PATH = f"/MULTIX/DATA/{prefix}_lr_conc05.pth"
            PATH = f"/MULTIX/DATA/{prefix}_init_lr_k{self.k}_conc05.pth"
#            PATH = f"/MULTIX/DATA/dir_mlp_with_vae_joint_conc1.pth"   #{prefix}_with_vae_joint_conc1_no_downweight.pth" #f"/MULTIX/DATA/dir_clf_no_vae1.pth" #f"/MULTIX/DATA/{prefix}_clf_with_vae.pth" # f"/MULTIX/DATA/{prefix}_mlp_with_vae_conc1.pth" #dir_clf_with_vae_conc1.pth" #"/MULTIX/DATA/dir_clf_no_vae1.pth"
        else:
            prefix = 'gau'
            PATH = f"/MULTIX/DATA/{prefix}_init_lr_k{self.k}.pth" #"/MULTIX/DATA/{prefix}_clf_final.pth"
            
        pretrained_dict = torch.load(PATH)
        pretrained_dict = {key.replace("models.", ""): value for key, value in pretrained_dict.items()}
 #       pretrained_dict = {key[1:]: value for key, value in pretrained_dict.items()}
        self.models.load_state_dict(pretrained_dict)