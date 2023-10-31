import torch
torch.set_printoptions(profile="full")

import numpy as np
from tqdm import tqdm
import json
import torch.nn.functional as F

import matplotlib.pyplot as plt

from utils import show_recons
import random

import mlflow
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_score, recall_score

from tensorboardX import SummaryWriter

LOG_DIR = "rharkness"
logger = SummaryWriter(LOG_DIR)

def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X-mean) / std

def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)

#init_all(model, torch.nn.init.normal_, mean=0., std=1) 

def class_wise_acc(multilabel_prob, multilabel_pred, y):
#   multilabel_pred = multilabel_pred.detach().cpu().numpy()
   y=y.detach().cpu().numpy()
   labels = ['No Finding','Lung Opacity', 'Pleural Effusion', 'Support Devices']
   class_metrics = {'acc':{}, 'prec':{}, 'recall':{}}
   for i in range(4):
         class_prob = multilabel_prob[...,i]
         class_pred = multilabel_pred[...,i]
         class_pred = np.where(np.isnan(class_pred), 0, class_pred)
         class_y = y[...,i]
         class_metrics['acc'][labels[i]] = accuracy_score(class_y, class_pred)
         class_metrics['prec'][labels[i]] = precision_score(class_y, class_pred)
         class_metrics['recall'][labels[i]] = recall_score(class_y, class_pred)
   return class_metrics

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )

        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def training_loop(model, clf, training_class,  train_loader, optimizer, clf_optimizer, epoch, sample_idx, dirichlet=False):
    model.train()
    clf.set_train()

    train_loss = 0
    train_kld_loss = 0
    train_bce_loss = 0
    train_clf_loss = 0

    acc = 0
    hamming = 0
    metrics = ['acc', 'prec', 'recall']
    labels = ['No Finding','Lung Opacity', 'Pleural Effusion', 'Support Devices']
    epoch_class_metrics = {m:{i:[] for i in labels} for m in metrics}

    for batch_idx, (batch_x, batch_y) in enumerate(tqdm(train_loader)):
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()

        batch_y = torch.squeeze(batch_y)
        if dirichlet:
            recon_x, alpha, z = model(batch_x)
            bce_loss, kld_loss, loss = model.loss_fn(recon_x, batch_x, alpha, epoch)
        else:
            z, recon_x, mu, logvar = model(batch_x)
            bce_loss, kld_loss, loss = model.loss_fn(recon_x, batch_x, mu, logvar, epoch)

        if model.get_mode(epoch) == 'clf' or model.get_mode(epoch)=='init_clf':
            multilabel_pred, multilabel_prob = clf(z)
            if training_class > 3:
                clf.set_train()
            else:
                clf.one_vs_rest_mode(training_class)

            y = batch_y.detach().cpu().numpy()
            clf_loss = clf.loss_fn(multilabel_prob, batch_y, training_class)
     
            accuracy = clf.accuracy(multilabel_pred, y)
            hamming_acc = hamming_score(multilabel_pred, y)
            loss = clf_loss + loss # consider clf on top of prior and recon
            acc += accuracy
            hamming +=hamming_acc
            train_clf_loss += clf_loss.item()
 
            if model.get_mode(epoch) == 'clf' or  model.get_mode(epoch) == 'init_clf':            
                class_met_dict = class_wise_acc(multilabel_prob.detach(), multilabel_pred, batch_y)
                for m in class_met_dict.keys():
                    epoch_class_metrics[m] = class_met_dict[m]
           
            clf_optimizer.zero_grad()
            clf_loss.backward(inputs=list(clf.parameters()), retain_graph=True)

        if model.get_mode(epoch) != 'init_clf':
            optimizer.zero_grad()
            loss.backward(inputs=list(model.parameters()))
            optimizer.step()

        if model.get_mode(epoch) == 'clf' or  model.get_mode(epoch) == 'init_clf':
            clf_optimizer.step()      

        train_bce_loss += bce_loss.item()
        train_kld_loss += kld_loss.item()
        train_loss += loss.item()
 
        if batch_idx == 1:
            train_enc_recon = show_recons(model, 'train', batch_x, recon_x, epoch, 'z')
            train_noise_recon = show_recons(model, 'train', batch_x, recon_x, epoch, 'noise')

            mlflow.log_figure(train_enc_recon, f'train_enc_recon_{epoch}.png')
            mlflow.log_figure(train_noise_recon, f'train_noise_recon_{epoch}.png')

    train_bce_loss /= len(train_loader.dataset)
    train_kld_loss /= len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)

    acc /= len(train_loader.dataset)
    train_clf_loss /= len(train_loader.dataset)
    hamming /= len(train_loader.dataset)

    for m in metrics:
        epoch_class_metrics[m] = {k: np.mean(v) for k, v in epoch_class_metrics[m].items()}

    print('====> TRAIN - Epoch: {} | Average BCE loss: {:.4f} | Average KLD loss: {:.4f} | Average Loss: {:.4f} | LR Loss {:.10f} | LR Accuracy {:.4f} | Hamming Score {:.4f}'.format(
        epoch,
        train_bce_loss,
        train_kld_loss,
        train_loss, train_clf_loss, acc, hamming))

    print(epoch_class_metrics)

    # labels = ['No Finding','Lung Opacity', 'Pleural Effusion', 'Support Devices']

    return [train_bce_loss, train_kld_loss, train_loss], model, clf, epoch_class_metrics


def validation_loop(model, clf, training_class, clf_optimizer, val_loader, epoch, sample_idx, dirichlet=False):
    model.eval()
    clf.set_eval()

    val_bce_loss = 0
    val_kld_loss = 0
    val_loss = 0
    
    val_clf_loss = 0
    hamming = 0
    acc = 0

    metrics = ['acc', 'prec', 'recall']
    labels = ['No Finding','Lung Opacity', 'Pleural Effusion', 'Support Devices']
    epoch_class_metrics = {m:{i:[] for i in labels} for m in metrics}

    for batch_idx, (batch_x, batch_y) in enumerate(tqdm(val_loader)):
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        batch_y = torch.squeeze(batch_y)
        if dirichlet:
            recon_x, alpha, z = model(batch_x)
            bce_loss, kld_loss, loss = model.loss_fn(recon_x, batch_x, alpha, epoch)
        else:
            z, recon_x, mu, logvar = model(batch_x)
            bce_loss, kld_loss, loss = model.loss_fn(recon_x, batch_x, mu, logvar, epoch) #loss = bce + beta * kld

        if model.get_mode(epoch) == 'clf' or  model.get_mode(epoch) == 'init_clf':
            multilabel_pred, multilabel_prob = clf(z)
            clf_loss = clf.loss_fn(multilabel_prob, batch_y, training_class)

            y = batch_y.detach().cpu().numpy()

            accuracy = clf.accuracy(multilabel_pred, y)
            hamming_acc = hamming_score(multilabel_pred, y)
            loss = clf_loss + loss # consider clf on top of prior and recon
            acc += accuracy
            hamming += hamming_acc
            val_clf_loss += clf_loss.item()

            if model.get_mode(epoch) =='clf' or  model.get_mode(epoch) == 'init_clf':     
                class_met_dict = class_wise_acc(multilabel_prob.detach(),multilabel_pred, batch_y)
                for m in class_met_dict.keys():
                    epoch_class_metrics[m] = class_met_dict[m]

        val_bce_loss += bce_loss.item()
        val_kld_loss += kld_loss.item()
        val_loss += loss.item()

        if batch_idx == 1:
            val_enc_recon = show_recons(model, 'val', batch_x, recon_x, epoch, 'z')
            val_noise_recon = show_recons(model, 'val', batch_x, recon_x, epoch, 'noise')

            mlflow.log_figure(val_enc_recon, f'val_enc_recon_{epoch}.png')
            mlflow.log_figure(val_noise_recon,f'val_noise_recon_{epoch}.png')

    val_bce_loss /= len(val_loader.dataset)
    val_kld_loss /= len(val_loader.dataset)
    val_loss /= len(val_loader.dataset) 
    val_clf_loss /= len(val_loader.dataset)
    acc /= len(val_loader.dataset)
    hamming /= len(val_loader.dataset)

    for m in metrics:
        epoch_class_metrics[m] = {k: np.mean(v) for k, v in epoch_class_metrics[m].items()}

    print('====> VAL - Epoch: {} | Average BCE loss: {:.4f} | Average KLD loss: {:.4f} | Average Loss: {:.4f} | LR Loss: {:.10f} | LR Accuracy {:.4f} | Hamming Score {:.4f}'.format(
        epoch,
        val_bce_loss,
        val_kld_loss,
        val_loss, val_clf_loss, acc, hamming))
    
    print(epoch_class_metrics)

    return [val_bce_loss, val_kld_loss, val_loss, val_clf_loss], epoch_class_metrics  


def train(model, clf, clf_optimizer, dataloaders, optimizers, dir, EPOCHS):
    clf_metrics = {'train':{}, 'val':{}}

    best_val_clf_loss = 1e10 
    no_improvement = 0
    training_class = 0
    labels = ['No Finding','Lung Opacity', 'Pleural Effusion', 'Support Devices']
    for epoch in range(1, EPOCHS + 1):
        if training_class < 4:
            print(f"training class {labels[training_class]}")
        else:
            print("training all classes")

        random_idx =1 
        train_losses, model, clf, clf_epoch_metrics = training_loop(model, clf, training_class, dataloaders['train'], optimizers, clf_optimizer, epoch, random_idx, dir)
        clf_metrics['train'][epoch] = clf_epoch_metrics
        
        mlflow.log_metric('train_bce_loss', train_losses[0], step=epoch)
        mlflow.log_metric('train_kld_loss', train_losses[1], step=epoch)
        mlflow.log_metric('train_loss', train_losses[2], step=epoch)

        val_losses, val_clf_epoch_metrics = validation_loop(model, clf, training_class, clf_optimizer, dataloaders['val'], epoch, random_idx, dir)   
        clf_metrics['val'][epoch] = val_clf_epoch_metrics
 
#        model.save_model(model, epoch)
        mlflow.log_metric('val_bce_loss', val_losses[0], step=epoch)
        mlflow.log_metric('val_kld_loss', val_losses[1], step=epoch)
        mlflow.log_metric('val_loss', val_losses[2], step=epoch)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if training_class < 4:
            print(f"training class {labels[training_class]}")
        else:
            print("training all classes")

        if model.get_mode(epoch) == 'clf' or model.get_mode(epoch) == 'init_clf':
            if np.round(val_losses[3],4) < np.round(best_val_clf_loss,4):
                best_val_clf_loss = val_losses[3]
                if model.get_mode(epoch) == 'clf':
                    model.save_model(model, epoch)
                clf.save_model(clf, dir)
                no_improvement = 0
                print(f'no improvement for {no_improvement} epochs')

            else:
                no_improvement += 1
                print(f'no improvement for {no_improvement} epochs')
                if no_improvement == 10:
                    training_class +=1
                    best_val_clf_loss = 1e10 

                    if training_class == 6:
                        break
        else:
            model.save_model(model,epoch)

        with open('vae_clf_lr.json', 'w') as fp:
            json.dump(clf_metrics, fp)
