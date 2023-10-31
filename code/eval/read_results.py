import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from sklearn.metrics import roc_curve, auc
from scipy import interp


def plot_roc_curve(fpr, tpr, aucs):

    kfold_tpr = {k:[] for k in labels}
    kfold_fpr = {k:[] for k in labels}
    kfold_aucs = {k:[] for k in labels}

    for idx, i in enumerate(results_paths):
        with open(i) as f:
            data = json.load(f)

        for l in labels:
            kfold_tpr.append(data['tpr'][l])
            kfold_fpr.append(data['fpr'][l])
            kfold_aucs.append(data['auc'][l])

    fig, ax = plt.figure()
    for l in labels:
        mean_tpr = np.mean(kfold_tpr[l], axis=0)
        mean_fpr = np.mean(kfold_fpr[l], axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(kfold_aucs[l])

        mean_tpr[-1] = 1.0

        ax.plot(mean_fpr, mean_tpr, 
            label='%s = %0.2f $\pm$ %0.2f' % (l + 'auc', mean_auc, std_auc),
            lw=1, alpha=.8)

        if l == 'Support Devices':
            ax.plot([0, 1], [0, 1], linestyle='--', lw=0.75, color='r',
            label='Chance', alpha=.6)
                
        std_tpr = np.std(tpr, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        ax.fill_between(mean_fpr, tprs_lower, tprs_upper,
                 alpha=.1)
 
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])

        ax.legend(loc='lower right', fontsize=4.5)
        ax.set_xlabel('False Positive Rate',fontsize=7)
        ax.set_ylabel('True Positive Rate',fontsize=7)
        ax.set_title('ROC Curve',fontsize=7)

    plt.savefig(f'/MULTIX/DATA/cxr_vae/kfold_roc.pdf', dpi=300, bbox_inches='tight')


results_paths = [f'/MULTIX/DATA/cxr_vae/gaussian_{i}.json' for i in range(1,4)] # create list of paths to json files

labels = ['No Finding', 'Lung Opacity', 'Pleural Effusion', 'Support Devices']

kfold_precision = {k:[] for k in labels}
kfold_recall = {k:[] for k in labels}
kfold_acc = {k:[] for k in labels}
kfold_auc = {k:[] for k in labels}
kfold_hamming = {k:[] for k in labels}
kfold_l1 = {k:[] for k in labels}

for idx, i in enumerate(results_paths):
    with open(i) as f:
        data = json.load(f)

    case_emr = data['emr']
    print('emr', case_emr)
    
    hamm = data['hamming']
    print('hamming', hamm)
    
    case_prec = data['case_prec']
    print('case prec', case_prec)
    
    case_recall = data['case_recall']
    print('case recall', case_recall)
    print(data.keys())

    l1 = data['l1']
    print('l1', l1)

    for l in labels:
        print("\n", l)

        kfold_acc.append(data['acc'][l])
        kfold_auc.append(data['auc'][l])
        kfold_precision.append(data['prec'][l])
        kfold_recall.append(data['recall'][l])
        
        kfold_hamming.append(data['hamming'][l])
        kfold_l1.append(data['l1'][l])


for l in labels:
    print(l)
    print('mean prec', np.mean(kfold_precision[l]))
    print('sd prec', np.std(kfold_precision[l]))

    print('mean recall', np.mean(kfold_recall[l]))
    print('sd recall', np.std(kfold_recall[l]))

    print('mean acc', np.mean(kfold_acc[l]))
    print('sd acc', np.std(kfold_acc[l]))

    print('mean auc', np.mean(kfold_auc[l]))
    print('sd auc', np.std(kfold_auc[l]))

    print('mean hamming', np.mean(kfold_hamming[l]))
    print('sd hamming', np.std(kfold_hamming[l]))

    print('mean l1', np.mean(kfold_l1[l]))
    print('sd l1', np.std(kfold_l1[l]))


        



