from dataloader import make_dataloaders
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

plt.style.use('science')
plt.rcParams.update({'font.size': 22})


def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: http://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
#    from itertools import zip
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, fontsize=22, **kw)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels):
    '''
    Inspired by:
    - http://stackoverflow.com/a/16124677/395857 
    - http://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out

    fig, ax = plt.subplots()    
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='Blues', vmin=0.0, vmax=1.0)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    ax.set_xlabel(xlabel, rotation=45, fontsize=16)
    ax.set_ylabel(ylabel, rotation=45, fontsize=16)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # resize 
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(40, 20))


def cm2inch(*tupl):
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

df = pd.read_csv("/MULTIX/DATA/chexpert_df.csv")

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
df.iloc[10000:15000]['split'] = 'val'

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_df = df[df["split"] == 'test']
#   print(df.columns)
print(df['split'].value_counts())

label_data = test_df[['No Finding', 'Lung Opacity', 'Pleural Effusion', 'Support Devices']].values
# Compute cooccurrence matrix 
cooccurrence_matrix = np.dot(label_data.transpose(),label_data)
print('\ncooccurrence_matrix:\n{0}'.format(cooccurrence_matrix)) 
# Compute cooccurrence matrix in percentage
# FYI: http://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
#      http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero/32106804#32106804

cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
with np.errstate(divide='ignore', invalid='ignore'):
    cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal[:, None]))
print('\ncooccurrence_matrix_percentage:\n{0}'.format(cooccurrence_matrix_percentage))
label_headers = ['No Finding', 'Lung Opacity', 'Pleural Effusion', 'Support Devices']
# Add count in labels
label_header_with_count = [ '{0} \n ({1})'.format(label_header, cooccurrence_matrix_diagonal[label_number]) for label_number, label_header in enumerate(label_headers)]  
print('\nlabel_header_with_count: {0}'.format(label_header_with_count))

# Plotting
x_axis_size = cooccurrence_matrix_percentage.shape[0]
y_axis_size = cooccurrence_matrix_percentage.shape[1]
title = "Co-occurrence matrix\n"
xlabel= ''#"Labels"
ylabel= ''#"Labels"
xticklabels = label_header_with_count
yticklabels = label_header_with_count
heatmap(cooccurrence_matrix_percentage, title, xlabel, ylabel, xticklabels, yticklabels)
plt.savefig('test_cooccurence.pdf', dpi=300, format='pdf', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
#plt.show()