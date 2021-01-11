import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import skimage.io
import glob
import re # REEEEEEe

import cellpose

from cellpose import models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from progress.bar import Bar

REG = r'\/([0-9]*)\/3765_([A-Z][0-9]{2})_T0001F001L01A01Z01C01' # regex for metadata

def get_wellID(filename, reg=REG):
    wellId = re.findall(reg, filename)
    return wellId[0]

def get_data(path, model, reg, dim=30):
    """
    """
    files = glob.glob(path+'/*.tiff')
    data = {
        'filename':[],
        'wellID': [],
        'counts':[]
    }

    for f in files:
        wellID = re.findall(reg, f)[0]
        img = skimage.io.imread(f)
        masks, flows, styles, diams = model.eval(img, diameter=dim, channels=[0,0], do_3D=False) # gen masks
        masks = np.array(masks) # conveter to np
        count = masks.max()

        # append data into dict
        data['filename'].append(f)
        data['wellID'].append(wellID)
        data['counts'].append(count)

    df = pd.DataFrame.from_dict(data)
    return df

def gen_df(dirs, model, reg, dim=None):
    """
    """
    col_names = ['filename','wellID','counts']
    df = pd.DataFrame(columns=col_names)
    for i, d in enumerate(dirs):
        print(f'\nGetting data from {d}\n')
        if dim != None:
            data = get_data(d, model=model, reg=reg, dim=dim)
        else:
            data = get_data(d, model=model, reg=reg, dim=dim)
        df = df.append(data)
    
    return df

def gen_boxplot(df, index, vals, cols, title='heatmap'):
    dh = pd.pivot_table(data=df, index=index, values=vals, columns=cols)
    heat = sns.heatmap(dh)
    heat.set_title(title)
    fig = heat.get_figure()
    fig.savefig(title+'.png')