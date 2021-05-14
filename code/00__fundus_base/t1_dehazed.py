'''
author: bg
goal: workflow and architecture 
type: fundus EDA specific setup
how: use mflow as base 
ref: 
refactors: abstract and move common components to mflow 
'''

# from mflow import utilz, report 
# from mflow import model, nnarchs, learning_bits 
# from mflow import zdata, preprocess, extract

from sklearn.pipeline import Pipeline

import model, nnarchs, zdata, content, utilz, preprocess, extract, report 

import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F  
import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer , OneHotEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn import svm 



if __name__ == "__main__":    
    print("Starting")

    ## 1. FETCH DATA INTO PD + SETUP REPORTING 
    report.ZReporter.start("fundusEDA")

    pdstats = zdata.PdDataStats(
                    {zdata.PdDataStats.DATA_DICT_RECORDZ_KEY: content.STARE_FUNDUS_CONTENT_FPATH,
                    zdata.PdDataStats.DATA_DICT_HAS_HEADERZ_KEY: True,
                    'rec_parser': utilz.FileIO.row_parser                     
                    },
                     ftype=zdata.PdDataStats.TYPE_TXT_LINES_FILE ) 
    
    dframe = pdstats.dframe.sample(n=130) 
    X_data = dframe  
    y_data = dframe['Normal'].values.astype(np.float32) ##TODO: 'dcodez_short'
    print("Loaded into PdFrame data of size: ", len(dframe) , " and into y_data of size ", len(y_data) )  
    print( dframe.columns )

    ### Setup y_label : n-ary classification 

    ## 2. PIPELINEZ 
    loader_p = [ ('fetch_img', preprocess.LoadImageFileTransform('fpath', crop_ratio=0.75) ), ]
    reshapeor_1 = [ ('flatten', preprocess.Flattenor()), ]
    funduzor_1 = [ ('funduzor', extract.FundusColorChannelz() ),]
    scaler_p = [('scaler', StandardScaler()), ]

    tmpz = Pipeline(loader_p + funduzor_1 ).transform(X_data)
    print( len(tmpz), tmpz[0].shape)
    # _ = [print(f"{t.shape}") for t in tmpz]
    utilz.Image.plot_images_list( [t[:,:,1:] for t in tmpz], nc=5, cmap=None)
