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
    c = "="

    report.ZReporter.start("Vessels_Segmentation")

    pdstats = zdata.PdDataStats(
                    {zdata.PdDataStats.DATA_DICT_RECORDZ_KEY: content.CHASEDB_FUNDUS_CONTENT_FPATH,
                    zdata.PdDataStats.DATA_DICT_HAS_HEADERZ_KEY: True,
                    'rec_parser': utilz.FileIO.row_parser                     
                    },
                     ftype=zdata.PdDataStats.TYPE_TXT_LINES_FILE ) 
    
    #pdstats.dframe
    ### =============
    X_data = pdstats.dframe 
    y_data = pdstats.dframe['i-L/R'] 

    ### =============== Pipeline 
    # load image file --> preproc: equalize hist, --> proc: color channelz, 
    ## 1. preprocessing = load, rescale, crop 
    preproc_pipe = [('load_file', preprocess.LoadImageFileTransform('fpath', resize=(105,105), crop_ratio=0.97)), ]
    ## 2. postprocessing = morphological seal, median smooth noise  +++ reshape: flatten for SVM etc 
    postproc_pipe = []
    ## 3. process permutaitons = color channels sep (Lab @ ry combo) || 
    color_c_pipe = [('color_channelz', extract.ColorChannelz())]


    data_pipez = [Pipeline(preproc_pipe + color_c_pipe + postproc_pipe)]
    model_pipez = [ ( Pipeline([ ('flatten', preprocess.Flattenor()), ('svm', svm.SVC() ) ]), {'kernel':('linear', 'rbf'), 'C':[1, 10]}) ,  ## 
                ( Pipeline([ ('flatten', preprocess.Flattenor()),('logit', LogisticRegression() ) ]), {'C':[1,10]} ), ##
             ] 

    # print( dpipez)


    mgr = model.ZTrainingManager() 
    mgr.build_permutationz(data_pipez=data_pipez, model_pipez=model_pipez)
    mgr.run( X_data , y_data, train_test_split=1.)
    print(f"{c*10} End ZTrainingManager {c*10}\n")


