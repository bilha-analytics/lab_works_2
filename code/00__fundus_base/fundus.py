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

    epochz = 4
    N, nf, nclasses  =  len(y_data), 224*224, 2 #.reshape(1, -1)
    mlp = nnarchs.ZNNArchitectureFactory.mlp(nf, nclasses, {'n_layers':3} ) 
    zmodel = model.ZModel( "FundusEDA", mlp, epochs=epochz, 
                    loss_func=(nn.CrossEntropyLoss, {} ), 
                    optimizer=(optim.SGD, {'lr':0.001, 'momentum':0.9} ) )
    print("Setup ZModel @ basic MLP") 


    base_data_pipe_pre = [ ('fetch_img', preprocess.LoadImageFileTransform('fpath') ), ]
    base_data_pipe_post = [ ('flatten', preprocess.Flattenor()), ]             

    print(f"\n{c*10} Starting TrainingManager with Grid Search {c*10}\n")  
    dpipez = [Pipeline( base_data_pipe_pre+[('basic_green', extract.ColorChannelz()),]+base_data_pipe_post +[('scaler', StandardScaler()), ]),
                Pipeline( base_data_pipe_pre+[('basic_green', extract.ColorChannelz()),]+base_data_pipe_post +[('power', PowerTransformer()), ]),  
                ## TODO: recheck size remaps
                # Pipeline( base_data_pipe_pre+[('color_chan', extract.FundusColorChannelz() ),]+base_data_pipe_post +[('scaler', StandardScaler()), ] ),
                # Pipeline( base_data_pipe_pre+[('eigenz_chan', extract.EigenzChannelz(topn=70) ),]+base_data_pipe_post +[('scaler', StandardScaler()), ] ),
                # Pipeline( base_data_pipe_pre+[('patch_chan', extract.PatchifyChannelz(nx_patchez=12) ),]+base_data_pipe_post +[('scaler', StandardScaler()), ] )
                ]
                
    mpipez = [ ( Pipeline([ ('flatten', preprocess.Flattenor()), ('svm', svm.SVC() ) ]), {'kernel':('linear', 'rbf'), 'C':[1, 10]}) ,  ## 
                ( Pipeline([ ('flatten', preprocess.Flattenor()),('logit', LogisticRegression() ) ]), {'C':[1,10]} ), ##
                # (Pipeline([('reshaper', preprocess.Reshapeor( (1, -1)) ), ('tensorfy', preprocess.ToTensor() ),('zmodel', zmodel)]), {}) 
             ] #*tmpX[0].shape

    # print( dpipez)

    mgr = model.ZTrainingManager() 
    mgr.build_permutationz(data_pipez=dpipez, model_pipez=mpipez)
    mgr.run( X_data , y_data, train_test_split=1.)
    print(f"{c*10} End ZTrainingManager {c*10}\n")

