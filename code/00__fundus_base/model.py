'''
author: bg
goal: 
type: modelz - sklearn for workflow management + PyTorch/keras for transfer learning components + pytorch for nn modules  
how: wrapper class for workflow management (pytorch opt&loss + sklearn pipe&metrics) + ArchitectureMixin and implementation for custom architectures. 
ref: 
refactors: 
'''

from tqdm import tqdm

from sklearn.base import BaseEstimator 
from sklearn.base import ClassifierMixin, RegressorMixin, ClusterMixin
from sklearn.pipeline import Pipeline 

from sklearn.model_selection import GridSearchCV, StratifiedKFold 

## TODO: use same @ transforms 
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils.multiclass import unique_labels 

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F  
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import nnarchs 
from zdata import ZSerializableMixin 
from report import ZReporter

import cv2 ## K-Means Clustering 


class ZModel(ZSerializableMixin, BaseEstimator): ## TODO: VS internal class for nn.Module subclassing  <<< subClassing Vs has-a 
    ## cuda, optimizer, loss, evaluate 
    
    MSG_LOSS_PER_EPOCH = "[{:20s}] Train Epoch {:5d}: Loss: {:15.4f} \tn = {:5d}".format
    MSG_ACC_ON_PREDICT = "[{:20s}] Predict on n ={:5d}: Acc: {:15.4f}".format
    MSG_YHAT_Y_VALZ = "{:3d}. {:4.2f} ===> {:4.2f}".format

    def __init__(self, parent_caller_name,  nnModel=None,  ##TODO: non-nn Models and traditional ML 
                use_cuda=False,  ##TODO: pass to ZNNArch 
                epochs=3,
                loss_func=(nn.CrossEntropyLoss, {}) ,  
                optimizer=(optim.SGD, {'lr':0.001, 'momentum':0.9}) ): 
        ## setup layers and initialize model weights and biases 
        self.parent_caller_name = parent_caller_name ## for loggin purposes
        self.nnModel = nnModel
        self.epochs = epochs 
        self.loss_func = loss_func 
        self.optimizer = optimizer
        self.loss_thresh = 1e-6
        ## TODO: cuda calls  
        self.use_cuda = use_cuda 
        self.init_cuda()
    
    ### === sklearn pipeline management imple === 
    def fit(self, X, y=None): ## X is NxF, y is Nx1
        ## train model + batches 
        self.train(X, y) 
        return self

    def transform(self, X, y=None):## X is NxF, y is Nx1
        ## predict with no_grad and return list of predictions as is 
        return self.predict(X, y, log=False)

    def score(self, X_, y_=None):## y is Nx1
        yhat = self.predict(X_, y_)  
        ACC_ = self.score(yhat, y_)
        return ACC_

    ### ===  things pyTorch and cuda ==========
    def zscore(self, yhat, y_):## y is Nx1 
        ACC_ = np.array( [ int( int(y) == int( yh ) ) for y, yh in zip(y_, yhat )]).mean()  
        ZReporter.add_log_entry( self.MSG_ACC_ON_PREDICT(self.parent_caller_name, len(y_), ACC_ ) )  
        return ACC_

    def init_cuda(self):
        if self.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print('Cuda is not available')
        self.device = torch.device('cpu')
    
    ## TODO: returns + skl.metrics
    def train(self, X_train, y_train=None):
        N_ = len(X_train) 
        lossor = self.loss_func[0]( **self.loss_func[1])
        optimor = self.optimizer[0]( self.nnModel.parameters(), **self.optimizer[1])
        def fwd_pass(x, y=None, ebatch=0): 
            if (y is not None) and isinstance(y, np.ndarray): ## force tensor <<< TODO: fix at source 
                y = torch.tensor(y) 
            ## 1. train on batch 
            self.nnModel.train() ## set to training mode 
            outz = self.nnModel.forward( x )

            # print(outz)
            # print(y) 
            # print( type(outz), type(y))
            # print( f"****** SHAPEZ::: *****\n yhat ={outz[0].shape} y = {y[0].shape }")

            l_ = lossor( outz, y)   

            optimor.zero_grad()
            l_.backward() 
            optimor.step()  
            
            ## 2. evaluate batch ##TODO: ensure batches
            if ebatch % 10 == 0 or ebatch == self.epochs:
                self.nnModel.eval() 
                outz = self.nnModel.forward(x) 
                l_ = lossor( outz, y) 
            return l_.item()

        LOSS_ = 0.
        for e in range( self.epochs):
            if isinstance(X_train, DataLoader):
                pass
            else:
                if y_train is None:
                    for x in X_train: 
                        LOSS_ += fwd_pass(x, ebatch=e+1) 
                else:
                    for x, y in zip(X_train, y_train): 
                        LOSS_ += fwd_pass(x, y, e+1)
            ZReporter.add_log_entry( self.MSG_LOSS_PER_EPOCH(self.parent_caller_name, e+1, LOSS_ ,N_ ) )  
            ## TODO: better flow and when to stop 
            if LOSS_ <= self.loss_thresh:
                break 

    def predict(self, X_, y_=None, log=True, to_numpy=True): 
        ## run evaluation and return list of results as is. Score will handle metrics 
        def eval_pass(x):
            with torch.no_grad():
                self.nnModel.eval()
                outz = self.nnModel.forward(x) 
                o = torch.argmax( outz  ).cpu()
                return o.numpy() if to_numpy else o 

        yhat = []
        if isinstance(X_, DataLoader):
            pass
        else:
            for x in X_: 
                yhat.append( eval_pass(x) )
        
        if log:        
            ACC_ = self.zscore(yhat, y_) 

        return yhat 


class ZModelKMeans(ZModel): 
    def __init__(self):
        pass

    def train(self, X_train):
        pass 

    def predict(self, X_):
        pass 

### --- TBD: Skorch == sklearn + pytorch already! for now i want to moi-flow 
### --- TODO: Captum == Model intepretability for PyTorch 
class ZTrainingManager():
    MSG_GSEARCH_RESULTS = "[{:7s}] Best score = {:2.4f} estimator = {:10s} paramz = {:50s}".format ## SCORE, ESTIMATOR, PARAMZ
    ### === workflow management, permutations and hyperparam tuning configuration 
    def __init__(self, data_pipez=None, model_pipez=None): ## NO:ByNNs = setup layers and initialize model weights and biases 
        # workflow management, permutations and hyperparam tuning configuration          
        if data_pipez is not None and model_pipez is not None:
            self.build_permutationz(data_pipez, model_pipez)

    ### === setup permutationz ===
    def build_permutationz(self, data_pipez, model_pipez):
        '''
            data_pipez : list of data pipelines
            model_pipez : list of (model pipelines and grid search params) tuples 
        '''            
        self.permutationz = [] ## reset  
        d_, m_ = np.meshgrid( range( len(data_pipez)), range(len(model_pipez)) )
        for x, y in zip(d_, m_):
            for i, j in zip(x, y):
                self.permutationz.append( (data_pipez[i], model_pipez[j]) ) 

    ### === run training with skl.grid search on each permutation 
    def run(self, data_X, data_y = None, train_test_split=1., save_best=False, log=False): 
        ## for each permutation apply data and grid search

        ## 1. Deal data allocation TODO: when to train_test split 
        if isinstance(data_X, np.ndarray):
            n_train = int(len(data_X)*train_test_split)         
            train_X, test_X = data_X[:n_train],  data_X[n_train:] ## TODO: check if a ZPdData or something or array/list  
            train_y, test_y = [], [] ## hack for print :/
            if data_y is not None:
                train_y, test_y = data_y[:n_train], data_y[n_train:]  
                if log:
                    print( type(train_y[0]), train_y[0].shape )
            if log:
                print( type(train_X[0]), train_X[0].shape )
                print(f"Train-Test-Split {train_test_split}: train = {len(train_X)}, {len(train_y)} \t test = {len(test_X)}, {len(test_y)}")
        else: ## TODO: dataloader, Zdataset
            train_X, train_y  = data_X, data_y
            test_X, test_y  = [], []
            
        
        ## 2. train 
        O_ = []
        for i in tqdm( range( len( self.permutationz ) ) ):
            o = self._run_permutation(i, train_X, train_y ) 
            p = f"Perm_{i+1}"  
            O_.append( [p,*o] ) 
            # print("<<<<<<<\n", o, "\n>>>>>>>>")
            ZReporter.add_log_entry( ZTrainingManager.MSG_GSEARCH_RESULTS(f"{p} {o[0]}", o[1], *[str(i) for i in o[2:]]) ) 

        ## 3. test/validate 

        return O_ 

    def _run_permutation(self, idx, X, y , log=False):
        data_pipe, model_pipe = self.permutationz[ idx]
        model_pipe, g_paramz = model_pipe 
        def update_gsearch_param_keys(mp, gp):
            O_ =  {}
            m = mp.steps[-1][0]
            for k, v in gp.items():
                O_[ f"model_pipe__{m}__{k}" ] = v
            if log:
                print(f"\n\n***********{m}***********") 
            return O_ , m

        g_paramz , m_name = update_gsearch_param_keys(model_pipe, g_paramz)
        
        # print( data_pipe )
        dz = "__".join([str(x[0]) for x in data_pipe.steps]) 
        m_name = f"{m_name} {dz}"

        piper = Pipeline([ ('data_pipe', data_pipe),
                            ('model_pipe', model_pipe)])
        # print(f"============\n{piper}\n{g_paramz}\n==============<<<<")

        gsearch = GridSearchCV(estimator=piper,
                                param_grid=g_paramz,
                                cv=StratifiedKFold(n_splits=2), ## random_state=99, shuffle=True
                                n_jobs=1,
                                return_train_score=True) 
        gsearch.fit(X, y) 

        return (m_name, gsearch.best_score_, gsearch.best_estimator_ , gsearch.best_params_)

    def _save_best_model(self): ## TODO
        pass 





if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator 
    # check_estimator( ZModel() )  ## check adheres to sklearn interface and standards 
    # TODO: parametrize_with_checks pytest decorator  
    epochz = 3 
    N, nf, nclasses  =  12, 40, 2 #.reshape(1, -1)
    tmpX = [ torch.tensor( np.random.randint(0, 100, size=nf).reshape(1, -1).astype(np.float32) ) for i in range(N)]
    tmpY = [  x.sum()**2 for x in tmpX]
    ymu = np.array(tmpY).mean()  ##mean
    ymu = np.percentile( np.array(tmpY), 0.5) ## median 
    tmpY = [ torch.tensor( np.array([ int(y > ymu),] ).astype(np.long) ) for y in tmpY] ## TODO: qcut percentile


    tmpX = tmpX * 300
    tmpY = tmpY * 300
    n_train = int(len(tmpY)*0.75) 

    mlp = nnarchs.ZNNArchitectureFactory.mlp(nf, nclasses) 
    print(mlp )
    
    
    model = ZModel( "Tryzex", mlp, epochs=epochz, 
                    loss_func=(nn.CrossEntropyLoss, {} ), 
                    optimizer=(optim.SGD, {'lr':0.001, 'momentum':0.9} ) )
    print( model )
 
    model.train(tmpX[:n_train], tmpY[:n_train]) 

    yhat = model.predict(tmpX[n_train:], tmpY[n_train:] , log=True)
    # model.score(yhat, tmpY[n_train:])

    c = "="
    print(f"{c*10} End Model --> Dumping to File {c*10}\n")

    fpath = "./tester.zmd"
    model.dump(fpath )
    model2 = ZModel("Tryzex_2",) 
    model2.load(fpath)
    model2.predict(tmpX, tmpY , log=True)

    model2.epochs =5
    model2.fit(tmpX[:n_train], tmpY[:n_train])
    yhat = model2.transform(tmpX[n_train:], tmpY[n_train:])
    model2.zscore(yhat, tmpY[n_train:])

    print( f"****** SHAPEZ::: *****\n yhat ={yhat[0].shape} y = {tmpY[0].shape }")

    print(f"{c*10} End Model2 <-- Loaded from file, predicted, retrained and predicted {c*10}\n")

    epochz = 5
    # print( type(tmpX), type(tmpY))
    # print( len(tmpX), len(tmpY))
    tmpX = tmpX * 3
    tmpY = tmpY * 3
    n_train = int(len(tmpY)*0.75) 
    # print( len(tmpX), len(tmpY))
    mlp = nnarchs.ZNNArchitectureFactory.mlp(nf, nclasses, {'n_layers':3} ) 
    model3 = ZModel( "Tryzex_3", mlp, epochs=epochz, 
                    loss_func=(nn.HingeEmbeddingLoss, {} ), 
                    optimizer=(optim.SGD, {'lr':0.001, 'momentum':0.9} ) )
    # print( model3 )
    model3.fit( tmpX[:n_train], tmpY[:n_train])
    yhat = model3.transform(tmpX[n_train:], tmpY[n_train:])
    model3.zscore(yhat, tmpY[n_train:])

    print(f"{c*10} End Model3 <-- new, hyperparams {c*10}\n")
    # ## TODO: classifier/regressor/clusterer/etc Mixin requirements
    # piper = Pipeline(['model', model2])
    # print( piper )

    # piper.fit_transform(tmpX, tmpY)


    print(f"\n{c*10} Starting TrainingManager with Grid Search {c*10}\n")
    import preprocess, extract 
    from sklearn.preprocessing import StandardScaler, PowerTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm 

    dpipez = [Pipeline([('scaler', StandardScaler()), ]),  
                Pipeline([('power', PowerTransformer()),])
                ]
    mpipez = [ ( Pipeline([ ('flatten', preprocess.Flattenor()), ('svm', svm.SVC() ) ]), {'kernel':('linear', 'rbf'), 'C':[1, 10]}) ,  ## 
                ( Pipeline([ ('flatten', preprocess.Flattenor()),('logit', LogisticRegression() ) ]), {'C':[1,10]} ), ##
                (Pipeline([('reshaper', preprocess.Reshapeor( (1, -1)) ), ('tensorfy', preprocess.ToTensor() ),('zmodel', model2)]), {}) 
             ] #*tmpX[0].shape

    print( mpipez)

    mgr = ZTrainingManager() 
    mgr.build_permutationz(data_pipez=dpipez, model_pipez=mpipez)
    mgr.run( [x.cpu().numpy().ravel() for x in tmpX], [y.cpu().numpy().ravel() for y in tmpY] , train_test_split=1.)
    print(f"{c*10} End ZTrainingManager {c*10}\n")