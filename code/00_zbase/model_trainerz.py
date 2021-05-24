'''
author: bg
goal: 
type: shared/utilz formodelz @@ TL, Common archs,  
how: 
ref: 
refactors: 
'''


from functools import total_ordering
from tqdm import tqdm 

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline 
from sklearn import metrics as skmetrics 
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import torch 
from torch import nn
from torch import optim 

from zreport import ZReporter 

class ModelNetwork(BaseEstimator, ClassifierMixin):
    '''
    TODO: ClassifierMixin 
    Once a model is assembled not add training related functionality
        loss funcs, optimization, residuals, graph Vs CNN Vs DNN etc 
        Using SKLearn for training b/c grid search and pipelines. So this wraps torch models as well

    @inputs:    model arch, loss func+argz, optim func+argz, 
    @outputs:   training paramz/weights, training loss, self @ pipeline chain 
    @actions:   a train run, an eval run, 
                a predict run, 
                fit=NOOP, transform=train with or without eval, score=predict and accuracy  
                    Vs fit=train with or without eval, transform=predict, score=predict and accuracy  
                    grid_search @ fit and score --> fit=train, transform=predict 
                cuda deal? 
    ''' 

    def __init__(self, model, 
                loss_func_argz_tuple=(nn.CrossEntropyLoss, {} ),  
                optim_func_argz_tuple=(optim.SGD, {'lr':1e-3, 'momentum':.9}), 
                scoring_func = skmetrics.accuracy_score, 
                model_trainer_callback=None, ## pass back training results 
                posttrain_callback=None,
                use_cuda = False ): 
        self.model = model 
        self.loss_func = loss_func_argz_tuple[0]( **loss_func_argz_tuple[1] )
        self.optim_func = optim_func_argz_tuple[0]( model.parameters(), **optim_func_argz_tuple[1] )  
        self.scoring_func = scoring_func 
        self.posttrain_callback = posttrain_callback 
        self.model_trainer_callback = model_trainer_callback  
        self.init_cuda(use_cuda) 

        ## WHAT?? TODO: update __dict__
        self.loss_func_argz_tuple =loss_func_argz_tuple
        self.optim_func_argz_tuple = optim_func_argz_tuple
        self.use_cuda = use_cuda

    def init_cuda(self, use_cuda): 
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print('Cuda is not available')
        self.device = torch.device('cpu')

    ## ==== 1. Sklearn grid search 
    def fit(self, X_, y_=None):
        running_loss = self.train(X_, y_) 
        if self.model_trainer_callback: ## pass back running loss to trainer 
            self.model_trainer_callback(running_loss, len(X_) )  
        return self 

    def transform(self, X_, y_=None): 
        O_ = self.predict(X_, y_)  
        return O_ 

    def score(self, yhat, y_):    
        # print(y_.shape )
        # print(yhat[0].shape) 
        yhat = np.array([y.detach().cpu().numpy().max() for y in yhat] ) 
        y_ = y_.detach().cpu().numpy()
        # print("=========SCORING 1========", type(y_), type(yhat), y_.shape, yhat.shape )
        return np.mean(  np.abs(y_ - yhat )/yhat.max() )# 
        # return self.scoring_func( y_, yhat , normalize=True)
        

    ## ==== 2. PyTorch train and eval/predict 
    def train(self, X_, y_=None): ##X_ is  a list of observations even iff single observation 
        ## operate on a single instance or a batch X_ = (n,c,h,w) or something like (n, x_i)
        running_loss = 0 
        y_ = [None for _ in range( len(X_) ) ] if y_ is None else y_  ## for case of unsupervised or something 
        # print( "****** X_ size = ",len(X_) )
        for x, y in zip(X_, y_):  
            # 0. prep y
            # print('b4: ', y) 
            # y = torch.tensor( [y,], ).reshape(1,1) # [y,] ) ##TODO: fix this for now force #y.reshape(1, *y.shape) 
            # y = y.reshape(1,1,*y.shape )
            y = y.reshape(1,*y.shape )
            # print(y.shape, " >>> ", y.item() )
            # print('after: ',y.shape, y)
            # a. zero the grads
            self.optim_func.zero_grad() 
            # b. set model to training mode
            self.model.train()
            # c. forward pass, calc loss, backprop, 
            x = self.model( x ) 
            # print('@LOSS: ', x.shape, y.shape )
            # print('AS VIEWS: ', x.shape, y.view(-1).reshape(1, -1).shape )
            # loss = self.loss_func(x, y.view(-1).reshape(1,-1) ) #.squeeze(1) )
            loss = self.loss_func(x, y ) #.squeeze(1) )
            loss.backward()
            self.optim_func.step() 
            # d. update aggregates and reporting 
            running_loss += loss.item() 
            # e. iff eval
            # f. iff other training activity that's model/arch specific TODO: what paraz for now, x and loss
            if self.posttrain_callback:
                self.posttrain_callback(x, loss) 

        return running_loss 

    def predict(self, X_):
        O_ = []
        with torch.no_grad():
            # a. go into eval mode and forward pass 
            self.model.eval() 
            O_ = self.model(X_) 
            # b. return predictions TODO: who to deal with argmax Vs all now at caller 
        return O_ 


class TrainingManager():
    '''
    Manage training process @ data fetch, grid search, per model train+validate, report metrics, 
        for any training you gotta load data, hyperparam tune, train model 
        Leave training specifics e.g. loss_func, optim_func,  to model network/arch wrapper --> so can change training for different approaches as per arch design etc
    @inputs:    model_pipez, data_pipez, epochz, batch_size, train_test_val_ratio, metrics_menu_option 
    @outputs:   per permute best grid search model results and metrics 
    @actions:   permute model-data pipez, 
                deal batching and train-test split etc, 
                train for epochz, when to break/determining convergence thresh
                track training metrics 
                freeze/pickle checkpoints 
    @TODO:      menu of metrics and sync pytorch sklearn
    ''' 

    ## menu of metrics 
    _METRICS = {
        # defaultz = all  ???
        'confusion' : skmetrics.confusion_matrix,
        'classify-all': skmetrics.classification_report,

        # 1. classification 
        'acc' : skmetrics.accuracy_score,
        'f1' : skmetrics.f1_score ,
        'precision' : skmetrics.precision_score,
        'recall': skmetrics.recall_score, 
        'roc-auc': skmetrics.roc_auc_score, 

        # 2. clustering 
        'adj_rand_score': skmetrics.adjusted_rand_score, 
        'mutual_info' : skmetrics.mutual_info_score, 

        # 3. regression 
        'r2' : skmetrics.r2_score, 
        'explained-var': skmetrics.explained_variance_score,
    }

    def __init__(self, name, metrics='full'):                
        self.metrics = self._METRICS.get( metrics, None)  ## if None use all ?? 
        ZReporter.start(name) 
        ## TODO: sanitize
        self.running_loss = 0             
        self.epochz = 0
        self.current_epoch = 0 
        self.permutationz = [] ## reset  

    ### === setup permutationz ===
    def _build_permutationz(self, data_pipez, model_pipez):
        '''
            data_pipez : list of data pipelines
            model_pipez : list of (model pipelines and grid search params) tuples 
        '''            
        d_, m_ = np.meshgrid( range( len(data_pipez)), range(len(model_pipez)) )
        for x, y in zip(d_, m_):
            for i, j in zip(x, y):
                self.permutationz.append( (data_pipez[i], model_pipez[j]) ) 

    def training_callback(self, loss, n):
        self.running_loss += loss 
        MSG_LOSS_PER_EPOCH = "Train Epoch {:d}/{:d}: Loss: {:15.4f}/{:f} (current/running) \tn = {:5d}".format
        ZReporter.add_log_entry( 
            MSG_LOSS_PER_EPOCH(
                self.current_epoch, self.epochz, loss, self.running_loss, n
            ) )  
   
    def posttraining_callback(self, x_, loss): ## this belongs to arch/network not training manager 
        pass 

    def run_permutez(self, data_pipez, model_pipez, X_data, y_data=None, 
                    epochz=3, 
                    batch_size=64, train_test_val_split=(.7, .2, .1) ): 

        MSG_GSEARCH_RESULTS = "[{:7s}] Best score = {:2.4f} <<< paramz = {:50s}".format ## SCORE, ESTIMATOR, PARAMZ <-- estimator = {:10s}
        ##TODO: fix
        self._build_permutationz(data_pipez, model_pipez) # permutes into data-model pairs 
        ## 1. Train-test-val data split and setup dataloader/dispatcher -- sync pytorch and sklearn 
        ## TODO: fix epochz init cycle Vs training callback etc 
        self.epochz = epochz

        ## 2. grid search on permutationz 
        O_ = []
        for p, (data_p, model_p) in tqdm( enumerate(self.permutationz, 1) ) :
            self.running_loss = 0             
            self.current_epoch = 0 
            for e in range(self.epochz):     ##TODO: recheck logic, saving per batch Vs epoch Vs per permute  
                self.current_epoch = e+1 
                o = self.run( data_p, model_p, X_data, y_data) 
            O_.append( o ) 
            ZReporter.add_log_entry( MSG_GSEARCH_RESULTS(f"Perm-{p} {o[0]}",  o[1], *[str(i) for i in o[3:]]) ) # ESTIMATOR = 2
        return O_   

    
    def run(self, data_pipe, model_pipe_pair, X_data, y_data=None): 
        # print( data_pipe, "======", len(model_pipe_pair), "++++", model_pipe_pair )
        model_pipe, gs_paramz = model_pipe_pair 

        def update_gsearch_param_keys(mp, gp):
            O_ =  {}
            m = mp.steps[-1][0]
            for k, v in gp.items():
                O_[ f"model_pipe__{m}__{k}" ] = v                 
            return O_ , m

        gs_paramz , m_name = update_gsearch_param_keys(model_pipe, gs_paramz)
        
        # print( data_pipe )
        dz = "__".join([str(x[0]) for x in data_pipe.steps]) 
        m_name = f"{m_name} {dz}"

        piper = Pipeline([ ('data_pipe', data_pipe),
                            ('model_pipe', model_pipe)])
        # print(f"============\n{piper}\n{g_paramz}\n==============<<<<")

        gsearch = GridSearchCV(estimator=piper,
                                param_grid=gs_paramz,
                                cv=StratifiedKFold(n_splits=2), ## random_state=99, shuffle=True
                                n_jobs=1,
                                return_train_score=True) 
        gsearch.fit(X_data, y_data) 

        return (m_name, gsearch.best_score_, gsearch.best_estimator_ , gsearch.best_params_)

