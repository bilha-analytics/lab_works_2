'''
author: bg
goal: 
type: shared/utilz formodelz @@ TL, Common archs,  
how: 
ref: 
refactors: 
'''


import copy
import pickle 

from tqdm import tqdm 

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline 
from sklearn import metrics as skmetrics 
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import torch 
from torch import nn
from torch import optim 
import torch.nn.functional as TF 

from interfacez import ZTrainableModel 

from zreport import ZReporter 
import model_sharez 

class BasicCNNModel(nn.Module, ZTrainableModel):
    def __init__(self, n_in, n_out, bias=True, conv_modulez=None, 
                 out_activation=nn.Sigmoid, act_argz={'dim':1},
                skip=False, 
                scoring_func = skmetrics.accuracy_score ):
        super(BasicCNNModel, self).__init__()

        self.scoring_func = scoring_func 

        self.skip = skip 
        self.conv_blocks = nn.ModuleList( conv_modulez )

        ## calc 
        n_in = self.__get_channels_from_last_conv_pool_block(n_in)  
        nfc = n_in//2 
        print(n_in, nfc, n_out)
        self.fc1 = nn.Linear(n_in, nfc, bias=bias)
        self.fc2 = nn.Linear(nfc, n_out, bias=bias)
        self.out_activation = out_activation()#**act_argz) 
        
    def __get_channels_from_last_conv_pool_block(self, dim_):
        ## TODO: refactor 
        # print(self.conv_blocks) 
        # print( "++++++++++ ", dim_ )
        def get_n(mb, ndim): 
            if isinstance(mb, torch.nn.Conv2d): 
                h,w,c = ndim 
                k = mb.kernel_size[0]
                s = mb.stride[0]
                p = mb.padding[0] 
                c_out = mb.out_channels 
                # print(k, s, p, c_out, ndim)  
                h = 1 + (h + 2*p - k)/s  
                w = 1 + (h + 2*p - k)/s  
                n = (h, w, c_out )

            elif isinstance(mb, torch.nn.MaxPool2d): 
                h,w,c = ndim 
                k = mb.kernel_size
                s = mb.stride ## TODO: if stride != 1 
                # print('maxpool: ',k,s)
                # n = ndim // (k*k)
                n = (h//k, w//k, c) 
            else: # default = isinstance(mb, nn.Linear):
                n = ndim 
            return n

        n_ = dim_
        for mb in self.conv_blocks:
            n_ = get_n(mb, n_) 
            # print(n_, " === ", mb)

        n_ = int(n_[0] * n_[1] * n_[2] )  
        ## forcing grayscale diff of 132 on 84x84TODO: find where why how etc and fix it
        if dim_[2] == 1:
            n_ -= 132 
        return n_ 

    def forward(self, X_):       
        # print(len(X_), end="\t:: ")
        if isinstance(X_, (list, np.ndarray) ):
            for xi in X_:
                x_ = self._fwd_xi( torch.tensor(xi) )
        else:
            x_ = self._fwd_xi(X_)             
        return x_
    
    def _fwd_xi(self, x):
        o_ = x.float()  
        skipped = -1
        rez = o_ 
        ## 1. conv_blocks
        for m in self.conv_blocks:
            # print(o_.shape, "\t@ ", m)
            if (not isinstance(m, nn.Conv2d) ) and (skipped < 0): 
                ## 2. skip/residuals after convz befor pulling et all 
                if self.skip:## do b4 the first non-conv for now
                    # print( rez.shape, " SKIP-ADD ", o_.shape)
                    o_ = o_ + rez 
                    skipped = 10            
            o_ = m(o_)
            if skipped < 0:
                rez = o_ 
            
        ## 3. FC and activation 
        ## TODO: hack @ downsampling beef --- pad with zeros if don't match 
        # ex_n_in = self.fc1.in_features 
        # diff_n = max(0, (len(o_.flatten()) - ex_n_in) ) 
        # if diff_n > 0:
        #     o_ = TF.pad(input=o_, pad=(0,0,0,diff_n), mode='constant', value=0)
        #     o_ = o_[:-diff_n] 
        # else:
        #     o_ = o_[:ex_n_in] ## TODO: who's causing this mess!!!! arghhh 33x33 becomes 35x35 at some point and kills everything after that 
        # print(o_.shape, "\?t@ ", m)
        o_ = o_.flatten(1).reshape(1, -1) # o_.view(-1)
        o_ = self.fc1( o_ ) 
        o_ = self.fc2( o_ ) 
        o_ = self.out_activation(o_)
        return o_
    
    def score(self, yhat, y_): 
        ### AARRRRGGGGGHHHHHH!!!!!!!!!!
        print("=========SCORING 1========", type(y_), type(yhat), len(y_), len(yhat), y_[0].shape, yhat[0].shape )
        # return 0.5 
        yhat = np.array([y.detach().cpu().numpy().flatten().argmax() for y in yhat] ) 
        # print(yhat)
        y_ = np.array([y.detach().cpu().numpy().item() for y in y_]) 
        # print(y_) 
        # print("=========SCORING 2========", type(y_), type(yhat), y_.shape, yhat.shape )
        # return np.mean(  np.abs(y_ - yhat )/yhat.max() )# 
        return self.scoring_func( y_, yhat , normalize=True)

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
                save_checkpoints=None, 
                model_trainer_callback=None, ## pass back training results 
                posttrain_callback=None,
                use_cuda = False ): 
        self.model = model 
        self.loss_func = loss_func_argz_tuple[0]( **loss_func_argz_tuple[1] )
        self.optim_func = optim_func_argz_tuple[0]( model.parameters(), **optim_func_argz_tuple[1] )  

        self.save_checkpoints = save_checkpoints 
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

        if self.save_checkpoints: ## save checkpoints 
            with open(self.save_checkpoints, 'wb') as fd:
                pickle.dump(self.model, fd ) 

        if self.model_trainer_callback: ## pass back running loss to trainer 
            self.model_trainer_callback(running_loss, len(X_) )  
            
        return self 

    def transform(self, X_, y_=None): 
        O_ = self.predict(X_, y_)  
        return O_ 

    def score(self, yhat, y_):    
        return self.model.score(yhat, y_) 
        

    ## ==== 2. PyTorch train and eval/predict 
    def train(self, X_, y_=None): ##X_ is  a list of observations even iff single observation 
        ## operate on a single instance or a batch X_ = (n,c,h,w) or something like (n, x_i)
        running_loss = 0 
        y_ = [None for _ in range( len(X_) ) ] if y_ is None else y_  ## for case of unsupervised or something 
        # print( "****** X_ size = ",len(X_), type(X_) ) # X_[0].shape 
        for x, y in zip(X_, y_):  
            # 0. prep y
            # print('b4: ', y, type(y) )  
            # y = torch.tensor( [y,], ).reshape(1,1) # [y,] ) ##TODO: fix this for now force #y.reshape(1, *y.shape) 
            # y = y.reshape(1,1,*y.shape )
            # y = y.reshape(1,*y.shape ) if y is not None else x.float() 
            y = torch.tensor(y).unsqueeze(0) if y is not None else x.float() 
            # print('after: ',y.shape, y, y.item())
            # a. zero the grads
            self.optim_func.zero_grad() 
            # b. set model to training mode
            self.model.train()
            # c. forward pass, calc loss, backprop, 
            x = self.model( x ) 
            # print('@LOSS: x,y', x.shape, y.shape, type(x.flatten()[0].item() ), type(y.flatten()[0].item() ) )
            # print('AS VIEWS: ', x.shape, y.view(-1).reshape(1, -1).shape )
            # loss = self.loss_func(x, y.view(-1).reshape(1,-1) ) #.squeeze(1) )
            loss = self.loss_func(x.float(), y.reshape(-1)  ) #.squeeze(1) )
            # print("@LOSS: ", type(loss) )
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
            # print( O_.shape )
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
        append_permute = lambda d_, m_: self.permutationz.append( ( Pipeline(d_), (Pipeline(m_[0]), m_[1] ) ) )    
        d_, m_ = np.meshgrid( range( len(data_pipez)), range(len(model_pipez)) )
        for x, y in zip(d_, m_):
            for i, j in zip(x, y):
                if model_pipez[j][0][-1][0] == 'cnn': ## == (('cnn', cnn_kwargz), kwargz)
                    D_ = data_pipez[i][0] 
                    c = data_pipez[i][1]
                    ckwargz = copy.deepcopy( model_pipez[j][0][-1][1]  ) 
                    ckwargz['n_in'] = (*ckwargz['n_in'], c) 
                    # print(i, ":>>>>> ", c , ckwargz) 
                    m_ = ModelNetwork( 
                            BasicCNNModel( **ckwargz, 
                                conv_modulez = model_sharez.CNNBlocksBuilder.conv_block( 
                                    c, ckwargz['n_out'], kernel=min(c,3) , is_modulelist=False ) ),
                            model_trainer_callback=self.training_callback ) 
                    # print("+++++++ ", m_) 
                    # f"cnn_{i+1}" 
                    M_ =  (model_pipez[j][0][:-1] + [ ('cnn', m_)],  model_pipez[j][1] )
                    # print(M_)
                    append_permute( D_, M_ ) 
                else:
                    append_permute( data_pipez[i][0], model_pipez[j] ) 

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
        # for p, (data_p, model_p) in tqdm( enumerate(self.permutationz, 1) ):
        p = 1
        for data_p, model_p in tqdm( self.permutationz ):
            # print("@@@@: ", p, data_p, " &&&---ON---&&&", model_p[0], " <<< ", model_p[1], "\n") 
            self.running_loss = 0             
            self.current_epoch = 0 
            for e in range(self.epochz) :     ##TODO: recheck logic, saving per batch Vs epoch Vs per permute  
                self.current_epoch = e+1 
                o = self.run( data_p, model_p, X_data, y_data) 
            O_.append( o ) 
            ZReporter.add_log_entry( MSG_GSEARCH_RESULTS(f"Perm-{p} {o[0]}",  o[1], *[str(i) for i in o[3:]]) ) # ESTIMATOR = 2
            p += 1
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
        # print(f"============\n{piper}\n{gs_paramz}\n==============<<<<")

        gsearch = GridSearchCV(estimator=piper,
                                param_grid=gs_paramz,
                                cv=StratifiedKFold(n_splits=2), ## random_state=99, shuffle=True
                                n_jobs=1,
                                return_train_score=True) 
        gsearch.fit(X_data, y_data) 

        return (m_name, gsearch.best_score_, gsearch.best_estimator_ , gsearch.best_params_)

