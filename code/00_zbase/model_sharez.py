'''
author: bg
goal: 
type: shared/utilz for modelz @@ TL, Common archs,  
how: 
ref: 
refactors: 
'''

from sys import builtin_module_names
from tqdm import tqdm 

import numpy as np

from torch import nn
from torchvision import models 


class TLModelBuilder:
    '''

    @inputs:    model type (vgg, inception, mobilenet, etc), model paramz, classifier block, trainable
    @outputs:   TL block/model with pretraining, classifier_blocks etc setup 
    @actions:   > create TL model --> model type, remove classifier block OR set to new definition, 
                > 2 level training strategy: I.E: 1. TL original model + 2. additional training on target image/dataset kind 
    @TODO:      > scikit Vs pytorch Vs keras/tensorflow wrapper  
                > pretraining strategy manager 
    ''' 
    ### ===== 1. models 
    supported_models = {
        'resnet'     : models.resnet18,
        'vgg16'      : models.vgg16_bn,
        'inception'  : models.inception_v3,
        'densenet'   : models.densenet161,
        'squeezenet' : models.squeezenet1_0,
        'mobilenet'  : models.mobilenet_v2

    }

    _model_classifier_blocks = {
        'resnet'     : ('fc', 'avgpool'),
        'vgg16'      : ('classifier', 'avgpool'), #classifier = linear, ReLU, dropout, linear, ReLU, dropout, Linear
        'inception'  : ('fc', 'dropout', 'avgpool'),
        'densenet'   : ('classifier',),
        'squeezenet' : ('classifier',), #classifier = dropout, conv2d1x1, ReLU, pool
        'mobilenet'  : ('classifier',), #classifier = dropout, linear
    }

    @staticmethod 
    def get_TL_model(model_name, pretrained=True, 
                        freeze_weights=True, 
                        classifier_update=None):
        model = TLModelBuilder.supported_models.get(model_name, None) 
        if model is None:
            raise Exception(f'Model {model_name} is not yet in the supported list. Try: {list(TLModelBuilder.supported_models.keys())}')  

        ## 1. instantiate with pretrained = 
        model = model(pretrained=pretrained)

        ## 2. freeze weights 
        if freeze_weights:
            for param in model.parameters():
                param.requires_grad = False  

        ## 3. update classifier layer as desired 
        ## TODO: refactor classifier_update values @ uniformity 
        # Leave as is=-999, Delete it=None, Set it to something new = object
        if classifier_update is None:
            # delete
            TLModelBuilder._delete_classifier_block(model_name, model) 

        elif isinstance(classifier_update, int) and (classifier_update < 0): # -999
            # leave as is
            pass 
        else: # anything else should be nn.Sequential type of sort  OR n_out_channels int 
            # set it to the new object 
            TLModelBuilder._update_classifier_block(model_name, model, classifier_update) 

        return model 

    @staticmethod 
    def _delete_classifier_block(model_name, model):
        mb_name = TLModelBuilder._model_classifier_blocks.get(model_name, None)
        if mb_name is None:
            raise Exception(f"Unsupported model {model_name}")
        delattr(model, mb_name[0] ) 
        # return model ##<-- in_place operation on model  


    @staticmethod
    def _update_classifier_block(model_name, model, classifier_update): 
        
        # 1. trace classifier blocks in kind model_name  << TODO: how to deal pool, drop, relu blocks @moment defined in classifier_update 
        def set_classifier_block(m_name, model, mb):
            mb_name = TLModelBuilder._model_classifier_blocks.get(m_name, None)
            if mb_name is None:
                raise Exception(f"Unsupported model {m_name}")
            setattr(model, mb_name[0], mb)
            #return model <-- in_place operation on model 

        # 2. update with new block 
        if isinstance(classifier_update, int): # and (classifier_update > 0):
            # use default fc update
            n_out = classifier_update 
            mb = getattr(model, TLModelBuilder._model_classifier_blocks[model_name][0] ) 
            if model_name in ['inception']: ## TODO: inception is different
                n_in = TLModelBuilder.get_n_in_features(mb) 
            elif model_name in ['vgg16',]:
                n_in = n_in = TLModelBuilder.get_n_in_features(mb[0])  
            elif model_name in ['squeezenet', 'mobilenet']: ## TODO: conv2d.in_features or not 
                n_in = n_in = TLModelBuilder.get_n_in_features(mb[1]) 
            else: #default is fc = Linear E.G. model_name in ['resnet', 'densenet',]: 
                n_in = n_in = TLModelBuilder.get_n_in_features(mb)            
            
            classifier_update = TLModelBuilder.get_fc_classifier_block(n_out, n_in) 

        set_classifier_block(model_name, model, classifier_update) 

        # return model 

    @staticmethod 
    def get_n_in_features(mb):
        if isinstance(mb, nn.Conv2d):
            n = mb.in_channels
        else: # default = isinstance(mb, nn.Linear):
            n = mb.in_features 
        return n 
         

    ### ===== 2. Menu of classifier blocks << TODO: 
    def get_fc_classifier_block(out_features, in_features, bias=True):
        m_ = nn.Linear(in_features = in_features, out_features= out_features, bias=bias)
        return m_ 



    ### ===== 3. Pretraining Strategy et al TODO: Syncy up ZModel + ModelManager + ModelTrainer

    # def __init__(self, encoder_decoder_model=None ): 
    #     self.autoencoder = encoder_decoder_model
    
    # ## TODO: move train/predict code to some shared manager 
    # def train(self, X_data, epochz=3,
    #                         loss_func = nn.MSELoss(),
    #                         optimizer = optim.SGD, 
    #                         optimizer_paramz = {'lr':1e-3, 'momentum':.9} 
    #                         ):
    #     # 1. go into training mode + setup optimizer
    #     self.autoencoder.train() 
    #     optimizer = optimizer( self.autoencoder.parameters(), **optimizer_paramz )  

    #     # 2. run epochs and per item in batch 
    #     for e in tqdm( range(epochz ) ):
    #         running_loss = 0
    #         n_inst = 0
    #         for x in X_data:
    #             x_ = x#.float() 
    #             # a. zero the grads, compute loss, backprop 
    #             optimizer.zero_grad()
    #             o_ = self.autoencoder( x_ ) 
    #             loss = loss_func(o_, x_ ) 
    #             loss.backward()
    #             optimizer.step() 
    #             # b. update aggregates and reporting 
    #             running_loss += loss.item() 
            
    #         print(f"E {e}: loss {running_loss}") 

    # def predict(self, x): 
    #     x_ = x#.float() 
    #     # 1. go into eval mode
    #     self.autoencoder.eval() 
    #     # 2. run through model 
    #     o_ = self.autoencoder( x_ )
    #     # 3. extract top-k 
    #     O_ = []
    #     for rec in o_:
    #         O_.append( rec.detach()  )
        
    #     return torch.stack( O_ )  ## why yield??? 

    # def eval(self, ):
    #     pass 

        
