'''
author: bg
goal: 
type: shared/utilz and building blocks for modelz @@ TL, Common archs,  
how: 
ref: 
refactors: 
'''

from sys import builtin_module_names
from numpy.lib.shape_base import dstack
from tqdm import tqdm 

import numpy as np
import torch 
from torch import nn
from torch.nn import functional as F
from torchvision import models 


### === 1. Transfer Learning blocks - encoders, backbones ===
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
    def get_TL_model(model_name, pretrained=True, in_channelz=3, 
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

        ## 3. update classifier layer as desired  Default = chuck it
        ## TODO: refactor classifier_update values @ uniformity 
        # @Values == Leave as is=-999, Delete it=None, Set it to something new = object
        if classifier_update is None:
            # delete
            TLModelBuilder._delete_classifier_block(model_name, model) 

        elif isinstance(classifier_update, int) and (classifier_update < 0): # -999
            # leave as is
            pass 
        else: # anything else should be nn.Sequential type of sort  OR n_out_channels int 
            # set it to the new object 
            TLModelBuilder._update_classifier_block(model_name, model, classifier_update) 


        ## 4. update first conv layer to deal different input channels 
        if in_channelz != 3:
            _ = TLModelBuilder._update_conv1(model_name, model, in_channelz) 

        return model 

    @staticmethod 
    def _delete_classifier_block(model_name, model):
        mb_name = TLModelBuilder._model_classifier_blocks.get(model_name, None)
        if mb_name is None:
            raise Exception(f"Unsupported model {model_name}")
        for mb in mb_name[:1]: ## TODO: nesting errors @ name + what layers ok to del 
            delattr(model, mb ) 
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
         

    @staticmethod
    def _update_conv1(model_name, model, in_channelz):
        ## weights_for_reinit
        mb = TLModelBuilder.__get_first_conv_layer(model_name, model)
        n_out = mb.out_channels 
        k = mb.kernel_size
        s = mb.stride
        p = mb.padding
        b = mb.bias
        if not isinstance(b, bool):
            # print(type(b))
            b = False #bool(b[0].item() )
        # print(in_channelz, n_out, k, s, p, b)
        mb2 = nn.Conv2d(in_channelz, n_out, kernel_size=k, stride=s, padding=p, bias=b) 
        # print("<<<:: ", mb2)
                        
        TLModelBuilder.__set_conv1(model_name, model, mb2) 
        return TLModelBuilder.__get_first_conv_layer(model_name, model) 

    @staticmethod 
    def __get_first_conv_layer(model_name, model): 
        mb = None 
        if model_name in ['vgg16','squeezenet', ]:
            mb = getattr(model, 'features', None)[0]
            
        elif model_name in ['inception']: ## TODO: inception is different
            mb = getattr(model, 'Conv2d_1a_3x3', None).conv
            
        elif model_name in ['densenet',]: 
            mb = getattr(model, 'features', None).conv0        
            
        elif model_name in ['mobilenet']: 
            mb = getattr(model, 'features', None)[0][0]
            
        else: #default is resnet
            mb = getattr(model, 'conv1', None)
        return mb 


    @staticmethod
    def __set_conv1(model_name, model, conv1, require_grad=True): ##TODO: require grad implications 
        def update_weights(mb, conv1):
            # print("\n%%%%STARTING:::", mb.state_dict().keys(), "\n" ) #dir(conv1))
            zweights = mb.state_dict().get('weight', None)
            zbias = mb.state_dict().get('bias', None)
            
            ## 1. weights
            if zweights is None: ## try densenet setup 
                zweights = mb.state_dict().get('conv0.weight', None)
                
            if zweights is not None:
                in_c = conv1.in_channels 
                m_w = zweights[:,0:3,:,:]
                # print( type(m_w), m_w.shape )
                m_w = m_w.sum(axis=1)
                print( "AFTER SUM m_w: ",type(m_w), m_w.shape, "for conv1.in_channels = ", in_c) 
                m_w = m_w.unsqueeze(1) 
                m_w = torch.tensor(np.hstack([ m_w for _ in range(in_c) ] )) 
                print( "New Weights: ", m_w.shape ) 
                conv1.weight = nn.Parameter(m_w, requires_grad=require_grad)  
            
            ## 2. bias 
            if zbias is not None:
                conv1.bias=nn.Parameter(zbias, requires_grad=require_grad) 
            
            print( f"DONE WEIGHTS_UPDATE: weights = {zweights is not None}, bias = {zbias is not None }")
            return conv1
        
        if model_name in ['vgg16', 'squeezenet',]:
            conv1 =update_weights( getattr(model, 'features',None)[0] , conv1)        
            getattr(model, 'features',None)[0] = conv1 
            
        elif model_name in ['inception']: 
            conv1 =update_weights( getattr(model, 'Conv2d_1a_3x3', None).conv , conv1)        
            getattr(model, 'Conv2d_1a_3x3', None).conv  = conv1
            
        elif model_name in ['densenet',]: ## Densenet is different 
            conv1 =update_weights( getattr(model, 'features',None), conv1)        
            setattr( getattr(model, 'features',None), 'conv0', conv1)
            
        elif model_name in ['mobilenet']: 
            conv1 =update_weights( getattr(model, 'features', None)[0][0] , conv1)        
            getattr(model, 'features', None)[0][0]  =  conv1 
            
        else: #default is resnet lilke 
            # mb = nameof( getattr(model, 'conv1', None))
            conv1 =update_weights( getattr(model, 'conv1', None), conv1)        
            setattr(model, 'conv1', conv1)    
        

    ### ===== 2. Menu of classifier blocks << TODO: 
    def get_fc_classifier_block(out_features, in_features, bias=True):
        m_ = nn.Linear(in_features = in_features, out_features= out_features, bias=bias)
        return m_ 



    ### ===== 3. Pretraining Strategy et al TODO: Syncy up ZModel + ModelManager + ModelTrainer

        
### === 2. CNN blocks menu - conv (basic, residual, non-activated, ), deconv(opposite per conv), 1x1 bottleneck, 
class CNNBlocksBuilder:
    '''
    @inputs:    menu option, paramz --> only at grad aware components, the rest leave to calling model 
    @outputs:   nn.ModuleList or list of nn.Modules TODO: traversal and registration implications
    @actions:   assemble ready to use common types 
    ''' 
    @staticmethod
    def conv_block(in_channelz, out_channelz, bias=True, 
                    kernel=3, stride=1, padding=1,  padding_mode='zeros', #kernel can be list. Padding is TODO
                    n_conv_layerz=2, 
                    batch_norm=True, batch_norm_affine=True, 
                    activation=nn.ReLU, activation_argz = {'inplace':True}, 
                    maxpool=True, maxpool_argz={'kernel_size':2, 'stride':2}, 
                    dropout=True, dropout_argz={'p':.4, 'inplace':True}, ## inplace Vs grad includes 
                    is_modulelist=True ):
        # kernel = (kernel, kernel) if isinstance(kernel, int) else kernel 

        modz = []
        ## 1. the conv layerz 
        n_in = in_channelz
        #n_out = out_channelz ## TODO: calc backwards @ kern, stride, padding et all <--- n_out = 1 + (n_in +2*p - k)/s ; p padding, k kernel , s stride sizes
        n_out = max( 1, ((out_channelz - 1)*stride) + kernel - (2*padding) ) ## TODO by n_conv_layerz @ > 2
        for i in range(n_conv_layerz):
            # print(n_in, n_out, end="\t")
            modz.append( nn.Conv2d(n_in, n_out, kernel_size=kernel, stride=stride, padding=padding) ) ## TODO: n_channelz deal, padding options
            n_in = n_out 
            n_out = 1 + ((n_in +(2*padding) - kernel)//stride) ## TODO: calc fwd 
            # print("-->", n_in, n_out) 
        n_out = modz[-1].out_channels 
        ## 2. batchnorm and activation TODO: affine=True/False @ learnable params <--- batchnorm and gradientz calc; should that behere and how to block grad iff
        if batch_norm:
            print("in ", in_channelz, "out ", out_channelz, 'use: ', n_out)
            # if in_channelz == 1: # use 1d
            #     pass ##modz.append(nn.BatchNorm1d(n_out, affine=batch_norm_affine) ) 
            # else:
            modz.append(nn.BatchNorm2d(n_out, affine=batch_norm_affine) )
            # print( modz )  

        if activation:
            modz.append( activation( **activation_argz) ) 

        if maxpool:
            modz.append( nn.MaxPool2d(**maxpool_argz ) )  ## TODO: @ grad includes
        
        if dropout:
            modz.append( nn.Dropout2d(**dropout_argz) ) ## TODO: @grads includes + order norm,pool,dropout??

        ## track residuals, etc @ caller b/c not grad 

        ## 3. return modulelist or list of assembled components
        if is_modulelist:
            return nn.ModuleList( modz )
        else:
            return modz 
        


### === 3. Algorithms/Computations e.g. loss funcs, fusion, residuals add, 
class AlgorithmzFunc:
    '''
    Collection of reusable computations <-- 
    @inputs:    e.g. from forward call, loss/criterion wrapper or an optim wrapper
    @outpus:    computed results on single or batches    
    @actions:   compute as per func 
    @TODO: strategy design approach Vs builder + chainable etc  THEN combine with pytorch or sklearn wrapper  or Model_trainerz approach 
    '''

    @staticmethod
    def add_residuals(x, r, r_ratio=1, method='sum'): 
        ## deal with different ways of incorporating residuals 
        return x + (r*r_ratio) 