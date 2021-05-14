'''
author: bg
goal: shared utilities for transformables 
type: util 
how: 
ref: 
refactors: 
'''
from tqdm import tqdm 

import numpy as np
from skimage.filters import unsharp_mask, gaussian 

import torch
from torch import optim 
from torch import nn
import torch.nn.functional as nn_F
from torchvision import models 

from model_sharez import TLModelBuilder 

''' TODO: replacement @ TL Vs EncoderDecoderz

    @inputs:    model type (vgg, inception, mobilenet, etc), model paramz, classifier block, trainable
    @outputs:   pretrained encoder/embedding model without classifier block 
    @actions:   > create TL model --> model type, remove classifier block OR set to new definition, 
                > 2 level training strategy: I.E: 1. TL original model + 2. additional training on target image/dataset kind 
    @TODO: scikit Vs pytorch Vs keras/tensorflow wrapper  
    ''' 

class Encoder(nn.Module):
    '''
    Encoder block/module 
        Works with: vgg, 
    @inputs:    n_channels_in, n_channels_out, model type , classifier_update
    @outputs:    
    @actions:   setup encoder block with given TL model type + forward pass flow 
    '''
    def __init__(self, model_name='vgg16', 
                    in_channelz=3, out_channelz=512, ## TODO: use these or delete 
                    pretrained=True, freeze_weights=True, classifier_update=None): 
        super(Encoder, self).__init__() 
        
        self.in_channelz = in_channelz ## TODO: img/fmap channel options update in TLModel
        self.out_channelz  = out_channelz 

        ## TODO: generalize better  
        # 2. https://medium.com/intelligentmachines/implementation-of-class-activation-map-cam-with-pytorch-c32f7e414923 
        #  

        ## setup model as pretrained and chuck/update classifier --> classfier_update=None-->delete it
        model = TLModelBuilder.get_TL_model(model_name, 
                                            freeze_weights=freeze_weights, 
                                            pretrained=pretrained, 
                                            classifier_update=classifier_update)

        # print("GEnerated TL Model @ VGG16_bn", model.__dict__ )
        self.encoder_model = self._encodify(model, classifier_update) 

    ## TODO: enforce at all types b/c of AE calls 
    @property    
    def get_encoder_block(self):
        return self.encoder_model 

    ## TODO: encodify non-VGG models ; only VGG at the moment 
    ## TODO: why, when to add back classifier blocks and related  
    def _encodify(self, model, classifier_update):
        ## freeze weights and get the pooling indices from max_pool layers for use by decoder unppoling <-- reinitialize these indices b/c TL doesn't regenerate
        modulez = nn.ModuleList()
        for mod in model.features: 
            if isinstance(mod, nn.MaxPool2d):
                mod_add = nn.MaxPool2d(
                    kernel_size = mod.kernel_size,
                    stride = mod.stride,
                    padding = mod.padding,
                    return_indices = True 
                )
                modulez.append( mod_add )
            else:
                modulez.append( mod ) 

        if classifier_update is not None:
            modulez.append( model.avgpool )
            modulez.append( model.classifier )

        return modulez 


    def forward(self, x):
        pool_indices = [] ## to be passed to decoder for unpooling 
        x_i = x 
        for mod in self.encoder_model:
            o_ = mod(x_i) 
            if isinstance(o_, tuple) and len(o_) == 2:
                x_i, idx = o_ 
                pool_indices.append( idx )
            else:
                x_i = o_ 
        return x_i, pool_indices 


class EncoderFused(Encoder): 
    '''
    @TODO: puporse and inclusion approach 
    '''
    def __init__(self, model_name, merger_type='mean', 
                in_channelz=3, out_channelz=512, 
                freeze_weights=True, pretrained=True, classifier_update=None):
        super(EncoderFused, self).__init__(model_name, in_channelz=in_channelz, out_channelz=out_channelz, 
                                            freeze_weights=freeze_weights, pretrained=pretrained, classifier_update=classifier_update)

        if merger_type is None:
            self.code_post_process = lambda x: x 
            self.code_post_process_kwargz = {}
        elif merger_type == 'mean':
            self.code_post_process = torch.mean 
            self.code_post_process_kwargz = {'dim':(-2, -1)}
        if merger_type == 'flatten':
            self.code_post_process = torch.flatten 
            self.code_post_process_kwargz = {'start_dim':1, 'end_dim':-1}
        else:
            raise ValueError("Unknown merger type for the encoder {}".format(merger_type) )

    def forward(self, x):
        x_current, _ = super().forward(x) 
        x_code = self.code_post_process(x_current, **self.code_post_process_kwargz) 
        return x_code 
    

class AutoEncoder(nn.Module):

    class Decoder(nn.Module):
        '''
        Decoder module/block; sorta transfposed version of encoder --> in reverse flow 
            Works in syncy with encoder --> if encoder can't then this can't
        @inputs:    encoder module to invert <--- in_channelz, out_channelz, classifier_update, model type 
        @outputs:
        @actions:      
        '''
        def __init__(self, encoder_block):
            super(AutoEncoder.Decoder, self).__init__()
            self.decoder = self._decodify(encoder_block)                

        def _decodify(self, encoder):
            ## invert encoder == mirror image
            ## operated @ 2d transfpose convolution and 2d unpooling 
            ## 1. 2D transpose convolution + batch norm + activation 
            ##    convert encoder.conv to decoder.transposed conv
            ## 2. 2d unpool : conver encoder.pool to decoder.unpool
            modulez = []
            for mod in reversed(encoder):
                if isinstance(mod, nn.Conv2d): # change conv2d to transposeConv2d with batch normalization and relu activation 
                    kwargz = {'in_channels':mod.out_channels,
                            'out_channels':mod.in_channels,
                            'kernel_size':mod.kernel_size, 
                            'stride': mod.stride,
                            'padding':mod.padding } 
                    mod_trans = nn.ConvTranspose2d( **kwargz ) 
                    mod_norm = nn.BatchNorm2d( mod.in_channels ) 
                    mod_act = nn.ReLU(inplace=True) 
                    modulez += [mod_trans, mod_norm, mod_act ] 

                elif isinstance(mod, nn.MaxPool2d): ## change to unpooling 
                    kwargz = {'kernel_size': mod.kernel_size,
                            'stride':mod.stride,
                            'padding':mod.padding}
                    modulez.append( nn.MaxUnpool2d(**kwargz)  )
            ## drop last norm and activation so that final output is from conv with bias
            modulez = modulez[:-2]

            return nn.ModuleList( modulez ) 

        def forward(self, x, pool_indices):
            ## x is a tensor from encoder and pool_indices is the list from encoder
            x_i = x
            k_pool = 0
            rev_pool_indices = list( reversed( pool_indices ) ) 
            for mod in self.decoder:
                if isinstance(mod, nn.MaxUnpool2d): #apply indices 
                    x_i = mod(x_i, indices=rev_pool_indices[k_pool])
                    k_pool += 1
                else:
                    x_i = mod(x_i) 
            return x_i 

    '''
    @inputs:    model type
    @outputs:   
    @actions:   combine encoder and decoder into one network, forward pass, return combined model params 
    ''' 
    def __init__(self, encoder_model): #in_channelz=3, out_channelz=512, pretrained=True, classifier_update=None 
        super(AutoEncoder, self).__init__() 
        self.encoder = encoder_model   #Encoder(model, in_channelz=in_channelz, out_channelz=out_channelz, pretrained=pretrained, classifier_update=classifier_update)
        self.decoder = AutoEncoder.Decoder( encoder_model.get_encoder_block ) 

    def forward(self, x):
        x_i, idx = self.encoder(x) 
        x_i = self.decoder(x_i, idx ) 
        return x_i

    def parameters(self):
        return list(self.encoder.parameters() ) + list(self.decoder.parameters() )

