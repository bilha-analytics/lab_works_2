'''
author: bg
goal: 
type: util/factory for model sub-components 
how: 
ref: 
refactors: 
'''

import numpy as np
import torch 
from skimage import transform as sktransform 

import keras
from keras.models import Model as KerasBaseModel 
from keras.applications import vgg16 # import preprocess_input, decode_predictions
from keras.applications import inception_v3 
from keras.applications import resnet50  
from keras.applications import mobilenet_v2  
from keras.applications import densenet
from keras.applications import nasnet 


class LossAlgorithms:
    pass

class OptimizationAlgorithms:
    pass

class TransferLearningBlocks: 
    ## TODO: Keras Vs PyTorch 
    ### ==== 1. Keras TL 
    @staticmethod
    def get_keras_pretrained_encoder(model_name, n_inputs, n_classes, weights_initor='imagenet'):
        #TODO: move hash/db
        _API_DB = { ##(model, app_preproc, img_tsize)
            'vgg16' : (vgg16.VGG16, vgg16, (224,224) ),
            'resnet50' : (resnet50.ResNet50, resnet50, (224,224) ),
            'inception_v3' : (inception_v3.InceptionV3, inception_v3, (299,299) ),
            'mobilenet_v2' : (mobilenet_v2.MobileNetV2, mobilenet_v2, (224,224) ),
            'densenet' : (densenet.DenseNet121, densenet, (224,224) ),
            'nasnet-mobile' : (nasnet.NASNetMobile, nasnet, (224,224) ),
            'nasnet-large' : (nasnet.NASNetLarge, nasnet, (331,331) ),
        }

        def get_encoder(model, preproc, tsize):
            ## 1. create model without the top/classifier and with set n_inputs and n_classes/outputs
            block = model(weights=weights_initor,
                        include_top=False, 
                        input_shape=n_inputs,
                        classes=n_classes, 
                        pooling='avg')

            ## 2. freeze weights 
            block = KerasBaseModel(inputs=block.inputs,
                                    outputs=block.output)
            for l in block.layers:
                l.trainable = False 

            return block 


        n_ = None
        rec = _API_DB.get(model_name, None) 
        if rec is not None:
            model, preproc, tsize = rec 
            n_ = get_encoder( model, preproc, tsize) 

        return n_

    @staticmethod
    def keras_encode(Xi_data, encoder, preproc, tsize=(224,224), topn=1):
        ## reshape
        x = sktransform.resize(Xi_data, anti_aliasing=True)
        x = Xi_data.reshape( (1. *x)) ## TODO: check shape + np Vs torch 
        x = preproc.process_input(x) 
        xhat = encoder.predict(x) 
        # xlbl = preproc.decode_predictions(xhat, top=topn)[0]
        return xhat 


class NNArchitecureBlocks:
    pass 