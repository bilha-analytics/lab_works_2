'''
author: bg
goal: shared utilities for transformables 
type: util 
how: 
ref: 
refactors: 
'''
from sys import builtin_module_names
from tqdm import tqdm 
import pickle 

import numpy as np

from skimage import color as sk_color
from skimage import img_as_float, img_as_ubyte 
from skimage.filters import unsharp_mask, gaussian 

import torch
from torch import optim 
from torch import nn
import torch.nn.functional as nn_F
from torchvision import models 

import encoderz
import utilz 

class ChainableMixin:
    '''
    @inputs:    list of images/fmaps to operate on
    @outputs:   list of processed fmaps with or without append to original input 
    @actions:   append or not append 
    ''' 
    append_to_current_fmap = True

    def chain_transforms(self, incoming, output):
        O_ = []
    
        if not self.append_to_current_fmap:
            return output

        def unpack_for_dstack(fmap):
            F_ = [] 
            if len(fmap.shape) == 2:
                F_.append( fmap ) 
            elif len(fmap.shape) == 3:
                F_ = [ fmap[:,:, i] for i in range(fmap.shape[2] ) ] 
            # else ## is GIGO for NOOP
            return F_

        ## 1. unpack incoming 
        O_ += unpack_for_dstack( incoming )

        ## 2. unpack output 
        O_ += unpack_for_dstack( output )

        return np.dstack(O_) 



class NormalizerzThresholderzAlgz:
    '''
    Primarily statistical methods/calculations --> at pixel or hood level Vs global??
    @inputs:    a single image/fmap 
    @outputs:   a transformed single image/fmap 
    @actions:   appy given algorithm --> pixel or hood level Vs global ?? 
    @TODO:      make chainable; decorator pattern 
    ''' 
    
    @staticmethod 
    def _clear_outter_circle(img, b_thresh=0.9): 
        '''

        ''' 
        ## set outter circle to zero b/c capturing glare as well 
        o = img.copy() 
        x, y = o.shape[0], o.shape[1]
        cx, cy = (x//2), (y//2)
        bx, by = int(x*b_thresh), int(y*b_thresh) 

        px_len = lambda p: (cx - p[0])**2 + (cy - p[1])**2   

        max_bh = px_len( (bx, by) )
        for i in range(x):
            for j in range(y):
                h = px_len( (i, j) ) 
                if h >= max_bh: 
                    o[i, j] = 0 
        return o 

    @staticmethod
    def _thresholded_center_range(o, om, orange, thresh, update_val, drxn_less, clear_ring):
        '''
        @inputs:
            clear_ring: None if NO-OP else a % value, indicating % of outter ring of image to clear
        ''' 
        ## editing o in place 
        if drxn_less:
            o[ ( (o - om)/orange ) < thresh ] = update_val 
        else:
            o[ ( (o - om)/orange ) > thresh ] = update_val 

        if clear_ring:
            o = NormalizerzThresholderzAlgz._clear_outter_circle(o, b_thresh=clear_ring)  
        return o 
        
    @staticmethod 
    def thresh_range(x, thresh, update_val=0, do_eq=True, 
                    drxn_less = True, clear_ring=None):
        o = NormalizerzThresholderzAlgz._get_channel_eq(x ) if do_eq else x.copy() 
        om = o.min()
        orange = o.max() - o.min() 
        o = NormalizerzThresholderzAlgz._thresholded_center_range(o, om, orange, thresh, 
                        update_val, drxn_less, clear_ring )
        return o 

    @staticmethod
    def thresh_norm_std(x, thresh, update_val=0, std_factor=1,  do_eq=True, 
                        drxn_less = True, clear_ring=None):        
        o = NormalizerzThresholderzAlgz._get_channel_eq(x ) if do_eq else x.copy()
        om = o.mean()
        orange = np.std( o )*std_factor 
        o = NormalizerzThresholderzAlgz._thresholded_center_range(o, om, orange, thresh, 
                        update_val, drxn_less, clear_ring )
        return o 

    @staticmethod 
    def thresh_double_red2(x, thresh, update_val=0,  do_eq=True, 
                            drxn_less = True, clear_ring=None):         
        o1 = NormalizerzThresholderzAlgz.thresh_range(x, thresh, update_val, do_eq=do_eq, drxn_less=drxn_less, clear_ring=clear_ring)
        o2 = NormalizerzThresholderzAlgz.thresh_norm_std(x, thresh, update_val, do_eq=do_eq, drxn_less=drxn_less, clear_ring=clear_ring)
        o = o2 + (o1*(1+o2) ) 
        return o 

    @staticmethod 
    def thresh_darks_blend_but(x, thresh, update_val=0,  do_eq=True, 
                                clear_ring=None): 
        ## 1. bloody regions         
        o = NormalizerzThresholderzAlgz.thresh_range(x, 1-2.6*thresh, update_val=0.1, do_eq=do_eq)
        o = o * x ## TODO: select single channel green like  from x 
        o = NormalizerzThresholderzAlgz.thresh_range(x, 2.8*thresh, drxn_less=False, do_eq=do_eq  ) 
        o = o * x ## TODO: select single channel green like  from x 
        o = unsharp_mask(o, radius=5, amount=2) 

        o1 = o*(o - x) + o 

        ## 2. blur 
        o2 = x - gaussian(x, 2) 
        o2 = np.log( o2 / o2.sum() ) 

        if clear_ring:
            o1 = NormalizerzThresholderzAlgz._clear_outter_circle(o, b_thresh=clear_ring) 

        return o1 #o1, o2  

### ==== TODO: refactor 
#o = img_as_ubyte(self._get_channel_eq( img[:,:,-1] ) ) 

    @staticmethod
    def thresh_yellow(x, do_eq=True, clear_ring=None, threshit=False, thresh=0.97): 
        o = NormalizerzThresholderzAlgz._get_channel_eq(x ) if do_eq else x.copy() 
        ##yellow is in A and is -ves
        # o = o[:,:,1] ## input is channel selected already << TODO: order of this viz a viz eq
        o[o >= 0 ] = 0
        o = -1 * o  

        if threshit: ## TODO: Order of this vis a viz clear ring 
            rrange = o.max() - o.min() 
            o[ ((o - o.min())/rrange) <  thresh ] = 0 

        if clear_ring:
            o = NormalizerzThresholderzAlgz._clear_outter_circle(o, b_thresh=clear_ring) 

        return o 

    @staticmethod
    def thresh_blue(x, thresh=1, do_eq=True):         
        o = NormalizerzThresholderzAlgz._get_channel_eq(x ) if do_eq else x.copy()
        o = img_as_ubyte(o)  
        o[ o != thresh] = 0 
        o[ o == thresh] = 255   
        return img_as_float(o) #.astype('uint8') 


    @staticmethod
    def _get_channel_eq(img, c=-1, eq_mtype=1): ## -1 is on gray scale 
        return utilz.Image.hist_eq( utilz.Image.get_channel(img, c), mtype=eq_mtype ) if c >= 0 \
            else utilz.Image.hist_eq( utilz.Image.gray_scale(img), mtype=eq_mtype  )
   
    @staticmethod
    def _get_lab_img(img, extractive=True): 
        o = sk_color.rgb2lab( img ) 
        if extractive:
            l = o[:,:,0]
            ## bluez are -ves, redz are positivez
            b = o[:,:,-1]
            b[ b >=0 ] = 0
            ## QUE: add back yellow to red or not ?? << does it seem to be useful << TODO: review
            r = o[:,:,1]
            # y = r.copy()
            # y[y>=0] = 0
            # process red
            r[r <= 0 ] = 0
            # add back yellow
            # r = r*y 
            o = np.dstack([l,r,b])  
        # normalize-ish :/ <<< TODO: fix 
        # abs_max = np.max( np.abs( o ) ) 
        # o = o/abs_max
        return o 

    ## TODO: Thresholding
    @staticmethod
    def _get_yellow_from_rgb2lab(img, threshit=False, thresh=0.97): 
        o = sk_color.rgb2lab( img )  
        ##yellow is in A and is -ves
        o = o[:,:,1]
        o[o >= 0 ] = 0
        o = -1 * o  
        if threshit:
            rrange = o.max() - o.min() 
            o[ ((o - o.min())/rrange) <  thresh ] = 0 
        return o 

    @staticmethod
    def _lab_to_rgb(img):
        return sk_color.lab2rgb( img ) 

    @staticmethod
    def _get_color_eq(img):
        ## 2. CLAHE/CStreching per channel 
        o = [NormalizerzThresholderzAlgz._get_channel_eq(img, i, eq_mtype=1) for i in range(3)] 
        o = np.dstack(o) 

        return o



class EncodeImage:
    '''
    @inputs:    image/fmap to encode, alread trained encoding model
    @outputs:   encoded image/fmap 
    @actions:   run predict fmap @ provided encoding model 
    @TODO:      > model training Vs loading from file 
    ''' 
    @staticmethod
    def encode_image(img, encoder_decoder):
        return EncodeImage.predict(encoder_decoder, img) 


    ### ===== 3. Pretraining Strategy et al TODO: Syncy up ZModel + ModelManager + ModelTrainer    
    ## TODO: move train/predict code to some shared manager 
    @staticmethod 
    def train(autoencoder, X_data, mname='vgg16', epochz=3,
                            loss_func = nn.MSELoss(),
                            optimizer = optim.SGD, 
                            optimizer_paramz = {'lr':1e-3, 'momentum':.9} 
                            ):
        # 1. go into training mode + setup optimizer
        autoencoder.train() 
        optimizer = optimizer( autoencoder.parameters(), **optimizer_paramz )  

        # 2. run epochs and per item in batch 
        for e in tqdm( range(epochz ) ):
            running_loss = 0
            n_inst = 0
            for x in X_data:
                x_ = x#.float() 
                # a. zero the grads, compute loss, backprop 
                optimizer.zero_grad()
                o_ = autoencoder( x_ ) 
                loss = loss_func(o_, x_ ) 
                loss.backward()
                optimizer.step() 
                # b. update aggregates and reporting 
                running_loss += loss.item() 
            
            print(f"E {e}: loss {running_loss}") 

            # freeze at each iteration TODO: rethink ++ at training manager proc 
            with open(f"../data/fundus_encoder_decoder_{mname}.pkl", 'wb') as fd:
                pickle.dump(autoencoder, fd ) 

    @staticmethod 
    def predict(autoencoder, x): 
        x_ = x#.float() 
        # 1. go into eval mode
        autoencoder.eval() 
        # 2. run through model 
        o_ = autoencoder( x_ )
        # 3. extract top-k 
        O_ = []
        for rec in o_:
            O_.append( rec.detach()  )
        
        return torch.stack( O_ )  ## why yield??? 

    @staticmethod 
    def eval(autoencoder):
        pass 

        
