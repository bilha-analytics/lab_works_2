'''
author: bg
goal: 
type: extraction transformers 
how: algorithms 
ref: 
refactors: 
'''
import numpy as np

from skimage.feature import local_binary_pattern

import utilz 
from interfacez import ZTransformableInterface
from transform_sharez import ChainableMixin

class Patchify(ZTransformableInterface, ChainableMixin):
    '''
    @inputs:    list of images/fmaps, number of patches Vs patches size???
    @outputs:   list of transformed output appended to input if desired 
    @actions:   split image into patches
    @TODO:  define overlap/windowing mechanism 
            work with patch x,y as opposed to num_patches 
            FIX: errors in some patches (black patches)
    '''
    def __init__(self, nx_patchez,  
                    overlap_px=10, 
                    append=True): 
        self.append_to_current_fmap = append 
        self.nx_patchez = nx_patchez
        self.overlap_px = overlap_px  

    def fit(self, X_, y=None):
        return self

    def transform(self, X_, y=None): 
        O_ = []
        for x in X_:
            o = utilz.Image.patchify_image(x, self.nx_patchez, self.overlap_px)  
            ox, oy = o.shape[0], o.shape[1] 
            x_ = utilz.Image.resize_image_dim(x, dim=(ox,oy) ) 
            O_.append( self.chain_transforms( x_, o) ) 
        return O_ 


class Extractor(ZTransformableInterface, ChainableMixin):
    '''
    Generic object that wraps shared transform methods from builders or something 

    @inputs:    list of images/fmaps, append option , thresholding method 
    @outpus:    list of transformed outputs with/without append input 
    @actions:   E.G. 
                > apply thresholding on given channel using selected method 
                > something something 
    @TODO:      REFACTOR: arch for threshold algorithms --> interface/decorator/factory/something ??
    ''' 
    def __init__(self, method, method_kwargz, append=True): 
        self.append_to_current_fmap = append 
        self.method = method 
        self.method_kwargz = method_kwargz 

    def fit(self, X_, y=None):
        return self

    def transform(self, X_, y=None):
        O_ = []
        for x in X_:
            o = self.method(x, **self.method_kwargz) 
            print( type(o), type(x), (x - o).mean() )
            O_.append( self.chain_transforms(x, o) ) 

        return O_ 


class LBPExtract(ZTransformableInterface, ChainableMixin):    
    '''
    @inputs:    list of images/fmaps , append option, lbp paramz 
    @outputs:   transformed output appended to input 
    @actions:   apply lbp on green channel by default 
    '''  
    def __init__(self, target_channel, 
                lbp_radius=1, lbp_method='uniform',
                append=True): 
        self.append_to_current_fmap = append 
        self.target_channel = target_channel 
        self.lbp_radius = lbp_radius
        self.lbp_method = lbp_method 
        self.lbp_k = 8*lbp_radius 

    def fit(self, X_, y=None):
        return self 
    
    def transform(self, X_, y=None):
        O_ = []

        for x in X_:
            o = x[:,:,self.target_channel]  ## assumes equalization, cleaning etc has been done already 
            o = local_binary_pattern(o, self.lbp_k, self.lbp_radius, self.lbp_method)
            O_.append( self.chain_transforms(x, o) )

        return O_ 

