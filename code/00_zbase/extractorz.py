'''
author: bg
goal: 
type: extraction transformers 
how: algorithms 
ref: 
refactors: 
'''
from os import EX_OK
import numpy as np


from skimage import color as sk_color
from skimage.feature import local_binary_pattern

import utilz 
from interfacez import ZTransformableInterface
from transform_sharez import ChainableMixin, NormalizerzThresholderzAlgz 

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
        super(Patchify, self).__init__() 
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
        super(Extractor, self).__init__() 
        self.append_to_current_fmap = append 
        self.method = method 
        self.method_kwargz = method_kwargz 

    def fit(self, X_, y=None):
        return self

    def transform(self, X_, y=None):
        O_ = []
        for x in X_:
            o_ = self.method(x, **self.method_kwargz)  
            o_ = self.chain_transforms(x, o_) 
            O_.append( o_ ) 
        return O_ 


class PseudoColorChannelz(ZTransformableInterface, ChainableMixin):
    '''
    @inputs:    RGB image array, list of channel_funcs_argz tuples for thresholded transforms 
    @actions:   apply methods to get thresholded versioz 
    @output:    fmap of color thresholded image 
    @TODO: refactor channel selection and proc methods; clean it up good 
    ''' 
    _default_threshold_fmap_on_origi_RGB = {
        'red': (0, NormalizerzThresholderzAlgz.thresh_double_red2, {'thresh': .97, 'color_space': 'rgb', 'do_eq': True}),
        'yellow': (1, NormalizerzThresholderzAlgz.thresh_yellow, {'thresh': .4, 'threshit': True, 'color_space': 'lab', 'clear_ring': .9, 'do_eq': True} ),
        'blue': (2, NormalizerzThresholderzAlgz.thresh_blue, {'thresh': 1, 'color_space': 'rgb', 'do_eq': True, 'clear_ring': .9}),
        'darks': (1, NormalizerzThresholderzAlgz.thresh_darks_blend_but, {'thresh': .97, 'color_space': 'rgb', 'clear_ring': .9, 'do_eq': True}),
    }

    DEFAULTZ_TITLEZ = ['origi', 'red', 'yellow', 'blue', 'darks']

    _color_spacez = {
        'lab': sk_color.rgb2lab, 
    }
    def __init__(self, funcz=None): 
        super(PseudoColorChannelz, self).__init__()
        self.funcz = self._default_threshold_fmap_on_origi_RGB if funcz is None else funcz 

    def fit(self, X_, y=None):
        return self

    def transform(self, X_, y=None): ## TODO: refactor color space stuff 
        O_ = [] 
        for x in X_: 
            o_ = [] 
            for k, method in self.funcz.items():
                c, func, argz  = method 
                kwargz = argz.copy() 
                color_space = kwargz.pop('color_space') 
                if color_space in list(self._color_spacez.keys()):
                    if k == 'yellow':
                        o = self._color_spacez[color_space](NormalizerzThresholderzAlgz._get_color_eq(x))[:,:,c] 
                    else:
                        o = self._color_spacez[color_space](x)[:,:,c]
                else:
                    o = x[:,:,c]
                o_.append( func(o, **kwargz ) ) 
            O_.append( self.chain_transforms(x, np.dstack(o_)) )
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
        super(LBPExtract, self).__init__()  
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

