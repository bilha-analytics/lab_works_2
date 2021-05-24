'''
author: bg
goal: 
type: preprocessing functions for different data types
how: algorithms 
ref: 
refactors: TODO: @ all txt, img data types or just one per file Vs Shared elements 
'''


import numpy as np 


from skimage import exposure, transform, filters 

import utilz 
from interfacez   import ZTransformableInterface
from transform_sharez import ChainableMixin, NormalizerzThresholderzAlgz 

import torch 
from torchvision import transforms 

class FileLoader(ZTransformableInterface):
    '''
    @inputs:    pdframe+fpath_field_name or list of paths 
    @outputs:   list of loaded data from given paths 
    @actions:   > fileIO (img, pkl, txt, csv???)
                > Option to resize/rescale and crop 
                > 
    @TODO:  generator or not + sync up dataloader approaches 
            other file/data  types    
    '''

    _file_readerz = {
        'img' : utilz.Image.fetch_and_resize_image,
        'pkl' : utilz.Image.fetch_and_resize_fmap, 
    }

    def __init__(self, listing_name, fileio='img', resize=None, crop_to_ratio=1):
        self.listing_name = listing_name 
        self.resize = resize
        self.crop_to_ratio = crop_to_ratio 
        self.fileio = fileio 

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        '''
        Assumes: listing_name = None if is an array of things else is a data frame 
        @TODO: generator or not
        ''' 
        if self.listing_name: ## we have a pd.dframe 
            X_ = X.loc[:, self.listing_name ] 
        else:
            X_ = X 
        O_ = []
        for fp in X_: 
            # print('FETCHING: ', fp, "IO mode = ", self.fileio )
            o = self._file_readerz[self.fileio](fp, self.resize )  
            o = utilz.Image.crop_image(o, self.crop_to_ratio )
            # yield o 
            O_.append( o ) 
        return O_ 


class ColorChannelSelector(ZTransformableInterface,ChainableMixin):
    '''
    @inputs:    list of channels to select
    @output:    new fmap 
    @actions:   select channels in provided order 
    ''' 
    def __init__(self, list_of_channels, trans_color_space=None, 
                    do_eq=False, append_to_current_fmap=False):
        super(ColorChannelSelector, self).__init__()
        self.append_to_current_fmap = append_to_current_fmap 
        self.list_of_channels = list_of_channels 
        self.trans_color_space = trans_color_space  #E.G. color.rgb2lab     
        self.do_eq = do_eq     

    def fit(self, X, y=None):
        return self 

    def transform(self, X_, y=None):
        O_ = [] 
        for x in X_:
            o = NormalizerzThresholderzAlgz._get_color_eq(x ) if self.do_eq else x.copy()   
            # print("INCOMING: ", o.shape , "CHAIN = ", self.append_to_current_fmap, "Cz = ", self.list_of_channels )
            if self.trans_color_space:
                o = self.trans_color_space( o )  ## TODO: deal with can skimage.color only do RGB can't do RGBA
            o = [ o[:,:,c] for c in self.list_of_channels ] ## will raise error if x not 3D
            o = self.chain_transforms(x, np.dstack(o) ) 
            # print("OUTGOING: ", o.shape )
            O_.append( o  ) 
        return O_ 


class EqualizerDenoiser(ZTransformableInterface, ChainableMixin):
    '''
    @inputs:    eq method/func + params,  sharpener method + params, blur/smooth func + params, etc 
    @outputs:   new cleaned image 
    @actions:   run image denoise in order of provided cleaners 
    ''' 

    def __init__(self, list_denoise_func_param_pairs=None, 
                    on_first_channel_only=False, 
                    append=True):   
        super(EqualizerDenoiser, self).__init__()  
        self.append_to_current_fmap = append 
        self.list_denoise_func_param_pairs = list_denoise_func_param_pairs
        self.on_first_channel_only = on_first_channel_only

    def fit(self, X, y=None):
        return self

    def transform(self, X_, y=None):
        O_ = [] 
        for x in X_:
            o = utilz.Image.rescale_and_float(x) #x.copy()             
            funcz = []
            ## 1. default denoise if none provided 
            if self.list_denoise_func_param_pairs is None: 
                p2, p98 = np.percentile(o, (2,98))  
                funcz.append( (exposure.rescale_intensity, {'in_range':(p2,p98)}) )
                funcz.append( (filters.median, {'mode':'nearest' } ) )
            else:
                funcz = self.list_denoise_func_param_pairs 

            # print("INCOMING: ", o.shape , "CHAIN = ", self.append_to_current_fmap, "Funcz = ", funcz ) 
            ## 2. run denoising per channel if mlti-color or something 
            for func, kwargz in funcz:
                if len( o.shape) == 2: 
                    o = func(o, **kwargz) 
                if len( o.shape) == 3:
                    o = np.dstack([ func(o[:,:,i], **kwargz) for i in range(o.shape[2] ) ] ) 
                else: ## n-D array and/or gigo on 1st channel only 
                    o = func(o[:,:,0], **kwargz) 

            ## 3. return 
            # print("OUTGOING ", o.shape , "b4 append to ", x.shape ) 
            O_.append( self.chain_transforms(x, o )  ) 

        return O_ 


class TLTensorfy(ZTransformableInterface):
    normalizer_TL =  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    def __init__(self, out_dim=(1, 3, 224, 224), rescale_TL=True, enforce_float=True): 
        super(TLTensorfy, self).__init__()  
        self.rescale_for_TL = rescale_TL 
        self.out_dim = out_dim 
        self.enforce_float = enforce_float 

    def fit(self, X_, y=None):
        return self 
    
    def transform(self, X_, y=None):
        # O_ = [ self.normalizer_TL(torch.tensor(o_.astype('f').reshape(self.out_dim) ) ) for x_ in X_]
        O_ = []
        for x_ in X_:
            o_ = x_.copy() 
            if self.enforce_float:
                o_ = o_.astype('f') 

            o_ = torch.tensor( o_.reshape(self.out_dim) ) 

            if self.rescale_for_TL: 
                o_ = self.normalizer_TL( o_ ) 
            O_.append( o_ )
        return O_ 
    

class TorchifyTransformz(ZTransformableInterface):
    '''
    @actions:   wrap torchvision transforms in sklearn transformable for pipeline and permutations
    ''' 
    ## TODO: menu of common transformz
    normalizer_TL =  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    # augmentor = transforms.

    def __init__(self, transformz_composed): 
        super(TorchifyTransformz, self).__init__()  
        self.transformz_composed = transformz_composed 

    def fit(self, X_, y=None):
        return self 
    
    def transform(self, X_, y=None):
        # O_ = [ self.normalizer_TL(torch.tensor(o_.astype('f').reshape(self.out_dim) ) ) for x_ in X_]
        O_ = []
        for x_ in X_:
            o_ = x_.copy() 
            ## TorchVision transformz op on PIL.Image or ndarray 
            # o_ = torch.tensor( np.transpose(o_, (2, 0, 1)) )  # TODO: confirm order ##tensorvision: C, H, W --> np.transpose C.H.W --> 012 
            o_ = self.transformz_composed( o_ ) 
            O_.append(o_) 
        return O_ 
    
class Reshapeor(ZTransformableInterface):
    def __init__(self, newshape):
        self.newshape = newshape         
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # a. flatten 
        if isinstance(self.newshape, int) and (self.newshape == -1): 
            O_ = [x.flatten() for x in X] 
        # b. reshape to tensor NCHW
        elif isinstance(self.newshape, int) and (self.newshape == -99): 
            O_ = [x.reshape( [1,]+list(reversed(x.shape)) ) for x in X] 
        # c. others on tuple 
        elif isinstance(self.newshape, int) and (self.newshape == 99): 
            O_ = [x.reshape( [1,]+list(x.shape) ) for x in X] 
        # c. others on tuple 
        else:
            O_ = [x.reshape( self.newshape ) for x in X] 
        # print( "3D -- input shape: ", X[0].shape, " Vs ", O_[0].shape )
        # print("RESHAPED: ", X[0].shape, ' to ', O_[0].shape )
        return O_

class YLabelzTorchify():
    pass 

class NdarrayToPILImage(ZTransformableInterface):    
    def fit(self, X, y=None):
        return self

    def transform(self, X_, y=None):
        O_ = []
        for x in X_:
            O_.append( utilz.Image.img_to_pil(x)  )   ##TODO: not just RGB
        return O_ 


