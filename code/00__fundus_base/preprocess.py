'''
author: bg
goal: 
type: Data preprocessing tools = column/field/data cleaning transformerz
how: 
ref: 
refactors: 
'''

import  utilz 

from sklearn.base import TransformerMixin, BaseEstimator 
from sklearn.preprocessing import FunctionTransformer 


from sklearn.compose import TransformedTargetRegressor  ## TODO: for ylabelz  ++ one-hot-encoding, 

import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader 



### 1. flatten data 
# TODO: numpy.ravel, since it doesn't make a copy of the array, but just return a view of the array, which is much faster than numpy.flatten. 
class Flattenor(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        # print("Calling Falttenor.fit")
        return self
    def transform(self, X, y=None):
        O_ = [x.ravel() for x in X] 
        # print( "Flattenor -- input shape: ", X[0].shape, " Vs ", O_[0].shape )
        return O_
## 2. reshapes e.g. (1, xxx), or from flat to 2D etc 
class Reshapeor(TransformerMixin, BaseEstimator):
    def __init__(self, newshape):
        self.newshape = newshape         
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        O_ = [x.reshape( self.newshape ) for x in X] 
        # print( "3D -- input shape: ", X[0].shape, " Vs ", O_[0].shape )
        return O_

## 3. load from file 
class LoadImageFileTransform(TransformerMixin, BaseEstimator):
    def __init__(self, fpath_colname, resize=(224,224), crop_ratio=1): 
        self.fpath_colname = fpath_colname 
        self.resize = resize 
        self.crop_ratio=crop_ratio 
    def fit(self, X, y=None):
        return self    
    def transform(self, X, y=None): ##TODO: memory mgt 
        fpathz = X.loc[:, self.fpath_colname ] ## from pdframe 
        # print( len(fpathz) , "============******<<<<<<< load image")
        O_ = []
        for fp in fpathz:
            img = self.get_image_from_file(fp)  
            O_.append( img ) 
        return O_      
    def get_image_from_file(self, fp):
        img = utilz.Image.fetch_and_resize_image(fp, self.resize) 
        x,y, _ = img.shape 
        ox, oy =int(x*self.crop_ratio), int(y*self.crop_ratio) 
        ix, iy = (x-ox)//2, (y-oy)//2
        img = img[ ix:ix+ox, iy:iy+oy, :] 
        return img 

## 4. tensor to numpy 
class ToTensor(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        O_ =  [ np.array(x).astype(np.float32) for x in X]  ## TODO: list of tensors or tensor of tensors??
        return [ torch.tensor(x.reshape(1, *x) ) for x in O_]  ## TODO: list of tensors or tensor of tensors??

## 5. to dataloader << TODO: torch.transforms + sync@PdDataStats and/or content.py 
class ListToDataLoader(TransformerMixin, BaseEstimator):
    # train_data = torch.utils.data.TensorDataset(x_train, y_train)
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=self.shuffle)
        
    class ZDSet(Dataset): ## TODO: ownership and reuse 
        def __init__(self, listing):
            self.listing = listing  
        def __len__(self):
            return len(self.listing) 
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            return self.listing[idx] 
            
    def __init__(self, kwargz): #batchsize=64, shuffle=True, n_workers=6 
        self.kwargz = kwargz
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        O_ = DataLoader( ZDSet(X), **self.kwargz ) 
        return O_  ## TODO: list of tensors or tensor of tensors??
