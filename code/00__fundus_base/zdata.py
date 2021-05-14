'''
author: bg
goal: 
type: Data extractors, objects and access handlers 
how: 
ref: 
refactors: 
'''

import pandas as pd 
import numpy as np 

import os, time, glob #TODO: pathlib 
import pickle 

#import skimage as skimage 
from skimage import io, color 
import utilz 


from sklearn.model_selection import train_test_split

## === TODO: relevance + displayable 
class ZSerializableMixin: 
    def load(self, fpath):
        with open(fpath, 'rb') as fd:
            self.__dict__.update(pickle.load(fd)) ## TODO: 
            

    def dump(self, fpath):
        with open(fpath, 'wb') as fd:
            pickle.dump( self.__dict__, fd ) 

## === Collection handler << TODO: sklearn transformer/pipeline on it + Make useful 
class ZMultiModalRecord(ZSerializableMixin, list): 
    def __init__(self, itemz):
        super().__init__(itemz) 

## === Modality data types 
class ZModal(ZSerializableMixin):
    TYPE_GEN = 0
    TYPE_TEXT = 1
    TYPE_IMAGE = 2
    TYPE_TXT_FILE = 3 
    TYPE_PICKLE_FILE = 4 

    def __init__(self, label, data=None, mod_type=TYPE_GEN):
        self.label = label 
        self.data = data 
        self.mod_type = mod_type 

    @property
    def size(self):
        return 0 if self.data is None else len(self.data ) \
                    if not isinstance(self.data, (np.ndarray, np.generic)) else self.data.shape 

    @property
    def stats(self): #TODO: fix error on some dtype i don't remember 
        headerz = ['label', 'size', 'min', 'max', 'mean']
        statz = []
        if (isinstance( self.data ,(np.ndarray, np.generic, list, tuple)) ):
            dmean = round(np.mean(self.data),3)
            dmin = round(np.min(self.data),3)
            dmax = round(np.max(self.data),3)
            dsize = self.data.shape if isinstance( self.data ,(np.ndarray, np.generic)) else len(self.data)
        else:
            dmean = 'N/A'
            dmin = min(self.data)
            dmax = max(self.data)
            dsize = len(self.data)
        statz = [self.label, dsize, dmin, dmax, dmean]
        return statz, headerz 

    def __repr__(self):
        s, h = self.stats
        return f"{self.__class__}: n={self.size}, data.dtype={type(self.data)}\n\t{h} \n\t{s}"


## === Image Objects
class ZImage(ZModal):    
    def __init__(self, label, fpath, resize_dim=(224,224), cleanit=True):
        super().__init__(label, mod_type=ZModal.TYPE_IMAGE) 
        self.fpath = fpath
        self.cleanit = cleanit 
        self.data = io.imread(self.fpath )
        if self.cleanit:
            self.data = utilz.Image.basic_preproc_img(self.data, resize_dim  ) 

    @property        
    def gray(self): 
        return color.rgb2gray( self.data )  ## TODO: to clean or not to clean 

    @property
    def red(self):
        return utilz.Image.get_channel(self.data, 0)
    @property
    def green(self):
        return utilz.Image.get_channel(self.data, 1)
    @property
    def blue(self):
        return utilz.Image.get_channel(self.data, 2)  
       

## === Stats, Visualize and Generator @ listing I/O
class PdDataStats:
    TYPE_IN_MEM_ARRAY = 0 ## pass pair of data and headerz 
    TYPE_DIR = 1
    TYPE_TXT_LINES_FILE = 2
    TYPE_JSON_FILE  = 3

    DATA_DICT_RECORDZ_KEY = 'recordz'
    DATA_DICT_HEADERZ_KEY = 'headerz'
    DATA_DICT_HAS_HEADERZ_KEY = 'has_header_row'
    ##TODO: with some shared defaults or remove for not 
    loaderz = [ (None, {}),
                (utilz.FileIO.folder_content,{}), 
                (utilz.FileIO.file_content, {}), #'rec_parser': utilz.FileIO.row_parser, 'has_header_row':False
                (None,{})
            ]
    ## data = dict of {'recordz': (array|fpath ), 'headerz':(list), } Minimum 
    def __init__(self, data_dict, ftype=TYPE_IN_MEM_ARRAY):
        self.data = data_dict 
        self.ftype = ftype 
        self.dframe = None 
        self.load()  
    
    @property
    def size(self):
        return len(self.dframe) if self.dframe is not None else 0 
    
    ## TODO: lighten 
    def load(self, **kwargz):
            # has_header_row=False, rec_parser=None, sep='\t', 
            # ext="*.*", additional_info_func=None, fname_parser=None, sep='-'): 
        loader, default_kargz = self.loaderz[self.ftype] 

        recz = self.data.get(PdDataStats.DATA_DICT_RECORDZ_KEY, None)
        headerz = self.data.get(PdDataStats.DATA_DICT_HEADERZ_KEY, None)
        has_header_row = self.data.get(PdDataStats.DATA_DICT_HAS_HEADERZ_KEY, False) 

        if loader is not None: ## assume in mem otherwise 
            fkwargz = self.data.copy() ## TODO: b/c maintain init state from load 
            # print(fkwargz)
            fkwargz.pop(PdDataStats.DATA_DICT_RECORDZ_KEY, None)
            fkwargz.pop(PdDataStats.DATA_DICT_HEADERZ_KEY, None)
            fkwargz.pop(PdDataStats.DATA_DICT_HAS_HEADERZ_KEY, None)
            fpath = recz
            recz = []
            loader = loader(fpath, **{**kwargz, **fkwargz })##generator auch!!
            if has_header_row:
                headerz = next(loader )
            for r in loader:  ##TODO: pd deal iterator, nrows to read 
                recz.append(r) 

        if headerz is None:
            n = len(recz[0])
            headerz = [f'col_{i}' for i in range(n) ] 

        self.dframe = pd.DataFrame.from_records(recz, columns=headerz, coerce_float=True) 
        self.dframe.convert_dtypes( ) ## TODO: review again for consistency w/r/t NA or some fields

    ## TODO: menu of commons ELSE below guys are pointless since can access dframe directly 
    ### ==== Stats and Visuals ====
    def select_colz_by_name(self, colz=None):
        return self.dframe.loc[:,colz] if colz is not None else self.dframe
    
    def summarize(self, colz=None, include='all'):
        return self.select_colz_by_name(colz).describe(include=include)
    
    ## TODO: seaborn etc 
    # kind = https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html 
    def visualize(self, colz=None, countz=False, plt_type='bar', **kwargz):
        if countz: ## TODO: refactor, generalize(dframe Vs dseries), Common charts menu 
            return self.dframe[colz].value_counts().plot(kind=plt_type, **kwargz)
        else:
            return self.select_colz_by_name(colz).plot(kind=plt_type, **kwargz) 




## === Training Dataset - split etc << TODO: pytorch Dataset and crossvalidation  + numpy Vs tensor && cpu Vs cuda ops + autograd.Variable <<< Move from sklearn to pytorch dataset 
class ZPdDataset(PdDataStats):
    def __init__(self, data_dict, ftype=PdDataStats.TYPE_IN_MEM_ARRAY):
        super().__init__(data_dict, ftype )
        ## TODO: have these as masks and not dframes from sklearn 
        self.train_mask = None
        self.test_mask = None
        self.validation_mask = None 
    
    def train_test_validate_split(self, test_perc=0.3, validate_perc=0, shuffle=True):
        self.train_set, self.test_set = train_test_split(self.dframe, 
                                                    test_size=test_perc,
                                                    shuffle=shuffle,
                                                    random_state=999)
        print(f"Done splitting {test_perc}% test with shuffle = {shuffle}")



if __name__ == "__main__":
    print("CWD: ", os.getcwd() )
    import content 
    pdstats = PdDataStats(
                    {PdDataStats.DATA_DICT_RECORDZ_KEY: content.STARE_FUNDUS_CONTENT_FPATH,
                    PdDataStats.DATA_DICT_HAS_HEADERZ_KEY: True,
                    'rec_parser': utilz.FileIO.row_parser                     
                    },
                     ftype=PdDataStats.TYPE_TXT_LINES_FILE ) 
    
    pdstats.dframe 

    print("\n----\n\n") 

    pdstats = ZPdDataset(
                    {PdDataStats.DATA_DICT_RECORDZ_KEY:'/mnt/externz/zRepoz/datasets/fundus/stare',
                    'fname_parser': utilz.FileIO.row_parser,
                    'additional_info_func':utilz.FileIO.image_file_parser   ,
                    'ext':"*.ppm"                  
                    },
                     ftype=PdDataStats.TYPE_DIR ) 
    
    pdstats.dframe 

    print("\n----\n\n") 