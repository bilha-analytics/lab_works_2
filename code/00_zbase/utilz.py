'''
author: bg
goal: 
type: util 
how: 
ref: 
refactors: 
'''

import os, time, glob #TODO: pathlib 
import pickle 
from datetime import datetime

import numpy as np 

from PIL import Image as PILimg 

import skimage 
from skimage import io, color , img_as_float, exposure, transform, filters 

import matplotlib.pyplot as plt 

### ==== Image Utilz
class Image:

    @staticmethod
    def fetch_and_resize_image(fpath, size=None):
        try:
            img = io.imread( fpath )
            return img if size is None else Image.resize_image_dim( img, dim=size)
        except:
            return None 

    @staticmethod
    def fetch_and_resize_fmap(fpath, size=None): ##TODO: check resizing with 3++ channels 
        try:
            img = FileIO.unpickle( fpath ) 
            return img if size is None else Image.resize_image_dim( img, dim=size)
        except Exception as e:
            print("SOmething went wrong: ", str(e))
            return None 


    @staticmethod
    def crop_image(img, crop_ratio=1): 
        ## crop to given ratio 
        x,y = img.shape[0] , img.shape[1]
        ox, oy =int(x*crop_ratio), int(y*crop_ratio) 
        ix, iy = (x-ox)//2, (y-oy)//2
        img = img[ ix:ix+ox, iy:iy+oy, :] if len(img.shape) > 2 else  img[ ix:ix+ox, iy:iy+oy]
        return img 
    
    @staticmethod
    def gray_scale(img):
        return color.rgb2gray(img) 
        
    @staticmethod
    def get_channel(img, cid): 
        c = len(img.shape)
        nc = img.shape[2] 
        return img[:,:,cid] if (c >=3 and cid<nc) else None 

    @staticmethod
    def rescale_and_float(img):
        o = img.copy()
        o = o/255 ## assumes 
        o = img_as_float(o) ## recheck with o/255  
        return o 

###======================
    '''
    @TODO:  > check upsample methods: e.g. zero padding Vs etc 
            > check downsampling methods e.g. resacle, pyramid, 
            > @both interpolations used <-- bilinear good fundus??
    ''' 
    @staticmethod
    def resize_image_perc(img, p=0.25):
        o = transform.rescale(img, p, anti_aliasing=True, multichannel=True)   
        return o
    
    @staticmethod
    def resize_image_dim(img, dim=(50,50) ): ##TODO: aspect ratio and padding to max dim
        return transform.resize(img, dim, anti_aliasing=True )
    
###======================
    @staticmethod
    def hist_eq(img, mtype=1): ## TODO: mtype consts
        # print("HIST_EQ_In: ", img.shape )
        p2, p98 = np.percentile(img, (2,98)) 
        mtypez = [
            (exposure.equalize_adapthist, {'clip_limit':0.03}),
            (exposure.rescale_intensity, {'in_range':(p2,p98)}), 
        ]
        pmod, kargz = mtypez[mtype]
        return pmod(img, **kargz)
   
    @staticmethod
    def edgez(img, mtype=0): ## TODO: mytpe  
        sharez = {} #'black_ridges': False} #TODO: shared params @ API setup 
        mtypez = [
            (filters.frangi, {'sigmas':range(1, 9, 2), 'gamma':15, 
                                'alpha':0.5, 'beta':0.5,}), # {'sigmas':range(4,10,2), 'black_ridges':1, 'alpha':0.75}), 
            (filters.sato, {}),
            (filters.meijering, {})
        ]
        pmod, kargz = mtypez[mtype] 
        return pmod(img, **{**sharez, **kargz} )

    @staticmethod
    def denoise(img, mtype=0): ##TODO
        o = img.copy() 
        return o
     
    @staticmethod #TODO: histo eq non-gray imagez
    def basic_preproc_img(img, dim=(50,50), denoise_mtype=0): #image resize, rescale, equalize as bare minimum preprocessing
        return Image.resize_image_dim(
                Image.rescale_and_float(img), dim
            )
        # return img 
    @staticmethod
    def plot_images_list(img_list, titlez=None, nc=2, cmap=None, tstamp=False, spacer=0.01, 
                         save=None , tdir=".", savedpi=800, withist=False, binz=None, tfont=3):
       
        if withist:   
            n = len(img_list)*2
            nr = n//nc + ( 0 if n%nc == 0 else 1) 
        else:
            n = len(img_list)
            nr = n//nc + ( 0 if n%nc == 0 else 1) 
            
        ## image rows
        for i, img in enumerate(img_list):
            plt.subplot(nr, nc, (i+1) )
            plt.imshow( img, cmap=cmap)  #.astype('uint8')
            plt.axis('off')
            if titlez and (i<len(titlez)):
                plt.title( f"{titlez[i]}", fontsize=tfont ) #min(i, len(titlez)-1)
        
        ## histo rows 
        if withist:      
            for i, img in enumerate(img_list):
                plt.subplot(nr, nc, (i+1)+(n//2) )
                plt.hist(img.flatten()*(1/img.max()), bins=binz)
                plt.tick_params(axis='y', which='both', labelleft=False, labelright=False) #TODO:off
                
        plt.subplots_adjust(wspace=spacer, hspace=spacer)
        
        if save:
            d = datetime.now().strftime("%H%M%S")
            fnout = f"{d}_{save}" if tstamp else f"{save}"
            plt.savefig(f"{tdir}/{fnout}.png", dpi=savedpi, 
                    facecolor='white', edgecolor='none', transparent=False)
        
        plt.show();

    @staticmethod 
    def patchify_image(img, nx_patchez, overlap_px=10):
        x, y = img.shape[0]  , img.shape[1]
        padded_wx = ((x//nx_patchez)*nx_patchez)  + nx_patchez 
        padded_hy = ((y//nx_patchez)*nx_patchez) + nx_patchez 

        c = img.shape[2] if len(img.shape) == 3 else -1  ##TODO: iff not 3d array 
        if c > 0:
            oimg = np.zeros( (padded_wx, padded_hy, c) ) #np.zeros_like 
            oimg[:x, :y, :] = img 
        else:
            oimg = np.zeros( (padded_wx, padded_hy) )
            oimg[:x, :y] = img if len(img.shape) == 2 else img[:,:,0] ## catch gigo 
        print(oimg.shape) 

        O_ = []
        px = padded_wx//nx_patchez
        py = padded_hy//nx_patchez
        for i in range(nx_patchez): 
            for j in range(nx_patchez):                 
                if c> 0:
                    O_.append( oimg[ (i*px):((i+1)*px), (j*py):((j+1)*py), :] ) 
                else:
                    O_.append( oimg[ (i*px):((i+1)*px), (j*py):((j+1)*py) ] ) 
        O_ = np.dstack(O_)  
        print('patch.dim: ', O_.shape )
        return O_ 

    @staticmethod
    def img_to_pil(img):
        return PILimg.fromarray(np.uint8( img ), 'RGB' )         
    
### ==== FileIO 
class FileIO:

    @staticmethod
    def unpickle(fpath):
        with open(fpath, 'rb') as fd:
            return pickle.load( fd ) 

    @staticmethod 
    def dump_pickle(fpath, dat):
        with open(fpath, 'wb') as fd:
            pickle.dump( dat, fd) 

    @staticmethod
    def file_content(fpath, rec_parser=None, sep='\t'):
        with open(fpath, 'r') as fd:
            # print( fpath )
            for rec in fd.readlines(): 
                # print(rec)
                yield rec if rec_parser is None else rec_parser(rec, sep) 
    
    @staticmethod
    def folder_content(fpath, ext="*.*", additional_info_func=None, fname_parser=None, sep='-'): 
        for f in sorted(glob.glob(f"{fpath}/{ext}")): 
            fname = os.path.splitext( os.path.basename(f) )[0]
            fname = [fname,] if fname_parser is None else fname_parser(fname, sep)
            # print( fname )
            xtraz = [] 
            if additional_info_func:
                xtraz = additional_info_func(f) 
            yield [*fname, *xtraz] 
    
    @staticmethod
    def row_parser(rec, sep='\t'):
        outiez = rec.strip().split(sep)
        # print(outiez)
        return [x.strip() for x in outiez if len(x) > 0] ##TODO: clean up paranoia 
    
    @staticmethod
    def image_file_parser(fpath):
        # print( fpath )
        outiez = []
        img = io.imread(fpath)
        outiez.append( img.shape ) 
        outiez.append( img.min() ) 
        outiez.append( img.max() ) 
        outiez.append( img.mean() ) 
        outiez.append( img.std() )
        img = None 
        return outiez 

### ==== Graphs/Plots


### ==== Tabular Charts/Plots 