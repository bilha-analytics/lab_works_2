'''
author: bg
goal: 
type: feature extraction transforms e.g. channels, coord-system, 
how: 
ref: 
refactors: 
'''

import utilz

from skimage.filters import unsharp_mask, gaussian
from skimage import img_as_ubyte, img_as_float, img_as_uint
from skimage import color as sk_color
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator 
from sklearn.decomposition import PCA 

from skimage.feature import local_binary_pattern

### TODO: remapped_data home/owner/mixin + save to file + skip/shortcuts use implications 
### ==== 1. Fundus  Color channelz ==== << TODO: beyond fundus 
class ColorChannelz(TransformerMixin, BaseEstimator):
    # def __init__(self):
    #     pass 

    def fit(self, X, y=None):
        return self 

    def transform(self, X, y=None):
        return [self.remapped_data(x) for x in X] 

    
    ## --- TODO: arch/struct these three well + decouple @clean_data Vs data
    def _get_channel_eq(self, img, c=-1, eq_mtype=1): ## -1 is on gray scale 
        return utilz.Image.hist_eq( utilz.Image.get_channel(img, c), mtype=eq_mtype ) if c >= 0 \
            else utilz.Image.hist_eq( utilz.Image.gray_scale(img), mtype=eq_mtype  )
    
    def _get_lab_img(self, img, extractive=True): 
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
    def _get_yellow_from_rgb2lab(self, img, threshit=False, thresh=0.97): 
        o = sk_color.rgb2lab( img )  
        ##yellow is in A and is -ves
        o = o[:,:,1]
        o[o >= 0 ] = 0
        o = -1 * o  
        if threshit:
            rrange = o.max() - o.min() 
            o[ ((o - o.min())/rrange) <  thresh ] = 0 
        return o 

    def _lab_to_rgb(self, img):
        return sk_color.lab2rgb( img ) 

    def _get_color_eq(self, img):
        ## rgb to lab --> equalize l --> lab to rgb  <<< TODO: move to utilz         
        ## REF: d-hazing and underwater images 
        # 1. LAB color space intensity Vs luminous 
        # a. rgb2lab -> clahe(l) -> lab2rgb
        # o = sk_color.rgb2lab( img ) 
        # eq_l = self._get_channel_eq( img_as_uint(o), 0, eq_mtype=0) 
        # o = np.dstack( [eq_l, o[:,:,1], o[:,:,2]])
        # ## b. rgb2gray -> gradient smooth -> gray to rgb 
        # o = sk_color.lab2rgb( o ) 

        ## 2. CLAHE/CStreching per channel 
        o = [self._get_channel_eq(img, i, eq_mtype=1) for i in range(3)] 
        o = np.dstack(o) 

        return o

    def remapped_data(self, img):  
        return self._get_channel_eq(img, 1) 


    def _clear_outter_circle(self, img, b_thresh=0.9):
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
 
### TODO: at choice of cleaning
class OrigiCleanedChannelz(ColorChannelz): 
    def remapped_data(self, img):  ## simply clahe something something 
        o = [self._get_channel_eq(img, i) for i in range(3)] 
        o = np.dstack(o) 
        # if len(img.shape) <= 2:
        #     return np.dstack([img, o]) if self.append_component else o 
        # else:
        #     return np.dstack([*[img[:,:,i] for i in range(c)], o]) if self.append_component else o  

        return self._get_color_eq(img)

class FundusColorChannelz(ColorChannelz):  
    
    def __init__(self, add_origi=True , red_thresh=0.97, color_space='rgb'):
        super(FundusColorChannelz, self).__init__()  
        self.add_origi = add_origi   
        self.red_thresh = red_thresh 
        self.color_space = color_space 
    
    ## --- TODO: arch/struct these three well + decouple @clean_data Vs data 
    def _green_channel_update(self, img):
        return self._get_channel_eq(img, 1) 

    def _red_channel_update(self, img, thresh=0.97):
        if self.color_space == 'lab':
            ## redz are positives 
            o = self._get_channel_eq( self._get_lab_img( img ), 1) 
        else:
            o = self._get_channel_eq(img, 0) 

        rrange = o.max() - o.min() 
        o[ ( o - o.min() /rrange ) < self.red_thresh ] = 0 
        # o[ ( (o - o.min() )/rrange ) < self.red_thresh ] = 0    
        # o[ ( (o - o.mean() )/np.std(o) ) < self.red_thresh ] = 0   
        return o  

    def _blue_channel_update(self, img, thresh=1): 
        # if self.color_space == 'lab':
        #     o = img[:,:,-1].copy()
        #     # print( o.min(), o.max() ) 
        #     # 2. now threshold the blue <<< TODO: auto-find the 'unfazzed' pixel <<<< TODO: Is LAB giving more color infor that we can do spectral analysis on or is best jut for pulling out intensity without affecting luminance
        #     o = img_as_ubyte(o)
        #     o[ o != thresh] = 0 #(thresh-o.min()+0.00001)/(o.max() - o.min())] = 0         
        #     o[ o == thresh] = 255  

        # else:
        o = img_as_ubyte(img[:,:,-1].copy() )
        # print( o.min(), o.max() ) 
        o[ o != thresh] = 0 #(thresh-o.min()+0.00001)/(o.max() - o.min())] = 0         
        o[ o == thresh] = 255   
        #o = utilz.Image.hist_eq( o )

        # o = self._get_channel_eq(img, 2)  
        # _omax = o.max() 
        # t = 1 if (thresh == 1 and o.max()==255) else (1/_omax) ##TODO: change o.max to dtype check + else case blue is lost;recompute thresh
        # o[ o != t] = 0
        return img_as_float(o) #.astype('uint8')

    def _vessels_channel(self, img, mtype=0):
        o = self._get_channel_eq(img, 2) ## on green channel 
        o = utilz.Image.edgez(o, mtype) #* 255
        return o #.astype('uint8')
    
    def remapped_data(self, img): 
        outiez = []
    
        # 1. vessels --- using green channel 
        outiez.append( self._vessels_channel(img ) ) 

        # 2. Color INFO
        # 2a. Green Channel clean up @ contrast and full info 
        outiez.append( self._green_channel_update( img )  ) 

        # # 2b. RGB Vs LAB @ red and blue spectrum 
        # if self.color_space == 'lab':
        #     cimg = self._get_lab_img( img )
        # else: 
        #     cimg = img #.copy() 
        # outiez.append( self._red_channel_update( cimg , self.red_thresh) )
        # outiez.append( self._blue_channel_update( img ) )   

        ## For Now Run RGB blue  and LAB red <<< TODO add switch 
        outiez.append( self._red_channel_update( img ) )
        outiez.append( self._blue_channel_update( img ) )   

        # append yellow for pigmentation
        outiez.append( self._get_yellow_from_rgb2lab(img) )

        # 3.  append origi as cleaned only 
        if self.add_origi:
            # _ = [outiez.append(self._get_color_eq(img) ) for i in range(3)]  
            outiez.append(self._get_color_eq(img) )
        # 4. combine color channelz
        o = np.dstack(outiez) 
        return o 


class FundusAddLBP(ColorChannelz):

    def __init__(self, g_channel, lbp_radius = 1, lbp_method = 'uniform', append=True):
        super(FundusAddLBP, self).__init__()  
        self.g_channel = g_channel 
        self.lbp_radius = lbp_radius 
        self.lbp_k = 8*lbp_radius
        self.lbp_method = lbp_method         
        self.append = append 

    def remapped_data(self, img):
        O_ = []
        o = self._get_channel_eq( img[:,:, self.g_channel]   )
        o = local_binary_pattern( o, self.lbp_k, self.lbp_radius, self.lbp_method)
        
        if not self.append:
            return o 

        c = img.shape[2]
        O_.append( o )
        O_ += [ img[:,:,i] for i in range(c)] 
        return np.dstack(O_) 
        
# Filter
class ChannelzSelector(TransformerMixin, BaseEstimator):
    def __init__(self, ls_channelz=(1,)):
        self.ls_channelz = ls_channelz 

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [ x[:,:,self.ls_channelz] for x in X] 

### ==== 2. Eigenz ==== 
## EigenFeatures/Matrix Decomposition component selection based -- SVD, PCA, Selection 
## e.g. PCA using randomized SVD : https://scikit-learn.org/stable/modules/decomposition.html 
class EigenzChannelz(TransformerMixin, BaseEstimator): 
    ## TODO: Menu correctness and indexing + Pipeline @ use SKLearn decompositions 
    TYPE_PCA = 0
    TYPE_SELECT = 1
    TYPE_PCA_SELECT = 2
    TYPE_LCA = 3
    ## Per Channel Vs Overall Vs Both  TODO: correctness and indexing 
    PER_CHANNEL = 0
    PER_FULLIMG = 1
    PER_CHANNEL_FULLIMG = 2 

    _selectorz = [
        (PCA, {'svd_solver': 'randomized', 'whiten':True} ) 
    ]
       ## --- TODO: arch/struct these three well + decouple @clean_data Vs data
    def _get_channel_eq(self, img, c=-1, eq_mtype=1): ## -1 is on gray scale 
        return utilz.Image.hist_eq( utilz.Image.get_channel(img, c), mtype=eq_mtype ) if c >= 0 \
            else utilz.Image.hist_eq( utilz.Image.gray_scale(img), mtype=eq_mtype  )
    

    def __init__(self, g_channel, topn, mtype=TYPE_PCA, mlevel=PER_CHANNEL, 
                append_component=True, clahe=False):
        self.g_channel = g_channel 
        self.topn = topn 
        self.mtype = mtype 
        self.mlevel = mlevel 
        self.append_component = append_component 
        self.clahe = clahe

        m, kargz = self._selectorz[mtype] 
        self.component_selector = m(n_components=topn, **kargz)

    def _get_op_channel(self, x):
        c = len(x.shape) 
        if c <= 2:
            o = x
        else:
            o = x[:,:,self.g_channel] #[for c in x] 
        # print( f"From {c} to {o.shape }")
        if self.clahe:
            o = utilz.Image.hist_eq(o, 1)

        return o.flatten()  #.reshape(1, -1) #

    def fit(self, X, y=None):
        ## first fit component_selector before transform 
        self.component_selector.fit( [self._get_op_channel(x) for x in X ]  )  ## np.vectorize 
        # print("**** FIN FIT ****")
        return self 

    def transform(self, X, y=None):
        ## first fit component_selector before transform 
        return [self.remapped_data(x) for x in X ]   
        
    # per channel 
    def remapped_data(self, img): ## appends to the stack unless otherwise

        ### TODO: per channel for now operating on one channel only but can append that to the original imag
        x,y, c = img.shape 

        o = self._get_op_channel(img) 

        to = self.component_selector.transform([o,])[0]

        tx =  int( len(to)  * 0.5 * (x/y) )
        to = to.reshape( (tx, -1) )
        o = np.zeros((x,y))
        # print( f"len = {len(to)}, tx = {tx}, to={to.shape} for img={img.shape} on o={o.shape}")
        ox,oy = to.shape 
        o[:ox, :oy] = to 
        
        # print("FIN-Egz: In: ", img.shape, " Out: ", o.shape , " on to=", to.shape, "append = ", self.append_component) 
        
        if len(img.shape) <= 2:
            return np.dstack([img, o]) if self.append_component else o 
        else:
            return np.dstack([*[img[:,:,i] for i in range(c)], o]) if self.append_component else o  

    
### ==== 3. Patchify ==== <<< overlapping or not 
### TODO: refactor image dimensions calc  + Overlap size 
class PatchifyChannelz(TransformerMixin, BaseEstimator):
    def __init__(self, nx_patchez=9, origi_dim=(224, 224, -1), overlap_px = 10 ):
        self.nx_patchez = nx_patchez 
        self.origi_dim = origi_dim 
        self.overlap_px = overlap_px 
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.remapped_data(x) for x in X]  

    def remapped_data(self, img):
        return utilz.Image.patchify_image(img, self.nx_patchez) 




### ==== 4. vessel structures ====

class FundusVesselsChannelz(ColorChannelz):  
    
    def __init__(self, add_origi=True , vesssels_thresh=0.9, red_thresh=0.97, color_space='lab'):
        super(FundusVesselsChannelz, self).__init__()  
        self.add_origi = add_origi   
        self.red_thresh = red_thresh 
        self.vesssels_thresh = vesssels_thresh 
        self.color_space = color_space 
    
    ## --- TODO: arch/struct these three well + decouple @clean_data Vs data 
    def _green_channel_update(self, img):
        return self._get_channel_eq(img, 1) 

    def _red_channel_update(self, img ):
        o = self._get_channel_eq(img, 0) 
        rrange = o.max() - o.min() 
        o[ ((o - o.min())/rrange) <  self.red_thresh ] = 0  
        # o[( (o - o.mean())/(np.std(o) * np.sqrt(o.shape[0]*o.shape[1] )) ) < (self.vesssels_thresh) ] = 0   
        # o[( (o - o.mean()) / ( np.std(o) ) ) < (self.red_thresh) ] = 0   
        return o  

    def _blue_channel_update(self, img, thresh=1): 
        o = img_as_ubyte(img[:,:,-1].copy() )
        o[ o != thresh] = 0 
        o[ o == thresh] = 255   
        return img_as_float(o) #.astype('uint8')

    def _vessels_channel_update(self, img, chrom=1 ):
        o = sk_color.rgb2lab( img )[:,:, chrom]   
        o[ o <= 0 ] = 0
        o = self._get_channel_eq( o )   
        rrange = o.max() - o.min() 
        # o[ ((o - o.min())/rrange) > (1 - self.vesssels_thresh) ] = 1
        o[( (o - o.mean())/np.std(o) ) < (self.vesssels_thresh) ] = 0
        return o 

    def _vessels_external(self, img, mtype=0): ##0,1,2 = 'frangi, sato, meijering, '
        o = self._get_channel_eq(img, 2) ## on green channel 
        o = utilz.Image.edgez(o, mtype) #* 255
        return o #.astype('uint8')
    
    def remapped_data(self, img): 
        outiez = []
    
        # 1. vessels --- using green channel 
        outiez.append( self._vessels_external(img ) ) 

        # 2. Color INFO
        # 2a. Green Channel clean up @ contrast and full info 
        outiez.append( self._green_channel_update( img )  )    

        ## For Now Run RGB blue  and LAB red <<< TODO add switch 
        outiez.append( self._red_channel_update(  self._get_lab_img( img ) ) )
        outiez.append( self._blue_channel_update( img ) )   
        # append yellow for pigmentation
        outiez.append( self._get_yellow_from_rgb2lab(img) )


        # 2b. RGB Vs LAB @ red and blue spectrum 
        outiez.append( self._vessels_channel_update( img ) )
        # b from Lab
        ob = sk_color.rgb2lab( img ) 
        bg = ob[:,:, 2] 
        # bg[ bg <= 0 ] = 0
        rrange = bg.max() - bg.min() 
        bg[ ((bg - bg.min())/rrange) > ( self.vesssels_thresh) ] = 0   
        outiez.append( self._get_channel_eq(bg)*-1   ) 


        # 3.  append origi as cleaned only 
        if self.add_origi:
            outiez.append(self._get_color_eq(img) ) 
        # 4. combine color channelz
        o = np.dstack(outiez) 
        return o 





### ===== 5. Psuedo Color Modes =========

class PseudoModalChannelz(ColorChannelz):  
    
    def __init__(self, g_channel, topn, 
                lbp_radius = 1, lbp_method = 'uniform', 
                add_origi=True , vesssels_thresh=0.9, red_thresh=0.97, yellow_thresh=0.4, color_space='lab'):
        super(PseudoModalChannelz, self).__init__()  
        self.add_origi = add_origi   
        self.red_thresh = red_thresh 
        self.yellow_thresh = yellow_thresh 
        self.vesssels_thresh = vesssels_thresh 
        self.color_space = color_space 

        self._fitted_eigen = EigenzChannelz(g_channel, topn, append_component=False, clahe=True) 

        self._lbp = FundusAddLBP(g_channel, lbp_radius, lbp_method, append=False)     
    
    def fit(self, X, y=None):
        self._fitted_eigen = self._fitted_eigen.fit(X, y) 
        return self

    def _red_channel_update_origi(self, img ):
        if self.color_space == 'rgb':
            o = self._get_channel_eq(img, 0) 
        else:
            o = self._get_channel_eq( self._get_lab_img( img ), 1) 

        om = o.min()  #o.mean() 
        rrange = o.max() - o.min()  #np.std(o) 
        o[( (o - om) / rrange ) < (self.red_thresh) ] = 0   
        return o  

    def _red_channel_update2(self, img ):
        if self.color_space == 'rgb':
            o = self._get_channel_eq(img, 0) 
        else:
            o = self._get_channel_eq( self._get_lab_img( img ), 1) 

        rrange = o.max() - o.min() 
        o1 = o.copy()
        o2 = o.copy()

        o1[ ((o - o.min())/rrange) <  self.red_thresh ] = 0  
        # o[( (o - o.mean())/(np.std(o) * np.sqrt(o.shape[0]*o.shape[1] )) ) < (self.vesssels_thresh) ] = 0   
        o2[( (o - o.mean()) / np.std(o) )  < (self.red_thresh) ] = 0   
        o = o2 + (o1*(1+o2) ) #/ np.sqrt(o.shape[0]*o.shape[1] ) )
        return o 

    def _blue_channel_update(self, img, thresh=1): 
        o = img_as_ubyte(self._get_channel_eq( img[:,:,-1] ) )
        o[ o != thresh] = 0 
        o[ o == thresh] = 255   
        return img_as_float(o) #.astype('uint8')     

    def _some_channel(self, img, c_n, c_thresh=0.97) :  
        if self.color_space == 'rgb':
            o = self._get_channel_eq(  img  , c_n)  
        else:
            # o = self._get_channel_eq(  sk_color.rgb2lab( img ) , c_n) 
            o = self._get_channel_eq( self._get_lab_img( img, extractive=False ), c_n ) 

        rrange = o.max() - o.min() 
        o1 = o.copy()
        o2 = o.copy()

        o1[ ((o - o.min())/rrange) <  c_thresh ] = 0  
        # o[( (o - o.mean())/(np.std(o) * np.sqrt(o.shape[0]*o.shape[1] )) ) < (self.vesssels_thresh) ] = 0   
        o2[( (o - o.mean()) / ( np.std(o) ) ) < (c_thresh) ] = 0   
        o = o2 + (o1*o2) #/ np.sqrt(o.shape[0]*o.shape[1] ) )
        
        ## set outter circle to zero b/c capturing glare as well 
        o = self._clear_outter_circle(o, b_thresh=0.75) 

        return o 
    
    def _darks_blend_but(self, img ):
        o = self._get_channel_eq(img, 0)    
        rrange = np.std(o)
        om = o.mean() 
        # o[( (o - o.mean()) / ( np.std(o) ) ) < (1-self.red_thresh) ] = 1 ### intereting visually but not necc info-wise
        # o[( (o - o.mean()) / ( np.std(o) ) ) < (1-self.red_thresh) ] = 1

        rrange = o.max() - o.min()
        om = o.min()  
        # print( o.min(), o.max(), o.mean(), rrange) 


        o[ ( ((o - om)/rrange) < (1-2.6*self.red_thresh) ) ] = .01 #and ( ((o - o.min())/rrange) <  (1-self.red_thresh) ) 
        o = o * self._get_channel_eq(img, 0)
        o[ ( ((o - om)/rrange) > (2.8*self.red_thresh) ) ] = 0
        o = o * self._get_channel_eq(img, 0)
        # o[ o != 1] = 
        # o = np.where( o < 1, o*10, o) 
        # o2 = o.copy()
        # o2[( (o2 - o2.mean()) / ( np.std(o2) ) ) < (1-self.red_thresh) ] = 0
        # o = np.logical_and(o, self._get_channel_eq(img, 0)     ) +  1/o2 

        # o =   (self._get_channel_eq(img, 0) - om)/rrange * (1-o) #(o - om)/rrange
        # return unsharp_mask(o, radius=5, amount=2) - self._get_channel_eq(img, 0)
        o = unsharp_mask(o, radius=5, amount=2) 

        ### 1. the bloody regions 
        # o1 = (o - self._get_channel_eq(img, 0)) 
        # o1 = o*(o - self._get_channel_eq(img, 0)) + o
        # o1 = (o*self._get_channel_eq(img, 0) ) + o 
        o1 = o*(o - self._get_channel_eq(img, 0)) + o ## adding back so uniform black bg outside the circle
        # o1 = o-o1 # nice looking version;like a kinder inversion; seems to highlight red-T2 equiv plus bloody areas 

        ### 2. capture % blur 
        ooin = self._get_channel_eq(img, 0) 
        ooto = gaussian(ooin, 2) #unsharp_mask(ooin, radius=2, amount=1) 
        o2 =  (ooin - ooto )   ## the blur that's been chucked 
        o2 = np.log( o2 / o2.sum() )
        
        return o1 , o2

    def remapped_data(self, img): 
        outiez = []
    
        # 1. geometric/structural info --- using green channel 
        outiez.append( self._lbp.remapped_data(img) )  

        # 2. Eigenz top 3 @ bloody/red lesions on green channel 
        # outiez.append( self._fitted_eigen.remapped_data( img )  )    
        # outiez.append( self._some_channel( img, 1  )  )  

        o1, o2 = self._darks_blend_but(img) 
        outiez.append( o1)    
        outiez.append( o2)    
        # outiez.append(self._darks_blend_but(img) )

        # print("Fitted_Eigen = ", outiez[1].shape , "_O len = ", len(outiez) ) 


        # 3. thresholded color channels 
        ## For Now Run RGB blue  and LAB red <<< TODO add switch 
        # outiez.append( self._red_channel_update_origi(  img ) )
        outiez.append( self._red_channel_update2(  img ) )

        outiez.append( self._blue_channel_update( img ) )   

        # append yellow for pigmentation
        o = self._clear_outter_circle( 
                self._get_yellow_from_rgb2lab(
                    self._get_color_eq(img), 
                    threshit=True, thresh= self.yellow_thresh ),
                b_thresh=0.85)   
        outiez.append(o) 

        
        # print("After Colorz: _O len = ", len(outiez) )      

        # 3.  append origi as cleaned only 
        if self.add_origi:
            outiez.append(self._get_color_eq(img) ) 


        # print("After add Origigz: _O len = ", len(outiez) )   
        # 4. combine color channelz
        o = np.dstack(outiez) 
        return o 

    @staticmethod 
    def channel_labels(add_origi=True):
        o = ["lbp", "darkbits", "blur", 
             # , "darkbits-o2",  "red-To", 
             "red-T2","blue-T", "yellow"] 
        o += ["origi"] if add_origi else []
        return o

