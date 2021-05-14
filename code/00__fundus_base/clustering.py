'''
author: bg
goal: 
type: Image Clustering DL learn <-- VGG Auto-encoder (AE) + 
how: DCNN clustering - Local Aggregation by Zhuang et al (2019)   + SegNet method of AE arch
ref: https://towardsdatascience.com/image-clustering-implementation-with-pytorch-587af1d14123  
refactors: 
'''

import numpy as np 

import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine as cosine_distance 

from tqdm import tqdm 

import preprocess, utilz 
from skimage import img_as_uint

### ============= 1. AutoEncoder ============
class EncoderVGG( nn.Module ): 
    ## VGG16 based
    channels_in = 3
    channels_code = 512

    def __init__(self, pretrained=True, n_channelz = 3, code_channels=512):
        super(EncoderVGG, self).__init__() 

        self.channels_in = n_channelz
        self.channels_code = code_channels 

        ## setup vgg encoder - chuck classifier and avg pool layers < only keep feature extraction layers 
        vgg = models.vgg16_bn(pretrained=pretrained) 
        del vgg.classifier
        del vgg.avgpool 

        self.encoder = self._encodify(vgg )

    def _encodify(self, encoder):
        ## adjust: avail pooling indices from encoder.max_pool to the decoder unpooling layers 
        ## the models.vgg16_bn does not generate the indices --> so reinitialilze them so tha they can do that

        modulez = nn.ModuleList()
        for mod in encoder.features:
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
        return modulez 
    
    def forward(self, x):
        ## forward pass 
        pool_indicies = [] ## to be passed to decoder for unpooling 
        x_current = x 
        for mod_encode in self.encoder:
            outie = mod_encode( x_current )
            # pooling layers return two outputs; 2nd is the indices 
            if isinstance(outie, tuple) and len(outie) == 2:
                x_current, idx = outie
                pool_indicies.append( idx )
            else:
                x_current = outie 
        return x_current, pool_indicies 

class DecoderVGG(nn.Module):
    ## sorta transposed version of the VGG16 network => looks like the encoder in reverse but not strictly so
    channels_in = EncoderVGG.channels_code
    channels_out = EncoderVGG.channels_in  
    def __init__(self, encoder):
        super(DecoderVGG, self).__init__()
        self.decoder = self._invert(encoder) 
    
    def _invert(self, encoder):
        ## decoder as a somewhat mirror of encoder
        ## BUT/AND: 1, 2D transpose convolution + 2. 2D unpooling
        ## 1. 2D transpose convolution + batch norm + activation 
        ##    convert encoder.conv to decoder.transposed conv
        ## 2. 2d unpool : conver encoder.pool to decoder.unpool 
        modulez = []
        for mod in reversed( encoder ):
            if isinstance(mod, nn.Conv2d):
                kwargz = {'in_channels':mod.out_channels,
                            'out_channels':mod.in_channels,
                            'kernel_size':mod.kernel_size, 
                            'stride': mod.stride,
                            'padding':mod.padding } 
                mod_trans = nn.ConvTranspose2d( **kwargz ) 
                mod_norm = nn.BatchNorm2d( mod.in_channels ) 
                mod_act = nn.ReLU(inplace=True) 
                modulez += [mod_trans, mod_norm, mod_act ] 
            elif isinstance(mod, nn.MaxPool2d):
                kwargz = {'kernel_size': mod.kernel_size,
                            'stride':mod.stride,
                            'padding':mod.padding}
                modulez.append( nn.MaxUnpool2d(**kwargz)  )
        ## drop last norm and activation so that final output is from a conv with bias 
        modulez = modulez[:-2]

        return nn.ModuleList( modulez )

    def forward(self, x, pool_indices ):
        ## x is a tensor from encoder and pool_indices is the list from encoder
        x_current = x
        k_pool = 0
        rev_pool_indices = list(reversed(pool_indices)) 
        for mod in self.decoder:
            ## if @ unpooling make use of the indices for that layer
            if isinstance(mod, nn.MaxUnpool2d):
                x_current = mod(x_current, indices=rev_pool_indices[k_pool]) 
                k_pool += 1
            else:
                x_current = mod(x_current)
        return x_current 

class AutoEncoderVGG(nn.Module): ## now combine the encoder and decoder 
    channels_in = EncoderVGG.channels_in
    channels_out = DecoderVGG.channels_out
    channels_code = EncoderVGG.channels_code 

    def __init__(self, pretrained=True, n_channelz=3,out_size=512):
        super(AutoEncoderVGG, self).__init__( ) 
        self.encoder = EncoderVGG(pretrained=pretrained, n_channelz=n_channelz, code_channels=out_size)
        self.decoder = DecoderVGG(self.encoder.encoder) 
        print("Setup AE")

    def forward(self, x):
        x_, idx = self.encoder(x) 
        x_ = self.decoder(x_, idx) 
        return x_ 

    def get_params(self):
        return list(self.encoder.parameters() ) + list(self.decoder.parameters() )

''' NOTES:
    - Use MSE to quantify diff --> nn.MSELoss as objective fx 

''' 
def train_autoencoder(model, X_data, n_epoch=3):
    loss_fx = nn.MSELoss() 
    o_k = {'lr':0.001, 'momentum':0.9} 
    # paramz = model.encoder.parameters() + model.decoder.parameters()
    paramz = model.get_params() 
    optimizer = torch.optim.SGD( paramz, **o_k) 
    # 1. train model
    model.train()
    for epoch in tqdm( range(n_epoch) ): 
        running_loss = 0
        n_inst = 0
        for x in X_data:
            # zero the grads > compute loss > backprop
            optimizer.zero_grad() 
            outie = model(x.float() ) 
            loss = loss_fx(outie, x.float() ) 
            loss.backward()
            optimizer.step() 
            # update aggregates and reporting 
            running_loss += loss.item() 
        #running_loss = running_loss/batch_size
        print(f"E {epoch}: loss {running_loss}") 

def predict_autoencoder(model, x):
    model.eval() 
    outie = model(x.float() ) 
    O_ = []
    ## what is this bit doing ?? 
    for img in outie:
        img = img.detach() 
        O_.append( img )

    yield torch.stack( O_  ) 


### ============= 2. Clustering ============
''' NOtes on clustering - Local Aggregation Loss method (Zhuang et al, 2019, arXiv:1903.12355v2)
 
- Images with similar xtics will have small L2 on encoded values, but encoded values are nonlinear --> not large deviations inter-cluster
- AutoEncoder == compress form high D to low D
- Now learn 'fundusness' and also values easily 'clusterable'

Local Aggregation Loss Method
- entropy based cost/objective function for clusters <-- p(cluster membership)
- TODO: review algz again 
- implement custom loss function 
- Memory Bank - arXiv:1903.12355v2
    - a way to deal with fact that the gradient of the LA obj func depends on the gradiens of all codes of the dataset
    - efficiently computing gradients during backprop << the tangled gradients of the codes w /r/t decoder params must be computed regardless 
        - b/c clustering @ comparing each element to all other elements in the dataset thus entablement 
    - memory bank trick is to treat other codes, other than those in minibatch/current, as constants. 
        - entanglement with derivatives of other codes thus goes away
        - as long as approximated gradients are good enough to guide optimization towards a minimum, it is good
'''
class MemoryBank:
    def __init__(self, n_vecs, dim_vecs, memory_mixing_rate):
        self.dim_vecs = dim_vecs
        self.vecs = np.array([ marsaglia(dim_vecs) for _ in range(n_vecs)])
        self.memory_mixing_rate = memory_mixing_rate
        self.mask_init = np.array([False]*n_vecs)

    def update_memory(self, vectors, index):
        if isinstance(index, int):
            self.vecs[index] = self._update(vectors, self.vecs[index])
        elif isinstance(index, np.ndarray):
            for idx, vev in zip(index, vectors):
                self.vecs[idx] = self._update(vec, self.vecs[idx] ) 

    def mask(self, inds_int):
        outie = []
        for r in inds_int:
            row_mask = np.full(self.vecs.shape[0], False) 
            row_mask[ r.astype(int) ] = True 
            outie.append( row_mask ) 
        return np.array( outie ) 

    def _update(self, vec_new, vec_recall):
        return vec_new * self.memory_mixing_rate + vec_recall * (1. - self.memory_mixing_rate)


class LocalAggregationLoss(nn.Module):
    def __init__(self, temperature, knns, 
                clustering_repeats, n_centroids, 
                memory_bank, kmeans_n_init=1,
                nn_metric=cosine_distance, nn_metric_params={} ):
        super(LocalAggregationLoss, self).__init__() 

        self.temperature = temperature 
        self.memory_bank = memory_bank 

        ## 1. Distance: Efficiently compute nearest neighbors << set B in alg 
        self.neighbour_finder = NearestNeighbors(n_neighbors=knns+1,
                                                algorithm='ball_tree',
                                                metric=nn_metric,
                                                metric_params=nn_metric_params)
        ## 2. Clusters: efficiently compute clusters << set C ini alg 
        self.clusterer = []
        for k_clusterer in range(clustering_repeats):
            self.clusterer.append(
                KMeans(n_clusters=n_centroids,
                        init='random',
                        n_init=kmeans_n_init)
            )

    def forward(self, codes, indices):
        assert codes.shape[0] == len(indices) 
        codes = codes.type( torch.DoubleTensor )
        code_data = normalize( codes.detach().numpy(), axis=1)

        ##constants in the loss function; no gradients@backpass
        self.memory_bank.update_memory(code_data, indices) 

        bg_neighbours = self._nearest_neighbours(code_data, indices)
        close_neighbours = self._close_grouper(indices) 
        neighbour_inersect = self._intersecter(bg_neighbours, close_neighbours) 
        
        ## compute pdf
        v = F.normalize(codes, p=2, dim=1) 
        d1 = self._prob_density(v, bg_neighbours)
        d2 = self._prob_density(v, neighbour_inersect)

        return torch.sum(torch.log(d1) - torch.log(d2))/codes.shape[0] 

    def _nearest_neighbours(self, codes_data, indices):
        self.neighbour_finder.fit(self.memory_bank.vectors )
        indices_nearest = self.neighbour_finder.kneighbours(codes_data, return_distance=False)
        return self.memory_bank.mask( indices_nearest )

    def _close_grouper(self, indices):
        ## ascertain
        memberships = [[]]*len(indices) 

        for clusterer in self.clusterer:
            clusterer.fit( self.memory_bank.vectors ) 
            for k_idx, cluster_idx in enumerate(clusterer.labels_[indices]) :
                other_members = np.where( clusterer.labels_ == cluster_idx)[0] 
                other_members_union = np.union1d(memberships[k_idx], other_members)
                memberships[k_idx] = other_members_union.astype(int) 
        return self.memory_bank.mask( np.array(memberships, dtype=object ) ) 

    def _intersecter(self, n1, n2):
        return np.array([
                        [v1 and v2 for v1, v2 in zip(n1_x, n2_x)] 
                            for n1_x, n2_x in zip(n1, n2 ) ])


    def _prob_density(self, codes, indices):
        ## unormalized differentiable probability densities 
        ragged = len(set([np.count_nonzero(idx) for idx in indices ] )) != 1 
        
        # In case the subsets of memory vectors are all of the same size, broadcasting can be used and the
        # batch dimension is handled concisely. This will always be true for the k-nearest neighbour density
        if not ragged:
            vals = torch.tensor([np.compress(ind, self.memory_bank.vectors, axis=0) for ind in indices],
                                requires_grad=False)
            v_dots = torch.matmul(vals, codes.unsqueeze(-1))
            exp_values = torch.exp(torch.div(v_dots, self.temperature))
            pdensity = torch.sum(exp_values, dim=1).squeeze(-1)

        # Broadcasting not possible if the subsets of memory vectors are of different size, so then manually loop
        # over the batch dimension and stack results
        else:
            xx_container = []
            for k_item in range(codes.size(0)):
                vals = torch.tensor(np.compress(indices[k_item], self.memory_bank.vectors, axis=0),
                                    requires_grad=False)
                v_dots_prime = torch.mv(vals, codes[k_item])
                exp_values_prime = torch.exp(torch.div(v_dots_prime, self.temperature))
                xx_prime = torch.sum(exp_values_prime, dim=0)
                xx_container.append(xx_prime)
            pdensity = torch.stack(xx_container, dim=0)

        return pdensity


''' Combining Envoder and LALoss 
''' 
def combine_run_LAClustering(X_data, merger_type='mean', n_vecs=5400, knns=500, n_centroids=600,n_epochs=3):
    model = EncoderVGGMerged(merger_type=merger_type)
    memory_bank = MemoryBank(n_vecs=n_vecs, dim_vecs=model.channels_code, 
                            memory_mixing_rate=0.5)
    memory_bank.vecs = normalize( model.eval_codes_for_(X_data), axis=1)
    loss_fx = LocalAggregationLoss(memory_bank=memory_bank,
                                    temperature=0.07, 
                                    knns = knns, 
                                    clustering_repeats=6, 
                                    n_centroids=n_centroids)

    
    o_k = {'lr':0.001, 'momentum':0.9} 
    paramz = model.get_params() ## parameters 
    optimizer = torch.optim.SGD( paramz, **o_k) 

     # 1. train model
    model.train()
    for epoch in tqdm( range(n_epoch) ): 
        running_loss = 0
        n_inst = 0
        for x in X_data:
            # zero the grads > compute loss > backprop
            optimizer.zero_grad() 
            outie = model(x.float() ) 
            loss = loss_fx(outie, x.float() ) 
            loss.backward()
            optimizer.step() 
            # update aggregates and reporting 
            running_loss += loss.item() 
        #running_loss = running_loss/batch_size
        print(f"E {epoch}: loss {running_loss}") 


class EncoderVGGMerged(EncoderVGG):
    def __init__(self, merger_type='mean', pretrained=True ):
        super(EncoderVGGMerged, self).__init__(pretrained=pretrained)

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

        

## -==========================
## 1. 
## 2. https://medium.com/intelligentmachines/implementation-of-class-activation-map-cam-with-pytorch-c32f7e414923
if __name__ == "__main__":
    print("****** STARTING ******")
    
    img_dim = (1, 3,224, 224)
    fetch_image = lambda x: utilz.Image.fetch_and_resize_image(x, img_dim).astype('f') 

    # cnn = AutoEncoderVGG(pretrained=True)
    # print(cnn) 

    # X_data = [ torch.tensor(np.random.rand( *img_dim ).astype('f') ) for i in range(5) ]
    fdir = "/mnt/externz/zRepoz/datasets/fundus/stare/"
    filez = [ f'{fdir}/im0012.ppm', f"{fdir}/im0232.ppm"] 
    _IMAGEZ = [fetch_image(x) for x in filez]

    # X_data = [torch.tensor( x ) for x in _IMAGEZ ]
    # train_autoencoder(cnn, X_data)


    # x_pred = torch.tensor( np.random.rand( *img_dim ).astype('f')  )
    # pred = list(predict_autoencoder(cnn, x_pred)) 

    # print( len(pred) , type(pred[0])) 

    # reformat_output_img = lambda x: x.reshape(224,224,3) # img_as_uint( ) 

    # utilz.Image.plot_images_list( [ reformat_output_img(x) for x in X_data], nc=len(X_data) )
    
    # utilz.Image.plot_images_list([ reformat_output_img(x) for x in pred], nc=len(pred) )
    # print("****** FIN ******")



    print("****** KMEANS *****")
    import cv2 
    import os 
    from glob import glob
    import matplotlib.pyplot as plt

    # img_dim = (224, 224, 3)
    # img_dim = (32, 32, 3)
    # def fetch_image(x):
    #     # utilz.Image.fetch_and_resize_image(x, img_dim).astype('f')
    #     o = cv2.imread( x ) 
    #     o = cv2.cvtColor(o, cv2.COLOR_BGR2RGB)
    #     o = np.float32( o.reshape((-1, 3)) )
    #     return o

    # fdir = "/mnt/externz/zRepoz/datasets/fundus/stare/"
    # filez = [ f'{fdir}/im0012.ppm', f"{fdir}/im0232.ppm"] ## glob( f"{fdir}/*.ppm")
    # #flattent into 3 color values per pixed 2D array
    # _IMAGEZ = [fetch_image(x) for x in filez]

    np.random.seed(1234) 

    n_clusters = 4    
    _N = 10000
    _FILEZ = []
    i = 0
    for f in glob( f"{fdir}/*.ppm"):
        _FILEZ.append( f )
        i+=1
        if i >= _N:
            break 
       
         
    fetch_name = lambda x: (os.path.basename(x).split(".")[0] , np.random.randint(0, n_clusters) )

    # X_data = np.array( [ fetch_image(x).reshape( (-1, 3) ) for x in _FILEZ  ] ) ##_IMAGEZ # 
    X_data_fnames = np.array( [ ((i%10 + i//33),*fetch_name(x)) for i, x in enumerate(_FILEZ)  ] ) ##_IMAGEZ # 
 
    ## ==== Open CV K-Means ====
    # stopping_criteria =  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # _, labels, (centers ) = cv2.kmeans(X_data, n_clusters, None, 
    #                                     stopping_criteria, 10, cv2.KMEANS_RANDOM_CENTERS )

    # centers = np.uint8( centers )
    # labels =labels.flatten()

    # ## show clusters
    # def fetch_cluster(i): 
    #     # dat = np.array([x[0] for x in X_data_fnames] )
    #     # return dat[ dat == i ]
    #     o_ = []
    #     for d in X_data_fnames:
    #         if int(d[2]) == i:
    #             o_.append( (d[0], d[2]) ) 
    #     # print( f"cluster {i}: {o_}")
    #     return np.array(o_)

    # # clusters = [ X_data_fnames[0][labels == i] for i in range(n_clusters ) ] 
    # clusters = [fetch_cluster(i) for i in range(n_clusters ) ] 
    # colorz = ('r', 'b', 'g', 'black')
    # numeric = lambda X : [ int(x) for x in X]
    # for clust, colr in zip(clusters, colorz[:len(clusters) ] ):
    #     plt.scatter( numeric(clust[0]), numeric(clust[1]), c=colr) 
    # plt.show()

    
    # print( centers.shape , labels.shape )  
    # # print(labels)
    # ## show centers
    # img_centers = [x.reshape(img_dim) for x in centers]
    # utilz.Image.plot_images_list( img_centers, nc=len(img_centers))


    # ## segmenting using the centroids
    # ## covert all pixels to the color of the centroids 
    # segmented_imagez = [ c[l].reshape( img_dim ) for c, l in zip(centers, labels)] 
    # utilz.Image.plot_images_list( segmented_imagez, nc=len(segmented_imagez))


    ### ==== K-NN clutering

    print("****** K-NN <<<< is supervised *****")
    # from sklearn.neighbors import KNeighborsClassifier 
    # X_data = np.array( [ fetch_image(x).flatten() for x in _FILEZ  ] ) 

    # import cv2
    # def extract_color_hist(img, bins=(8,8,8)):
    #     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #     hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0, 180, 0, 256, 0, 256])
    #     cv2.normalize(hist, hist) ## hist = cv2.normalize(hist)
    #     return hist.flatten()
    # X_histz = [extract_color_hist(x.reshape(img_dim)) for x in X_data]

    # model = KNeighborsClassifier(n_neighbors=n_clusters  )
    # model.fit(X_data)
    # # print(">>>>ACCURACY**: ", model.score())


    # model = KNeighborsClassifier(n_neighbors=n_clusters  )
    # model.fit(X_histz)
    


     # X_data = [torch.tensor( x ) for x in _IMAGEZ ]
    # train_autoencoder(cnn, X_data)


    print("****** K-MEANS on VGG encoded data *****") 
    reformat_output_img = lambda x: x.reshape(224,224,3) # img_as_uint( ) 
    
    X_data = [torch.tensor( fetch_image(x) ) for x in _FILEZ  ]
    print("1. Data Loaded")

    cnn = AutoEncoderVGG(pretrained=True)    
    train_autoencoder(cnn, X_data)
    print("2. Encoder trained")

    X_encoded = [ list(predict_autoencoder(cnn, x)) for x in X_data]  
    print( len(X_encoded) , type(X_encoded[0])) 
    print("3. data encoded")

    # print(X_encoded[0])

    ## ==== Open CV K-Means ====
    X_data2 = np.array( [ np.dstack(x).reshape(-1, 3) for x in X_encoded  ] ) ##_IMAGEZ # [0].reshape( -1, 3 ) 
    print("4. Reshaped for cv2.Kmeans")

    stopping_criteria =  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers ) = cv2.kmeans(X_data2, n_clusters, None, 
                                        stopping_criteria, 10, cv2.KMEANS_RANDOM_CENTERS )

    centers = np.uint8( centers )
    labels =labels.flatten()
    print("5. K-Means done!")

    ## show clusters
    def fetch_cluster(i): 
        # dat = np.array([x[0] for x in X_data_fnames] )
        # return dat[ dat == i ]
        o_ = []
        for d in X_data_fnames:
            if int(d[2]) == i:
                o_.append( (d[0], d[2]) ) 
        # print( f"cluster {i}: {o_}")
        return np.array(o_)

    # clusters = [ X_data_fnames[0][labels == i] for i in range(n_clusters ) ] 
    clusters = [fetch_cluster(i) for i in range(n_clusters ) ] 
    colorz = ('r', 'b', 'g', 'black')
    numeric = lambda X : [ int(x) for x in X]
    for clust, colr in zip(clusters, colorz[:len(clusters) ] ):
        plt.scatter( numeric(clust[0]), numeric(clust[1]), c=colr) 
    plt.show()

    
    print( centers.shape , labels.shape )  
    # print(labels)
    ## show centers
    img_centers = [x.reshape(img_dim) for x in centers]
    utilz.Image.plot_images_list( img_centers, nc=len(img_centers))

    ## segmenting using the centroids
    ## covert all pixels to the color of the centroids 
    segmented_imagez = [ c[l].reshape( img_dim ) for c, l in zip(centers, labels)] 
    utilz.Image.plot_images_list( segmented_imagez, nc=len(segmented_imagez))



    # utilz.Image.plot_images_list( [ reformat_output_img(x) for x in X_data], nc=len(X_data) )
    
    # utilz.Image.plot_images_list([ reformat_output_img(x) for x in pred], nc=len(pred) )
    # print("****** FIN ******")