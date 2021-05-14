'''
author: bg
goal: 
type: Collection of Architectures and a Factory??
how: 
ref: 
refactors: design patterns - resuse, composition, decorator/wrapper 
'''

import numpy as np 
import torch 

import torch.nn as nn 
import torch.nn.functional as F  

import torch.optim as optim

from report import ZReporter 


class ZNNArchitecture(nn.Module):
    ## layers, weights, forward, 
    def __init__(self, n_outputs, 
                layers=None, 
                h_activation=F.relu, ## TODO: kwargz 
                o_activation=F.softmax, 
                weights_initor=(nn.init.xavier_uniform_, { 'gain':nn.init.calculate_gain('relu')}) ):  
        super(ZNNArchitecture, self).__init__() 
        self.n_outputs = n_outputs 
        self.h_activation = h_activation
        self.o_activation = o_activation
        self.weights_initor =  weights_initor 
        ## setup layers & init weights 
        self.modules_list = None 
        self.add_layers(layers)
    
    def add_layers(self, layers):        
        if layers is None:
            return 
        if self.modules_list is None:
            self.modules_list = nn.ModuleList( layers )
        else:
            _ = [self.modules_list.append(l) for l in layers] 
        # self.modules_list = [] 
        # for i, l in enumerate(layers):
        #     setattr(self, f"layer_{i+1}", l)
        #     self.modules_list.append( l )

        def init_weights(l):
            if type(l) == nn.Linear:
                self.weights_initor[0](l.weight, **self.weights_initor[1] )
                l.bias.data.fill_(0.01) 
        # for l in self.modules_list:
        #     init_weights(l) 
        if self.weights_initor is not None:
            self.apply( init_weights )

    def forward(self, x): ## TODO: layers = (nn.layer, activation, activation params)
        # print(type(x), x.shape) 
        n = len(self.modules_list)
        for i, m in enumerate(self.modules_list): 
            # print( f"{i+1}: {m}")
            x = m.forward(x)  #(x) ## These are layers so not . 
            ## 1. TODO: check if should have activation function or not 
            if i == (n - 1) and self.o_activation is not None: ##TODO: output kwrags
                x = self.o_activation(x, dim=1) 
            elif self.h_activation is not None:
                x = self.h_activation(x) 
            ## 2. TODO: check if should flatten or not 
        # print("AFTER.FWD: ", x.shape)
        return x 



class ZMLP(ZNNArchitecture):
    def __init__(self, n_inputs, n_outputs, 
                n_layers=3, n_hunits=64,  
                h_activation=F.relu, # TODO: kwargz 
                o_activation=F.softmax):
        super(ZMLP, self).__init__(n_outputs=n_outputs, 
                                    h_activation=h_activation, 
                                    o_activation=o_activation)   
        
        ## 1. Hidden Layers
        mm = []
        n_out =  min(n_hunits, n_inputs**2) 
        n_fout = (n_outputs//2)
        for i in range(n_layers - 1):
            n_in = n_inputs if i == 0 else n_out ## input layer width 
            l = nn.Linear(n_in, n_out )
            mm.append( l )  
        self.add_layers( mm )  
        #self.modules_list.append(nn.Linear(n_out, (n_out//2)))
        self.add_layers( [nn.Linear(n_out, n_fout),] )
        ## 2. output layer
        self.add_layers([nn.Linear(n_fout, n_outputs),] ) 
        print( f"===== {type(self.modules_list) }  ======")

###### =========================================================== 

class ZNNArchitectureFactory:
    __registry = {} ## TODO: instance Vs singletons+common_types reuse  

    @staticmethod
    def mlp(n_inputs, n_outputs, kwargz={}):
        #if __registry.get('mlp', None) is None:
        O_ = ZMLP(n_inputs, n_outputs, **kwargz) 
        return O_




if __name__ == "__main__":
    epochz = 3 
    N, nf, nclasses  =  12, 4, 2 #.reshape(1, -1)
    tmpX = [ torch.tensor( np.random.randint(0, 100, size=nf).reshape(1, -1).astype(np.float32) ) for i in range(N)]
    tmpY = [  x.sum()**2 for x in tmpX]
    ymu = np.array(tmpY).mean() 
    tmpY = [ torch.tensor( np.array([ int(y > ymu),] ).astype(np.long) ) for y in tmpY] ## TODO: qcut percentile
    print(tmpY)
    print( len(tmpX), tmpX[0].shape )
    mlp = ZNNArchitectureFactory.mlp(nf, nclasses) 
    print(mlp )
    # yhat = [mlp.forward(x) for x in tmpX]
    lossor = nn.CrossEntropyLoss()
    optimor = optim.SGD( mlp.parameters(), lr=0.001, momentum=0.9) 

    for e in range(epochz): 
        L_= 0.0
        for i, ( x_, y_ ) in enumerate( zip(tmpX, tmpY) ):
            optimor.zero_grad()
            yhat = mlp(x_) 
            # print(y_, " ===> ", yhat)
            loss_ = lossor( yhat, y_ ) 
            loss_.backward() 
            optimor.step() 
            L_ += loss_.item() 
        print( f"Epoch {e+1}: loss { L_ }")

    yhat__ = [mlp(x) for x in tmpX] #torch.zeros([len(tmpY),1]) 
    yhat__ = [ torch.max(p, 1)[-1] for p in yhat__]
    #print(">>>> ", len(yhat__), yhat__[0].shape )
    for i, y in enumerate(yhat__):
        y_ = tmpY[i]
        # ypred = torch.argmax(y) 
        # print( (i+1), ".  ", y_, " <===", ypred,  y.shape , "\t", y[0]) 
        print( (i+1), ".  ", y_, " <===", y ) 
    print( "ACC: ", np.array([ int(y==y_) for y, y_ in zip(tmpY, yhat__) ]).mean() )