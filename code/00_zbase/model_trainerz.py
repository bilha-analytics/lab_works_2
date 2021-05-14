'''
author: bg
goal: 
type: shared/utilz formodelz @@ TL, Common archs,  
how: 
ref: 
refactors: 
'''

from sys import builtin_module_names
from numpy.core.fromnumeric import squeeze
from tqdm import tqdm 

import numpy as np

from torch import nn
from torchvision import models 


    ### ===== 3. Pretraining Strategy et al TODO: Syncy up ZModel + ModelManager + ModelTrainer

    # def __init__(self, encoder_decoder_model=None ): 
    #     self.autoencoder = encoder_decoder_model
    
    # ## TODO: move train/predict code to some shared manager 
    # def train(self, X_data, epochz=3,
    #                         loss_func = nn.MSELoss(),
    #                         optimizer = optim.SGD, 
    #                         optimizer_paramz = {'lr':1e-3, 'momentum':.9} 
    #                         ):
    #     # 1. go into training mode + setup optimizer
    #     self.autoencoder.train() 
    #     optimizer = optimizer( self.autoencoder.parameters(), **optimizer_paramz )  

    #     # 2. run epochs and per item in batch 
    #     for e in tqdm( range(epochz ) ):
    #         running_loss = 0
    #         n_inst = 0
    #         for x in X_data:
    #             x_ = x#.float() 
    #             # a. zero the grads, compute loss, backprop 
    #             optimizer.zero_grad()
    #             o_ = self.autoencoder( x_ ) 
    #             loss = loss_func(o_, x_ ) 
    #             loss.backward()
    #             optimizer.step() 
    #             # b. update aggregates and reporting 
    #             running_loss += loss.item() 
            
    #         print(f"E {e}: loss {running_loss}") 

    # def predict(self, x): 
    #     x_ = x#.float() 
    #     # 1. go into eval mode
    #     self.autoencoder.eval() 
    #     # 2. run through model 
    #     o_ = self.autoencoder( x_ )
    #     # 3. extract top-k 
    #     O_ = []
    #     for rec in o_:
    #         O_.append( rec.detach()  )
        
    #     return torch.stack( O_ )  ## why yield??? 

    # def eval(self, ):
    #     pass 

        
