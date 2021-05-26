'''
author: bg
goal: bridge/resuse with Scikit or Pytorch pipelines, transformers, models etc
type: API,
how: interfaces/definitions, wrappers, builders 
ref: 
refactors: TODO: use data loaders 
'''
import abc 
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn import metrics as skmetrics 

class ZTransformableInterface(metaclass=abc.ABCMeta): 
    '''
    @input:     data list or array, X and y=None (supervised or unsupervised ) 
    @output:    list of preprocessed results 
    @actions:   fit, transform 
    TODO: requirements/asserts ???? 
    ''' 
    @classmethod
    def __subclasshook__(cls, subclass):
        return(
            hasattr(subclass, 'fit') and callable(subclass.fit) and
            hasattr(subclass, 'transform') and callable(subclass.transform)  
            or NotImplemented 
        )

    @abc.abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError 


    @abc.abstractmethod
    def transform(self, X, y=None):
        raise NotImplementedError 


class ZTrainableModel(metaclass=abc.ABCMeta):
    '''
    avail score method to gridsearch 
    @actions:   score
    TODO: requirements/asserts ???? 
    ''' 
    ## TODO: fix @ Abstract Vs Mixer Vs Interfeace def Vs ??? 
    # scoring_func = skmetrics.accuracy_score

    @classmethod
    def __subclasshook__(cls, subclass):
        return(
            hasattr(subclass, 'score') and callable(subclass.score) 
            or NotImplemented 
        )

    @abc.abstractmethod
    def score(self, yhat, y_):
        raise NotImplementedError 



class PreprocessorBuilder:
    '''
    @input:     a preprocessing transform,
    @output:    appropriate object type wrapper on input 
    @actions:   generate scikit or pytorch transform objects to wrap provided transformer     
    TODO: assertions 
    ''' 
    
    class ScikitTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, transformable ):
            self.transformable = transformable

        def fit(self, X, y=None):
            self.transformable.fit(X, y)
            return self 

        def transform(self, X, y=None):
            return self.transformable.transform(X, y) 

        def fit_transform(self, X, y=None):
            self.fit(X, y) 
            return self.transform(X, y)


    @staticmethod
    def scikit_transformer(transformable):
        '''
        wrap transformables in sckit object 
        '''
        return PreprocessorBuilder.ScikitTransformer(transformable) 



