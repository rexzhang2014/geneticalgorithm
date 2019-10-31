import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import Iterable


class Selector(ABC):
    def __init__(self, constraints=None) :
        self.constraints = [lambda x : True] if constraints is None else constraints
    def __feasible__(self, individual) :
        for cons in self.constraints : 
            if cons(individual) == False :
                return False
        return True

    @abstractmethod
    def select(self, df):
        pass
    
    def feasible(self, individuals) :
        if self.constraints is not None :
            feasible  = individuals.apply(func=lambda x : self.__feasible__(x))
        return individuals[feasible[feasible==True].index]


class LeadingSelector(Selector) : 
    def __init__(self, ratio=0.5, constraints=None) :

        Selector.__init__(self,constraints)
        self.ratio = ratio

    
    def select(self, df) :
        sorted_df = df.sort_values(by="fitness",ascending=False)
        n_sel     = int(np.ceil(sorted_df.shape[0]*self.ratio))
        selection = sorted_df.iloc[:n_sel if n_sel > 1 else 2]
        return selection

class LinearRankingSelector(Selector) : 
    def __init__(self, constraints=None) :

        Selector.__init__(self,constraints)

    def select(self, df) :
        
        rank        = df["fitness"].rank(method='first',ascending=True)
        rank[rank==max(rank)-1]  = max(rank)
        # n           = 1 / len(rank)
        prob        = 0.3 + (1 - 0.3) * (rank - 1) / (rank.max() - 1)
        
        # print(prob[prob<0]) 
        
        prob_bool   = prob.apply(func=lambda x : np.random.choice([0,1],1,p=[1-x, x]).astype(bool)).transpose()
        
        selection   = df.loc[prob_bool[prob_bool==True].index, :]
        return selection


class DenyLinearRankingSelector(LinearRankingSelector) : 
    def __init__(self, n_deny = 1) :

        Selector.__init__(self,constraints=[lambda ind : len(ind.indexOfPositive()) == len(ind) - n_deny])
        self.n_deny = n_deny

    def feasible(self, individuals) :
        # if the individual do not deny anything, make the lowest N weights to 0
        for i in individuals :
            r = pd.Series(i.chromosome).rank(method='first',ascending=True)
            i[r[r <= self.n_deny]] = 0
            i.reweigh()

        
        if self.constraints is not None :
            feasible  = individuals.apply(func=lambda x : self.__feasible__(x))
        
        
        return individuals[feasible[feasible==True].index]

