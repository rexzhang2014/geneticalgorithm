import random
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from collections import Collection 
from .individual import *

class Fitness(ABC) :
    @abstractmethod
    def run(self, individual) :
        pass
        
class SumFitness(Fitness) :
    def run(self, x) :
        return x.sum()                                          

class HeadingFitness(Fitness) :
    def __init__(self, t=0) :
        self.t = t
    def run(self, individuals) :
        t = self.t
        return individuals[:t].sum() - individuals[t:].sum()

class ConstraintFitness(Fitness) :

    def __init__(self, fx, hx=None, gx=None, alpha=1, beta=1) :
        self.fx = fx
        self.hx = hx
        self.gx = gx
        self.alpha = alpha
        self.beta = beta

    def run(self, individual) :
        
        # the overall fitness
        fitness = self.fx(individual) 

        # add bonus to the fitness
        if isinstance(self.hx, Collection) :
            for h in self.hx :
                fitness += self.alpha * h(individual)

        # minus from the fitness
        if isinstance(self.gx, Collection) :
            for g in self.gx :
                fitness -= self.beta * g(individual)

        return fitness

class SumConstraintFitness(ConstraintFitness) :

    def __init__(self, total) :
        ConstraintFitness.__init__(self, fx=lambda x : x.sum() ** 2, hx=[lambda x : (x.sum()-total) ** 2] ) 
        self.total = total

class InSetFitness(ConstraintFitness) :
    def _fx(self, individual) :
        avail_set = []
        for a in self.avail :
            avail_set.extend(a)
        avail_set = set(avail_set)
        f = sum( [
            1.0 * int(individual[i] in self.c_avail[i]) for i in range(len(individual))
        ] + [
            0.1 * int(individual[i] in avail_set) for i in range(len(individual))
        ])
        return f
    def _hx(self, individual) :
        return len(set(individual.chromosome))

    def _gx(self, individual) :
        return len(set(individual.chromosome)) - len(individual.chromosome)


    def __init__(self, c_avail, avail, r_c=None) :
        self.c_avail = c_avail if c_avail else None
        self.avail   = avail if avail else None
        self.r_c = r_c if r_c else None
        ConstraintFitness.__init__(self, fx=self._fx, hx=[self._hx], gx=[self._gx], alpha=1, beta=0)
        





class PortfolioFitness(Fitness) :
    def __init__(self, data, alpha = 0.5 ) :
        # if type(data) != pd.DataFrame : 
        #     raise ArgumentTypeError("data must be pd.DataFrame")
        self.data       = data
        

        # self.revenue  = [np.mean(d) for d in data]
        self.revenue    = self.data.apply(func=lambda f: self.r_t(f).mean(), axis=1)

        # self.risk       = [np.std(d) for d in data]
        self.risk       = self.data.apply(func=lambda f: self.r_t(f).std(), axis=1)
        
        # self.reve_daily = [self.r_t(d) for d in data]
        self.reve_daily = self.data.apply(func=self.r_t,axis=1)
        
        self.alpha      = alpha

    def r_t(self, nav) :
        # nav  = pd.Series(nav)
        nav1 = nav.shift(1)
        r    = (nav - nav1) / (nav1 + 0.00001)
        # return r[1:].values # as shifted by 1 step, the first r_t is NA 
        return r[1:] # as shifted by 1 step, the first r_t is NA 

    
    def revenue_port(self, revenue, weights) :
        # revenue : a Series or list of funds' mean(expectation) revenue
        # weights : portfolio weights represented by individual's chromosome
        return np.matmul(revenue.T , weights)

    def risk_port(self, port, weights) :
        # port : Series of portfolio fund history
        # weights : portfolio weights represented by individual's chromosome(list)
        reve_port = self.reve_daily.loc[port.index]
        # reve_port = self.reve_daily.loc[port]
        risk_port = self.risk.loc[port.index]
        # risk_port = self.risk.loc[port]

        # print(revenue)
        cov_matrix = np.cov(reve_port)
        
        assert port.shape[0] == len(weights), "inconsistent shapes of port and weights"


        mat_sum = np.matmul(np.matmul(weights, cov_matrix), weights.T) 

        asset_sum = np.matmul(weights ** 2, risk_port ** 2) 

        risk_p = np.sqrt(asset_sum + mat_sum )
        
        return risk_p, cov_matrix

    
    def run(self, individual) :
        indices, weights = individual.weights()
        invest_funds     = self.data.iloc[indices,:]
        revenue_port     = self.revenue_port(self.revenue[indices], weights)
        risk_port, _   = self.risk_port(invest_funds, weights)
        fitness          = revenue_port * (1-self.alpha) - risk_port * self.alpha
        return fitness #, revenue_port, risk_port, cov

if __name__ == "__main__" :
   


    w = np.array([1,2,3])
    m = np.array([[1,2,3], [4,5,6], [7,8,9]])

    p = np.matmul(np.matmul(w,m),w.T)

    print(p)

    cov_sum = 0
    for i in range(3) :
        for j in range(3) :
            cov_sum += w[i] * w[j] * m[i,j]
    
    print(cov_sum)
   