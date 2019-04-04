#params aaaaa
import numpy as np 
from copy import copy , deepcopy
from collections import Iterable, Collection
import pandas as pd
DEBUG = False
class Individual() :
    class KeyValueError(Exception) :
        def __init__(self, err_msg) :
            self.err_msg = err_msg
        def __str__(self) :
            str(self.err_msg)

    def __init__(self, *args, **kwargs) :
        if args is not None and isinstance(args[0], Individual) and len(args) == 1:
            tmp = args[0]
            self.chromosome = tmp.chromosome.copy()
            self.generation = tmp.generation
            self.age = tmp.age
            self.fitness = 0
        elif len(args) + len(kwargs) >= 3 :
            self.chromosome = args[0] if args[0] is not None else kwargs["chromosome"]
            self.generation = args[1] if args[1] is not None else kwargs["generation"]
            self.age = args[2] if args[2] is not None else kwargs["age"]
            self.fitness = 0
        else :
            raise Exception("non sufficient arguments")
    def mutate(self, t = 0.1, prob=None) :
        if prob is None :
            prob = np.random.rand(len(self.chromosome))
        
        factor = prob < t
        xor = lambda x, y : (x != y).astype(np.int32)

        chromosome = xor(np.array(self.chromosome), factor)
        generation = self.generation #+ 1
        age = 0
        
        self.grow()

        return Individual(chromosome.tolist(), generation, age)

    def grow(self) :
        self.age += 1
    
    def __setitem__(self,k,v) :
        if type(k) == int :
            self.chromosome[k] = v
        elif type(k) == slice :
            self.chromosome[k] = v
        elif isinstance(k, Collection) :
            s = pd.Series(self.chromosome)
            s[k] = v
            self.chromosome = s.values.tolist()

        else : 
            raise Individual.KeyValueError("Cannot set chromosome with a key type is not int, slice or Collection")


    def __getitem__(self,k) :
        if type(k) == int :
            return Individual(self.chromosome[k],self.generation,self.age)
        elif type(k) == slice :
            return Individual(self.chromosome[k],self.generation,self.age)
        elif isinstance(k, Collection) :
            s = pd.Series(self.chromosome)
            return Individual(s.values.tolist(), self.generation, self.age)
        
        else :
            raise Individual.KeyValueError("Cannot get chromosome with a key type is not int, slice or Collection")

    def __str__(self) :
        if DEBUG == True :
            return str([self.chromosome, self.generation, self.age])
        else :
            return str(self.chromosome)
    
    def __add__(self, another) :
        # implement concatenation of two chromosome so it is not mutable addition.
        chromosome = self.chromosome + another.chromosome
        generation = self.generation + 1
        age = 0
        
        self.grow()
        another.grow()

        return Individual(chromosome, generation, age)

    def __truediv__(self, div) :
        # div : a number or a np.ndarray
        chromosome = (np.array(self.chromosome) / div).astype(float).tolist()
        generation = self.generation 
        age        = self.age

        return Individual(chromosome, generation, age)
    def __mul__(self, mul) :
        # mul : a number or a np.ndarray
        chromosome = (np.array(self.chromosome) * mul).astype(float).tolist()
        generation = self.generation 
        age        = self.age

        return Individual(chromosome, generation, age)

    def indexOf(self, vals) :
        if isinstance(vals, Collection) :
            indices = []
            for v in vals :
                indices.append(self.chromosome.index(v))
            return indices
        else :
            indices = []
            for i in range(len(self.chromosome)) :
                if self.chromosome[i] == vals :
                    indices.append(i)
            return indices
    def indexOfPositive(self) :
        indices = []
        for i in range(len(self.chromosome)) :
            if self.chromosome[i] > 0 :
                indices.append(i)
        return indices

    def copy(self) :
        return deepcopy(self)

    def sum(self) :
        return sum(self.chromosome)

    def __len__(self) :
        return len(self.chromosome)

class WeightedIndividual(Individual) :
    def __init__(self, *args, **kwargs) :
        Individual.__init__(self, *args, **kwargs)
        self.cost = kwargs["cost"]
        # self.reweigh()

    def weights(self) :
        indices = self.indexOfPositive()
        w = [self.chromosome[i] for i in indices]
        # for i in indices :
        #     w.append(self.chromosome[i])
        return indices, np.array(w)

    def reweigh(self) :
        self.chromosome = (np.array(self.chromosome) / sum(self.chromosome) * self.cost).tolist()

    def mutate(self, t = 0.1, prob=None) :
        
        if prob is None :
           prob = np.random.normal(0, 1, len(self.chromosome))
        #    prob = (prob - np.mean(prob)) / (np.max(prob) - np.min(prob))
           prob = 0 + (1 - (0)) * (prob - np.min(prob)) / (np.max(prob) - np.min(prob))
        #    prob = np.random.rand(len(self.chromosome))
        
        factor = []
        for p in prob : 
            if abs(p) <= t : 
                factor.append(0)
            else :
                factor.append(p)
        factor = np.array(factor)
        # factor = prob if prob < t else 1
        action = lambda x, p : (1 - p ) * x + p * ( self.cost - x )
    

        chromosome = action(np.array(self.chromosome), factor)
        
        # Equal Rights: make every gene has the proportion of cost as they own the weights
        chromosome = (chromosome / sum(chromosome) * self.cost).tolist()
        generation = self.generation + 1
        age = 0
        
        self.grow()

        return WeightedIndividual(chromosome, generation, age, cost=self.cost)
    
    def __getitem__(self,k) :
        if type(k) == int :
            return WeightedIndividual(self.chromosome[k],self.generation,self.age, cost=self.cost)
        if type(k) == slice :
            return WeightedIndividual(self.chromosome[k],self.generation,self.age, cost=self.cost)
        elif isinstance(k, Collection) :
            s = pd.Series(self.chromosome)
            return WeightedIndividual(s.values.tolist(), self.generation, self.age)
        
        else :
            raise Individual.KeyValueError("Cannot get chromosome with a key type is not int, slice or Collection")

    def __add__(self, another) :
        # by default, chromosome is a list. In weighted individual , it is arithmetically added as vector(np.ndarray)
        chromosome = (np.array(self.chromosome) + np.array(another.chromosome)).tolist()
        generation = self.generation + 1
        age = 0
        
        self.grow()
        another.grow()

        return WeightedIndividual(chromosome, generation, age, cost=self.cost)

    def __truediv__(self, div) :
        # div : a number or a np.ndarray
        chromosome = (np.array(self.chromosome) / div).astype(float).tolist()
        generation = self.generation 
        age        = self.age

        return WeightedIndividual(chromosome, generation, age, cost=self.cost)
    def __mul__(self, mul) :
        # mul : a number or a np.ndarray
        chromosome = (np.array(self.chromosome) * mul).astype(float).tolist()
        generation = self.generation 
        age        = self.age

        return WeightedIndividual(chromosome, generation, age, cost=self.cost)


class Portfolio(Individual) :
    def __init__(self, *args, **kwargs) :
        Individual.__init__(self, *args, **kwargs)
        self.cost = kwargs["cost"]
        # self.total = kwargs["total"]

    def weights(self) :
        indices = self.indexOfPositive()
        w = []
        for i in indices :
            w.append(self.chromosome[i])
        return indices, np.array(w)
    
    def mutate(self, t = 0.1, alpha = 0.01, prob=None) :
        
        if prob is None :
            prob = np.random.rand(len(self.chromosome))
        
        factor = []
        for p in prob : 
            if p < t : 
                factor.append(p)
            else :
                factor.append(1)
        factor = np.array(factor)
        # factor = prob if prob < t else 1
        action = lambda x, p : (1 - alpha) * x * p + alpha * self.cost
    

        chromosome = action(np.array(self.chromosome), factor)
        chromosome = (chromosome / sum(chromosome) * self.cost).tolist()
        generation = self.generation #+ 1
        age = 0
        
        self.grow()

        return Portfolio(chromosome, generation, age, cost=self.cost)
    
    def __getitem__(self,k) :
        if type(k) == int :
            return Portfolio(self.chromosome[k],self.generation,self.age, cost=self.cost)
        if type(k) == slice :
            return Portfolio(self.chromosome[k],self.generation,self.age, cost=self.cost)

    def __add__(self, another) :
        chromosome = self.chromosome + another.chromosome
        generation = self.generation + 1
        age = 0
        
        self.grow()
        another.grow()

        return Portfolio(chromosome, generation, age, cost=self.cost)

    def __truediv__(self, div) :
        # div : a number or a np.ndarray
        chromosome = (np.array(self.chromosome) / div).astype(float).tolist()
        generation = self.generation 
        age        = self.age

        return Portfolio(chromosome, generation, age, cost=self.cost)
    def __mul__(self, mul) :
        # mul : a number or a np.ndarray
        chromosome = (np.array(self.chromosome) * mul).astype(float).tolist()
        generation = self.generation 
        age        = self.age

        return Portfolio(chromosome, generation, age, cost=self.cost)



if __name__ == '__main__' :
    # np.random.seed(1)
    ind = Portfolio([1,1,0,0,0],0,0,cost=100)
    ind1 = ind.copy()
    ind2 = ind.mutate()
    ind1[0] = 100
    ind3 = ind1 + ind2
    print("ind: {}".format(ind))
    print("ind1: {}".format(ind1))
    print("ind2: {}".format(ind2))
    print("ind3: {}".format(ind3))
    print("indexOf 1: {}".format(ind3.indexOfPositive()))
    print("ind3 / 2: {}".format(ind3/2))
    print("ind3 * 2: {}".format(ind3*2))