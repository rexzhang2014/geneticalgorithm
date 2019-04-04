import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from Individual import Individual
from CrossOver import SinglePointCrossOver
from copy import deepcopy
from collections import Iterable
from Selector import LinearRankingSelector, LeadingSelector
from Fitness import HeadingFitness, SumConstraintFitness
from datetime import datetime
import logging 

class Species() :
    def __init__(self, individuals=[], selector=LinearRankingSelector(), crossover = SinglePointCrossOver(), fitness_func=HeadingFitness()) :
        self.individuals = pd.Series(individuals)
        self.crossover = crossover
        self.fitness_func = fitness_func
        self.selector = selector
        self.df = None
        self.fitness = None
        logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def calculate_fitness(self, func=None) :
        # in this step, fitness is re-calculated as individuals have changed. df is rebuilt as individuals and fitness have changed
        func = self.fitness_func.run if func is None else func 
        fitness = self.individuals.apply(func=func)
        self.fitness = fitness
        self.df = pd.concat([self.individuals, fitness],axis=1)
        self.df.columns = ["individuals", "fitness"]
        print("{} INFO: -- FITNESS -- top:{}; sum: {}; avg:{}; population:{}".format(datetime.now(), self.fitness.max(), self.fitness.sum(), self.fitness.mean(),self.population()))
        return pd.concat([self.individuals, fitness],axis=1)

    def select(self, func=None, verbose=True) :
        func = self.selector.select if func is None else func 
        self.df = func(self.df) #Select with df
        self.individuals = self.df["individuals"]
        self.fitness     = self.df["fitness"]
        if verbose :
            print("{} INFO: -- SELECTION -- top:{}; sum: {}; avg:{}; population:{}".format(datetime.now(), self.fitness.max(), self.fitness.sum(), self.fitness.mean(),self.population()))
        
            print(self.individuals.iloc[0])
        
        return self.df

    def feasible(self, func=None) :
        # at this step, only feasible individuals survive in feasibility constraints
        func = self.selector.feasible if func is None else func 
        self.individuals = func(self.individuals)
        # self.individuals = self.df["individuals"]
        # self.fitness     = self.df["fitness"]
        print("{} INFO: -- FEASIBLE -- top:{}; sum: {}; avg:{}; population:{}".format(datetime.now(), self.fitness.max(), self.fitness.sum(), self.fitness.mean(),self.population()))
        return self.individuals

    def reproduce(self, func=None, *args, **kwargs ) :
        if len(args) > 1 :
            t = args[1] 
        elif "t" in kwargs.keys() :
            t = kwargs["t"]
        else :
            t = 0.5
        
        # get the mutations generation. The mutations are also individuals can join crossover step
        mutations = self.individuals.apply(func = lambda x : x.mutate(t))
        self.individuals = self.individuals.append(pd.Series(mutations)).reset_index(drop=True)
        print("{} INFO: -- MUTATION -- population: {}; generation: {}".format(datetime.now(), self.population(), self.generations()))   
        # get the crossover generation
        func = self.crossover.run if func is None else func 
        offsprings = func(self.individuals)
        self.individuals = self.individuals.append(pd.Series(offsprings)).reset_index(drop=True)
        print("{} INFO: -- XOVER -- population: {}; generation: {}".format(datetime.now(), self.population(), self.generations()))
        return mutations.append(pd.Series(offsprings)).reset_index(drop=True)

    def sort(self, func=None) :
        return self.df.sort_values(by="fitness",ascending=False)
        # return self.df

    def max(self) :
        if self.fitness is None :
            raise Exception("fitness has not been calculated")
        return self.fitness.max()

    def sum(self) :
        if self.fitness is None :
            raise Exception("fitness has not been calculated")
        return self.fitness.sum()

    def size(self) :
        return len(self.individuals)
    
    def population(self) :
        return len(self.individuals)

    def generations(self) :
        return max(self.individuals.apply(func=lambda x : x.generation))
    
    def copy(self) :
        return deepcopy(self)

    def __str__(self) :
        return str(individuals)


        
class Environment():
    def __init__(self, individuals=[], selector=LinearRankingSelector(),crossover=SinglePointCrossOver(), fitness_func=HeadingFitness(),*args, **kwargs) :
        self.CAPACITY = kwargs["CAPACITY"] if "CAPACITY" in kwargs.keys() else 1000
        self.MAX_GENERATION = kwargs["MAX_GENERATION"] if "MAX_GENERATION" in kwargs.keys() else 1000
        self.MAX_ITERATION  = kwargs["MAX_ITERATION"] if "MAX_ITERATION" in kwargs.keys() else 1000
        self.mut_rate = kwargs["mut_rate"] if "mut_rate" in kwargs.keys() else 0.5
        self.epsilon = kwargs["epsilon"] if "epsilon" in kwargs.keys() else 1e-5
        # self.sel_rate = kwargs["sel_rate"] if "sel_rate" in kwargs.keys() else 0.5

        self.species = Species(individuals, selector, crossover, fitness_func)

    def __punish__(self) :
        pun = LeadingSelector(0.3,self.species.selector.constraints)
        self.species.select(func=pun.select,verbose=False)
        print("{} INFO: -- PUNISHMENT -- population: {}".format(datetime.now(), self.species.population()))

    def evolute(self) :
        last_fitness = -1e9
        for i in range(int(self.MAX_ITERATION)) :
            print("{} INFO: -- ITERATION START -- : {}".format(datetime.now(), i))
            
            # calculate fitness
            self.species.calculate_fitness()
            # select from the fit individuals
            self.species.select()
            
            # if the population is beyond environment's capability, the environment will punish the species, forcing the non-opitma disappeared.
            while self.species.population() >= self.CAPACITY :
                self.__punish__()
            
            # Stop evolution if the generation has been exceeded MAX_GENERATION
            if self.species.generations() > self.MAX_GENERATION :
                break
            
            # Reproduce next generation, including mutation and crossover
            self.species.reproduce(t=self.mut_rate)
            # Check or make the population be feasible under the pre-defined constraints
            self.species.feasible()
            
            # Stop evolution if total/maximum fitness converges
            # if self.species.sum() - last_fitness <= self.epsilon :
            #     break
            # else :
            #     last_fitness = self.species.sum()

            
        self.species.select()
    
    def evaluate(self) :
        # fitness, revenue, risk, cov = 
        return self.species.fitness_func.run(self.species.individuals)

if __name__ == '__main__' :
    DEBUG = False
    # np.random.seed(1)
    ind = Individual([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0)
    individuals = [Individual([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0],0,0),
                    Individual([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0],0,0),
                    Individual([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0],0,0),
                    Individual([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0],0,0),
                    Individual([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0],0,0),
                    Individual([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0],0,0),
                    Individual([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0],0,0),
                    Individual([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],0,0),
                    Individual([1,1,0,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0,0,0],0,0),]    
    # for i in range(10) :
    #     individuals += [ind.mutate(0.0)]
    for offs in individuals :
        print("offs: {}".format(offs))  

    # crossover = SinglePointCrossOver()
    #next_gen  = crossover.crossover([ind, ind1, ind2, ind3, ind4])

    # spc = Species([ind, ind1, ind2, ind3, ind4], crossover=crossover)
    constraints = [lambda ind : len(ind.indexOfPositive()) == 7]
    env = Environment(individuals,selector=LeadingSelector(0.5,constraints),crossover=SinglePointCrossOver(), fitness_func=HeadingFitness(5),MAX_GENERATION=50,CAPACITY=100, MAX_ITERATION=100)

    env.evolute()

    print(env.species.population(), env.species.generations())
    for i in env.species.individuals :
        DEBUG=True
        print(i)



