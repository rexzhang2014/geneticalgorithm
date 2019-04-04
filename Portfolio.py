from Environment import Environment
from Individual import Portfolio, WeightedIndividual
from Selector import LeadingSelector, LinearRankingSelector, DenyLinearRankingSelector
from Fitness import HeadingFitness, PortfolioFitness
from  CrossOver import SinglePointCrossOver, ArithmeticCrossOver
from FundHistory import loadFundHistory
import json, os
import pandas as pd 
import numpy as np


if __name__ == "__main__" :
    ROOT = os.path.join( os.getcwd(), "data")
    patterns = ["[0-9]{6}.json"]
    fcpatten = "[0-9]{6}"
    dataset = loadFundHistory(ROOT, patterns, pattern=fcpatten)
    dataset = dataset.loc[:,["fundcode","DWJZ"]]
    # for i in np.unique(dataset["fundcode"]) :
    #     ds = dataset.loc[dataset["fundcode"] == i,"DWJZ"]


    # dataset = dataset.groupby(by="fundcode").apply(lambda x: x["DWJZ"].astype(float))
    train, test, revenue, funds = [], [], [], []
    for f in np.unique(dataset["fundcode"]) :
        nav = dataset.loc[dataset["fundcode"]==f, "DWJZ"]
        if len(nav) > 500 :
            nav    = nav.apply(lambda x : 0 if x == '' else x.replace(",","")).astype(float)
            train.append(nav.values[:400])
            test.append(nav.values[400:])
            revenue.append(nav.values[len(nav)-1] - nav.values[399])
            funds.append(f)



    train_df = pd.DataFrame(train)

    # create random initials
    rnd = np.random.uniform(0,100,len(train)) 
    cost = 100

    rnd = rnd / np.sum(rnd) * cost

    ind = WeightedIndividual(rnd.tolist(),0,0,cost=cost)
    


    individuals = [ ind.mutate(0.8) for i in range(20) ]    
        
    for offs in individuals :
        print("offs: {}".format(offs))  

   
    # config and create environment
    # constraints=[lambda ind : len(ind.indexOfPositive()) == 3]
    env = Environment(individuals,selector=DenyLinearRankingSelector(train_df.shape[0]-5),crossover=ArithmeticCrossOver(), fitness_func=PortfolioFitness(train_df),MAX_GENERATION=2,CAPACITY=20, MAX_ITERATION=150, )

    # do evolutional optimization
    env.evolute()


    output = pd.DataFrame()
    for i in env.species.individuals :
        DEBUG=True
        output = pd.concat([output, pd.DataFrame(i.chromosome)], axis=1)
        print(i)
        print(np.matmul(np.array(i.chromosome), np.array(revenue)))

    print(env.species.population(), env.species.generations())
    stats = pd.DataFrame([funds, env.species.fitness_func.revenue.values.tolist(), env.species.fitness_func.risk.values.tolist()]).T
    stats.columns=['funds','revenue mean','risk std']
    # print("revenue is {}".format(env.species.fitness_func.revenue))
    # print("risk is {}".format(env.species.fitness_func.risk))
    print(stats)
    print("actual GL is {}".format(revenue))
 
    # print("Evaluate:--------------------------------------------------------")
    # fitness, revenue, risk, cov = env.evaluate()
    # print("revenue is {}".format(revenue))
    # print("risk is {}".format(risk))
    # print("cov is {}".format(cov))
    stats.to_csv("output/stats.csv")
    output.to_csv("output/individuals.csv")