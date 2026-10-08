import time
import os
from model import WECswarm

import concurrent.futures
from functools import partial

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def parallel_results():

    # Costruisci una lista di tutti i parametri da combinare
    params = [
        (seed, alg, sep)
        for alg in ['Static', 'Dynamic', 'Gaussian Process']
        for sep in ['Step', 'Probabilistic']
        for seed in range(0, 3)
    ]

    rows = []

    # Esecuzione parallela
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, seed, alg, sep) for seed, alg, sep in params]

        for f in concurrent.futures.as_completed(futures):
            try:
                result = f.result()
                rows.append(result)
            except Exception as e:
                print(f"Errore nella simulazione: {e}")

    return

def results():
    rows = []
    for alg in ['Static', 'Dynamic', 'Gaussian Process']:
        for sep in ['Step', 'Probabilistic']:
            for seed in range(0, 50):
                model = WECswarm(population_size=40,
                                speed=0.1,
                                vision=25,
                                separation=3,
                                load=1,
                                seed=seed,
                                info_sep = sep,
                                type=alg,
                                save=True
                                )
                model.run_model()
                results = create_df(seed=seed, alg=alg,sep=sep, results=model.results)
                rows.append(results)
                
                
                #create_df(seed=seed, type=alg, sep=sep, results=model.results.iloc[100:])
    df = pd.concat(rows, ignore_index=True)
    return df

def create_df(seed, alg, sep, results):
    results = results.iloc[100:].copy()
    results.loc[:, 'alg'] = alg
    results.loc[:, 'sep'] = sep
    results.loc[:, 'seed'] = seed
    results.loc[:, 'mean efficacy'] = results['efficacy'].mean()
    results.loc[:, 'mean ocean power'] = results['mean energy available'].mean()


    #print(model.datacollector.model_reporters)
    print(seed,':  ', alg, '-', sep, '-')
    #plt.plot(model.datacollector.model_reporters["total_energy_harvested"])
    return results

def analysis():

    return

def plot_resume(df):
    sns.set_theme(style="ticks", palette="pastel")

    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x='alg', y='efficacy', hue='sep')
    plt.title('Efficacy distribution for algorithm and separation type')
    plt.ylabel('efficacy')
    plt.xlabel('algorithm')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    sns.scatterplot(
        data=df,
        x='mean ocean power',
        y='mean efficacy',
        hue='alg',
        style='sep',
        alpha=0.7
    )
    plt.title('Efficacy distribution over energy available')
    plt.ylabel('mean efficacy')
    plt.xlabel('mean energy available')
    plt.legend(title='Separazione')
    plt.show()
    return

def run_simulation(seed, alg, sep):
    model = WECswarm(
        population_size=40,
        speed=0.1,
        vision=25,
        separation=3,
        load=1,
        seed=seed,
        info_sep=sep,
        type=alg,
        save=True
    )
    model.run_model()
    return create_df(seed=seed, alg=alg, sep=sep, results=model.results)


if __name__ == "__main__":
    
    start_time = time.time()

    df = parallel_results() #results()
    df.to_csv(f"{os.getcwd()}/output/df_simulation.csv", index=False)
    plot_resume(df)
    
    end_time = time.time()
    formatted = time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))
    print(f"Simulation ends in : {formatted} (hh:mm:ss)")
