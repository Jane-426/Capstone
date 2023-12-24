# my_functions.py

import os
import sys
from multiprocessing import Pool
import json

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from bo_bidder import BOBidder
from ad_auction import AdAuction

# makes testing easier
g_params = dict(
    num_iterations=1000,
    n_calls=84,
    num_reps=10,
)

def evaluate(ad_auction, num_iterations):
    trace = []
    for _ in range(num_iterations):
        r = ad_auction.run_episode()
        trace.append(r)
    return np.array(trace).mean()


# Function to run a single replication in a separate process
def run_single_replication(args):
    g_params, acq_func, i_replication = args
    ad_auction = AdAuction(
        config_file_name="configs/config-advantage-BO.json",
        seed=i_replication,
        extra_classes={"BOBidder": BOBidder},
    )

    # Define the parameter space
    num_params = 24
    param_space = [Real(-1.0, 1.0, name=f"param{i}") for i in range(num_params)]

    bo_bidder = ad_auction.us().bidder

    # Objective function
    i_call = 0
    @use_named_args(param_space)
    def objective(**params):
        nonlocal i_call
        param_values = np.array(list(params.values()))
        bo_bidder.set_parameters(param_values)
        advantage = evaluate(ad_auction, num_iterations=g_params['num_iterations'])
        print ("OBJECTIVE:", os.getpid(), i_call, g_params['num_iterations'], g_params['n_calls'], advantage)
        sys.stdout.flush()
        i_call+=1
        return -advantage

    result = gp_minimize(
        objective,
        param_space,
        n_calls=g_params['n_calls'], 
        acq_func=acq_func,
        random_state=i_replication,
        verbose=True,
    )

    y_total_replication = -sum(result.func_vals)

    # Save results to a file
    output_path = f"results_{acq_func}_{i_replication}.json"
    with open(output_path, "w") as file:
        json.dump(
            {
                "acq_func": acq_func,
                "replication": i_replication,
                "y_total": y_total_replication,
            },
            file,
        )

    return output_path

def run_experiment(acq_func):
    print (f"EXPERIMENT: {g_params}")
    sys.stdout.flush()
    
    with Pool(g_params['num_reps']) as p:
        args = [(g_params, acq_func, i) for i in range(g_params['num_reps'])]
        result_files = p.map(run_single_replication, args)

    # Collect results from files
    y_totals = []
    for file_name in result_files:
        with open(file_name, "r") as file:
            data = json.load(file)
            y_totals.append(data["y_total"])
            os.remove(file_name)

    avg_total_advantage = np.mean(y_totals)
    return avg_total_advantage
