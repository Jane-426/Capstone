# my_functions.py
import os
import sys
import json
import numpy as np
from multiprocessing import Pool
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from bo_bidder import BOBidder
from ad_auction import AdAuction

# Global parameters for ease of testing
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

def run_single_replication(args):
    g_params, acq_func, i_replication = args
    ad_auction = AdAuction(
        config_file_name="configs/config-advantage-BO.json",
        seed=i_replication,
        extra_classes={"BOBidder": BOBidder},
    )

    num_params = 24
    param_space = [Real(-1.0, 1.0, name=f"param{i}") for i in range(num_params)]
    bo_bidder = ad_auction.us().bidder

    i_call = 0
    trace = []

    @use_named_args(param_space)
    def objective(**params):
        nonlocal i_call
        param_values = np.array(list(params.values()))
        bo_bidder.set_parameters(param_values)
        advantage = evaluate(ad_auction, num_iterations=g_params["num_iterations"])
        trace.append(advantage)
        i_call += 1
        return -advantage

    gp_minimize(
        objective,
        param_space,
        n_calls=g_params["n_calls"],
        acq_func=acq_func,
        random_state=i_replication,
        verbose=True,
    )

    output_path = f"results_{acq_func}_{i_replication}.json"
    with open(output_path, "w") as file:
        json.dump({"acq_func": acq_func, "replication": i_replication, "trace": trace}, file)

    return output_path

def run_all_experiments(acquisition_functions, start_rep, end_rep):
    all_results = []

    for i_replication in range(start_rep, end_rep):
        results = {}
        for acq_func in acquisition_functions:
            print(f"Running experiment for acquisition function: {acq_func} with replication {i_replication}")
            sys.stdout.flush()

            result_file = run_single_replication((g_params, acq_func, i_replication))

            with open(result_file, "r") as file:
                data = json.load(file)
                trace = data["trace"]
                cumulative_trace = np.cumsum(trace)
                results[acq_func] = cumulative_trace
                os.remove(result_file)

        all_results.append(results)

    return all_results
