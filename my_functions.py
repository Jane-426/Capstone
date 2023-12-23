# my_functions.py

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import numpy as np
import json
import os




# Function to run a single replication in a separate process
def run_single_replication(args):
    acq_func, i_replication = args
    ad_auction = AdAuction(config_file_name="configs/config-advantage-BO.json", seed=i_replication)
    result = gp_minimize(
        objective,
        param_space,
        n_calls=84,
        acq_func=acq_func,
        random_state=i_replication,
        verbose=True
    )

    y_total_replication = -sum(result.func_vals)

    # Save results to a file
    output_path = f"results_{acq_func}_{i_replication}.json"
    with open(output_path, "w") as file:
        json.dump({"acq_func": acq_func, "replication": i_replication, "y_total": y_total_replication}, file)

    return output_path
