import numpy as np
from main import parse_config, instantiate_agents, instantiate_auction

def ad_auction(config_path):
    # Parse configuration file
    rng, config, agent_configs, agents2items, agents2item_values, \
    num_runs, max_slots, embedding_size, embedding_var, \
    obs_embedding_size = parse_config(config_path)

    # Modify the first agent entry to use TruthfulBidder
    config['agents'][0]['bidder']['type'] = 'TruthfulBidder'
    config['agents'][0]['name'] = 'Truthful Oracle 1'
    config['agents'][1]['name'] = 'Truthful Oracle 2'
    config['agents'][0]['num_copies'] = 1

    # Re-instantiate agents with the modified configuration
    agents = instantiate_agents(rng, config['agents'], agents2item_values, agents2items)

    auction, num_iter, rounds_per_iter, output_dir = \
        instantiate_auction(rng,
                            config,
                            agents2items,
                            agents2item_values,
                            agents,
                            max_slots,
                            embedding_size,
                            embedding_var,
                            obs_embedding_size)
    # Read the configuration file
    with open(config_path) as f:
        config = json.load(f)

    # Run the auction
    for _ in range(rounds_per_iter):
        auction.simulate_opportunity()

    # Return the net utility of the first agent
    return agents[0].net_utility
