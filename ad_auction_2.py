from main import parse_config, instantiate_agents, instantiate_auction



def ad_auction_2(config_file_name, rounds_per_iter=100):
    # Parse configuration file
    (
        rng,
        config,
        agent_configs,
        agents2items,
        agents2item_values,
        num_runs,
        max_slots,
        embedding_size,
        embedding_var,
        obs_embedding_size,
    ) = parse_config(config_file_name)

    agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
    auction, num_iter, _SKIP_ME_rounds_per_iter, output_dir = instantiate_auction(
        rng,
        config,
        agents2items,
        agents2item_values,
        agents,
        max_slots,
        embedding_size,
        embedding_var,
        obs_embedding_size,
    )

    for _ in range(rounds_per_iter):
        auction.simulate_opportunity()

    return agents[0].net_utility


breakpoint()
