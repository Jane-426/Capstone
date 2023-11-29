import numpy as np

from main import parse_config, instantiate_agents, instantiate_auction


class AdAuction:
    def __init__(self, config_file_name, rounds_per_iter=100, warm_up_iterations=1000):
        self._rounds_per_iter = rounds_per_iter
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
        
        self._agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
        self._auction, num_iter, _, output_dir = instantiate_auction(
            rng,
            config,
            agents2items,
            agents2item_values,
            self._agents,
            max_slots,
            embedding_size,
            embedding_var,
            obs_embedding_size,
        )
        self._warm_up(warm_up_iterations)

    def _warm_up(self, warm_up_iterations):
        for _ in range(warm_up_iterations):
            self._auction.simulate_opportunity()
        
    def us(self):
        return self._agents[0]

    def them(self):
        return self._agents[1:]

    def _mean_gross_utility(self):
        return np.mean([a.gross_utility for a in self._agents])

    def _net_utilty(self, agents):
        return np.mean([a.net_utility for a in agents])
    
    def run_episode(self):
        for a in self._agents:
            a.clear_utility()
        
        for _ in range(self._rounds_per_iter):
            self._auction.simulate_opportunity()

        return (self._net_utilty([self.us()]) - self._net_utilty(self.them())) / self._mean_gross_utility()
    
