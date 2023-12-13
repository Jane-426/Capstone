import numpy as np
import torch
from parameters_direct import ParametersDirect
from Models import BidShadingPolicy
from Bidder import Bidder

class BOBidder(Bidder):

    def __init__(self, rng):
        super().__init__(rng)
        self.model = BidShadingPolicy()
        self.parameter_handler = ParametersDirect(self.model)
        num = len(self.get_parameters())
        self.set_parameters(.1*rng.normal(size=(num,)))
        # TODO: consider smaller values of BidShadingPolicy.min_sigma
        
    def bid(self, value, _, estimated_CTR):
        bid = value * estimated_CTR

        x = torch.Tensor([estimated_CTR, value])
        gamma, _ = self.model(x)
        gamma = torch.clip(gamma, 0.0, 1.0).detach().item()
        
        bid *= gamma
        return bid


    def clear_logs(self, memory):
        pass

    def get_parameters(self) -> np.ndarray:
        return self.parameter_handler.get_params()

    def set_parameters(self, parameters: np.ndarray):
        self.parameter_handler.set_params(parameters)
