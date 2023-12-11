import numpy as np
import torch
from parameters_direct import ParametersDirect
d


class PolicyLearningBidder(Bidder):
    """ A bidder that estimates the optimal bid shading distribution via policy learning """

    def __init__(self, rng, gamma_sigma, init_gamma=1.0):
        self.gamma_sigma = gamma_sigma
        #self.model = BidShadingContextualBandit(loss)
        self.model_initialised = False
        self.parameter_handler = ParametersDirect(self.model)
        super(PolicyLearningBidder, self).__init__(rng)

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        bid = value * estimated_CTR
        if not self.model_initialised:
            x = torch.Tensor([estimated_CTR, value])
            gamma, propensity = self.model(x)
            gamma = torch.clip(gamma, 0.0, 1.0)
        else:
            x = torch.Tensor([estimated_CTR, value])
            gamma, propensity = self.model(x)
            gamma = torch.clip(gamma, 0.0, 1.0)

        bid *= gamma.detach().item() if self.model_initialised else gamma
        return bid


    def clear_logs(self, memory):
        if not memory:
            self.gammas = []
            self.propensities = []
        else:
            self.gammas = self.gammas[-memory:]
            self.propensities = self.propensities[-memory:]


    def get_parameters(self) -> np.ndarray:
        return self.parameter_handler.get_params()

    def set_parameters(self, parameters: np.ndarray):
        self.parameter_handler.set_params(parameters)