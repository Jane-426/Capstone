from Bidder import Bidder


class ConstShadingBidder(Bidder):
    def __init__(self, rng):
        super().__init__(rng)
        self.gamma = rng.uniform()
        print(f"ConstShadingBidder: gamma = {self.gamma}")

    def bid(self, value, _, estimated_CTR):
        return self.gamma * value * estimated_CTR

    def clear_logs(self, memory):
        pass
