import numpy as np

def evaluate(ad_auction, num_iterations):
    trace = []
    for _ in range(num_iterations):
        r = ad_auction.run_episode()
        trace.append(r)
        # print (ad_auction.agent().name, r)
    return np.array(trace)

