import numpy as np
import torch


class ParametersDirect:
    def __init__(self, model):
        self._model = model
        self._num_params = self._count_params()

    def num_params(self):
        return self._num_params

    def _count_params(self):
        n = 0
        for p in self._model.parameters():
            if p.requires_grad:
                n += p.numel()
        return n

    def set_params(self, params):
        assert len(params.shape) == 1
        assert params.shape[0] == self._num_params
        i = 0
        for p in self._model.parameters():
            if p.requires_grad:
                iend = i + p.numel()
                p.data.copy_(torch.from_numpy(params[i:iend]).reshape(p.shape))
                i = iend

    def get_params(self):
        data = []
        for p in self._model.parameters():
            if p.requires_grad:
                data.extend(list(p.data.numpy().flatten()))
        return np.array(data)
