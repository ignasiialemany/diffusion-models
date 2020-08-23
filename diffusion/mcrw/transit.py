from abc import ABC, abstractmethod

import numpy as np


class TransitABC(ABC):

    @classmethod
    @abstractmethod
    def probability(cls, dx_i, dx_j, D_i, D_j, P):
        pass

    @classmethod
    def _crossing(cls, decision, dx_i, dx_j, D_i, D_j):
        step = np.abs(dx_i) + np.abs(dx_j)
        dx_after = (step - dx_i) * np.sqrt(D_j/D_i)  # if success
        dx_j[decision] = dx_after[decision]  # modify in-place

    @classmethod
    def crosses(cls, dx_i, dx_j, D_i, D_j, P):
        p_t = cls.probability(dx_i, dx_j, D_i, D_j, P)
        uval = np.random.uniform(0, 1, dx_j.size)
        decision = uval <= p_t
        cls._crossing(decision, dx_i, dx_j, D_i, D_j)
        return decision
