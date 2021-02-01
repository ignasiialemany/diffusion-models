from abc import ABC, abstractmethod

import numpy as np


class TransitABC(ABC):

    @abstractmethod
    def probability(self, dx_i, dx_j, D_i, D_j, P):
        pass

    def _crossing(self, decision, dx_i, dx_j, D_i, D_j):
        dx_j[decision] *= np.sqrt(D_j/D_i)[decision]  # modify in-place

    def crosses(self, dx_i, dx_j, D_i, D_j, P):
        p_t = cls.probability(dx_i, dx_j, D_i, D_j, P)
        uval = np.random.uniform(0, 1, dx_j.size)
        decision = uval <= p_t
        cls._crossing(decision, dx_i, dx_j, D_i, D_j)
        return decision


class Constant(TransitABC):
    """Transit model using a fixed probability.
    Can be used as an override, to set all barriers to be
    impermeable (p_t=0), fully permeable (p_t=1), or anything inbetween.
    """

    def __init__(self, p_t=0.5):
        super().__init__()  # call superclass constructor
        self.p_t = p_t

    def probability(self, *args):
        return self.p_t


class Parrot(TransitABC):
    """Transit model that repeats barrier permeability
    Clips the value of permeability to [0, 1]
    """

    def probability(self, dx_i, dx_j, D_i, D_j, P):
        p_t = np.clip(P, 0, 1)
        return p_t


class Maruyama2017(TransitABC):
    """Interface transit model
    Based on (Maruyama, 2017, DOI:10.1103/PhysRevE.96.032135)
    """

    def probability(self, dx_i, dx_j, D_i, D_j, P):
        p_t = np.minimum(np.sqrt(D_j/D_i), 1.0)
        return p_t


class Fieremans2010(TransitABC):
    """Membrane transit model
    Based on (Fieremans et al, 2010, DOI:10.1002/nbm.1577)
    """

    def probability(self, dx_i, dx_j, D_i, D_j, P):
        term = 2*dx_i*P/D_i
        p_t = np.where(np.isinf(P), 1, term/(1+term))
        return p_t


class Szafer1995(TransitABC):
    """Membrane transit model
    Based on (Szafer et al, 1995, DOI:10.1002/mrm.1910330516)
    """

    def probability(self, dx_i, dx_j, D_i, D_j, P):
        dx = dx_i + dx_j
        v_i = 2*D_i/dx
        p_t = 4*P/v_i
        return p_t
