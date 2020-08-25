import warnings

import numpy as np
from property_cached import cached_property


class Domain:

    def __init__(self, lengths, diffusivities, permeabilities):
        # store
        self.lengths = np.array(lengths)
        self.diffusivities = np.array(diffusivities)
        self.permeabilities = np.array(permeabilities)
        # verify size of arrays
        if len(self.lengths) != len(self.diffusivities):
            raise Exception("Inconsistent size of 'lengths' and 'diffusivities'")
        if len(self.permeabilities) == len(self.lengths)-1:  # default impermeable ends
            self.permeabilities = np.append(0, np.append(self.permeabilities, 0))
        elif len(self.permeabilities) != len(self.lengths)+1:
            raise Exception("Inconsistent size of 'lengths' and 'permeabilities'")

    @property
    def N(self):
        return len(self.lengths)

    @cached_property
    def barriers(self):
        return np.append(0, np.cumsum(self.lengths))

    @cached_property
    def total_length(self):
        return np.sum(self.lengths)

    def locate(self, positions, **kwargs):
        return locate_inside(positions, self.barriers, **kwargs)


def locate_inside(positions, barriers, verify=False):
    N = positions.size
    indices = np.full(N, -1, dtype=int)  # -1 = default for out-of-bounds
    idx, inside = np.where((positions >= barriers[:-1, np.newaxis]) &
                           (positions <= barriers[1:, np.newaxis]))
    indices[inside] = idx  # this also handles the case where pos==barrier
    if verify and inside.size != N:
        warnings.warn("Some walkers are located outside the domain, index = -1")
    return indices
