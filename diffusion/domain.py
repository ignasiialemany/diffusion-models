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

    @property
    def barriers(self):
        return np.append(0, np.cumsum(self.lengths)) - self.total_length/2

    @property
    def total_length(self):
        return np.sum(self.lengths)

    def locate(self, positions, **kwargs):
        return locate_inside(positions, self.barriers, **kwargs)
    
    def update_barriers(self, velocity_func, curr_time, dt):
        old_barriers = self.barriers
        velocity_at_barriers = velocity_func(old_barriers, curr_time)
        new_barriers = old_barriers + velocity_at_barriers*dt
        self.lengths = np.diff(new_barriers)
        
        
        

def locate_inside(positions, barriers, verify=False):
    N = positions.size
    indices = np.full(N, -1, dtype=int)  # -1 = default for out-of-bounds
    idx, inside = np.where((positions >= barriers[:-1, np.newaxis]) &
                           (positions <= barriers[1:, np.newaxis]))
    indices[inside] = idx  # this also handles the case where pos==barrier
    if verify and inside.size != N:
        warnings.warn("Some walkers are located outside the domain, index = -1")
    return indices
