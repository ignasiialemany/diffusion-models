import numpy as np


class MonteCarlo:

    def __init__(self, N, *, rngseed=None):
        self.position = np.zeros(N)
        self.indices = np.zeros(N, dtype=int)
        self.rng = np.random.default_rng(rngseed)

    @property
    def N(self):
        return self.position.size

    # ======= #
    # seeding
    # ======= #

    def seed(self, domain, x0=None):
        self.domain = domain
        if x0 is None:
            # Seed walkers randomly in the domain
            self._place(self.rng.uniform(0, 1, self.N) * self.domain.total_length)
        else:
            # Seed walkers in one place
            self._place(np.repeat(x0, self.N))

    def _place(self, pos):
        self.position = pos
        self.indices = self.domain.locate(pos, verify=True)

    # ======= #
    # execute
    # ======= #

    def run(self, T, dt, transit_model):
        """Execute the random walk
        If T is an integer, run T time steps until T*dt. Otherwise
        (if T is a float), run until T using a time step of ~dt
        """
        if isinstance(T, int):
            steps = [dt]*T
        else:  # T is a float
            time = np.linspace(0, T, int(T//dt)+1)
            steps = np.diff(time)
        for dt in steps:
            self.one_step(dt, transit_model)

    def one_step(self, dt, transit_model):
        """Perform a single step"""

        # calculate the step
        D_current = self.domain.diffusivities[self.indices]
        step = np.array([-1,+1])[self.rng.choice(2, self.N)] * np.sqrt(2*D_current*dt)

        # old and new state
        old_pos = self.position
        new_pos = self.position + step
        old_idx = self.indices
        new_idx = self.domain.locate(new_pos)

        interacting = old_idx != new_idx
        if np.any(interacting):

            # get the relevant variables
            step = step[interacting]
            pos_before, pos_after = old_pos[interacting], new_pos[interacting]
            idx_before = old_idx[interacting]
            idx_after = idx_before + np.where(step > 0, +1, -1)
            idx_after[idx_after >= self.domain.N] = -1  # declare out-of-bounds

            # get substrate parameters
            D_before = self.domain.diffusivities[idx_before]
            D_after = self.domain.diffusivities[idx_after]
            barrier_indices = idx_before + np.where(step > 0, +1, 0)  # +1 is legal (N_b = N_c + 1)
            barriers = self.domain.barriers[barrier_indices]
            P = self.domain.permeabilities[barrier_indices]

            # transit
            d_after = np.abs(barriers - pos_after)
            d_before = np.abs(barriers - pos_before)
            crosses = transit_model.crosses(d_before, d_after, D_before, D_after, P,
                                            exclude=(idx_after==-1),  # out-of-bounds / left domain
                                            rng=self.rng,
                                           )

            # update based on decision
            new_pos[interacting] = barriers + d_after * np.sign(step) * np.where(crosses, 1, -1)
            new_idx[interacting] = np.where(crosses, idx_after, idx_before)

        # write
        self.position = new_pos
        self.indices = new_idx
