import numpy as np
import matplotlib.pyplot as plt

class MonteCarlo:

    def __init__(self, N, *, rngseed=None, velocity_function=None):
        self.position = np.zeros(N)
        self.indices = np.zeros(N, dtype=int)
        self.rng = np.random.default_rng(rngseed)
        if velocity_function is None:
            self.velocity_function = lambda x,t : 0
        else:
            self.velocity_function = velocity_function

    @property
    def N(self):
        return self.position.size

    # ======= #
    # seeding
    # ======= #

    def seed_in_interval(self,domain,xmin,xmax):
        self.domain = domain
        self._place(self.rng.uniform(xmin,xmax,self.N))
        
        
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
        
        curr_time = 0
        for dt in steps:
            self.one_step(dt, transit_model, curr_time)
            curr_time += dt


    def one_step(self, dt, transit_model, curr_time):
        """Perform a single step"""

        # calculate the step
        D_current = self.domain.diffusivities[self.indices]
        step = np.array([-1,+1])[self.rng.choice(2, self.N)] * np.sqrt(2*D_current*dt)
        
        #Predictor-Corrector
        velocities = self.velocity_function(self.position, curr_time)
        #predict_pos = self.position + velocities * dt
        #velocity_after = self.velocity_function(predict_pos, curr_time)
        #average_velocity = (velocities + velocity_after)/2
        
        #Update particle to strain
        self.position = self.position + velocities * dt
        
        #Update domain barriers to strain
        self.domain.update_barriers(self.velocity_function, curr_time, dt)
        
        # old and new state
        old_pos = self.position 
        old_idx = self.domain.locate(old_pos)
        new_pos = self.position + step 
        
        #print(np.max(ratio))
        
        #old_idx = self.indices
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
        
        #bins = np.concatenate([np.arange(-200,-60,5),np.arange(-60,60,5),np.arange(60,200,5)])
        #plt.clf()
        #plt.hist(self.position, bins=bins, density=True, histtype='step')
        #plt.title(f'time = {curr_time:.1f}')
        #plt.draw()
        #plt.pause(0.01)
        
