# This class will run the main algorithm.

import numpy as np


class MonteCarlo:

    def __init__(self, nP):
        self.nP = nP
        # Instead of initializing Walker class we initialize positions/index
        self.position = np.zeros(nP)
        self.indices = np.zeros(nP)

    def seedWalkersSpecificLocation(self, x0, domain):
        self.domain = domain
        self.position.fill(x0)

    def seedWalkersInGeometry(self, domain):
        # Seed walkers randomly in the domain
        self.domain = domain
        self.position = np.random.uniform(0, 1, self.nP) * self.domain.length

    def locateWalkers(self):
        self.indices = self.locateIndex()

    def locateIndex(self):
        indices=np.zeros(self.nP,dtype=int)
        # This one works but we will need to loop through the entire vector. Maybe vectorize is faster than loop?
        for i in range(len(self.domain.barriers_position) - 1):
            leftBarrier = self.domain.barriers_position[i]
            rightBarrier = self.domain.barriers_position[i + 1]
            index = np.where((self.position <= rightBarrier) & (self.position >= leftBarrier))[0]
            indices[index] = i
        return indices

    def runAlgorithm(self, T, dT, transit_model):
        time = 0
        while (time <= T):
            self.allWalkersOneStep(dT, transit_model)
            time += dT
            if (time > T):
                self.allWalkersOneStep(time - T, transit_model)
                break

    def crossesBarrier(self, dx, P, D, transit_model):
        walkerProbability = np.random.uniform(0,1,len(dx))
        transitProbability = np.array(transit_model(np.abs(dx), P, D))
        crosses = (transitProbability == 1) | (walkerProbability <= transitProbability)
        return crosses

    def allWalkersOneStep(self, dt, transit_model):
        step = np.random.normal(0, 1, len(self.position)) * np.sqrt(self.domain.diffusivity[self.indices] * 2 * dt)
        self.position += step
        newIndices = self.locateIndex()
        interacting = self.indices != newIndices
        pos_interact = self.position[interacting]
        barrier_index = np.where(step > 0, newIndices, self.indices)[interacting]
        dx = np.abs(self.domain.barriers_position[barrier_index] - pos_interact)
        crosses = self.crossesBarrier(dx, self.domain.permeability[barrier_index],
                                      self.domain.diffusivity[self.indices][interacting], transit_model)
        pos_reflect = -1 * np.sign(step[interacting]) * dx + self.domain.barriers_position[barrier_index]
        pos_pass = pos_interact
        pos_new = np.where(crosses, pos_pass, pos_reflect)
        new_indices = np.where(crosses, newIndices[interacting], self.indices[interacting])
        self.position[interacting] = pos_new
        self.indices[interacting] = new_indices

    def getWalkerPositions(self):
        return self.position