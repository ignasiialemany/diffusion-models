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
        #Default value when walkers are out of the domain
        indices=np.array([-1] * self.nP, dtype=int)
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

    def crossesBarrier(self, d_before, d_after, P, D_i, D_j, transit_model):
        crosses,modifiedStep = transit_model.crosses(d_before,d_after,D_i,D_j,P)
        return crosses,modifiedStep

    def allWalkersOneStep(self, dt, transit_model):
        step = np.random.normal(0, 1, len(self.position)) * np.sqrt(self.domain.diffusivity[self.indices] * 2 * dt)
        initPos = self.position.copy()
        self.position += step
        newIndices = self.locateIndex()
        interacting = self.indices != newIndices
        if np.any(interacting == True):
            initialPosition = initPos[interacting]
            pos_interact = self.position[interacting]
            #Correct default value when walkers are out of the domain
            newIndices[np.where((newIndices == -1) & (self.position>self.domain.length))] = self.domain.numberOfCompartments-1
            newIndices[np.where((newIndices == -1) & (self.position<0))] = 0
            barrier_index = np.where(step > 0, self.indices+1, self.indices)[interacting]
            d_after = np.abs(self.domain.barriers_position[barrier_index] - pos_interact)
            d_before = np.abs(self.domain.barriers_position[barrier_index] - initPos[interacting])
            crosses,modifiedStep = self.crossesBarrier(d_before, d_after, self.domain.permeability[barrier_index], self.domain.diffusivity[self.indices][interacting],
                                          self.domain.diffusivity[newIndices][interacting], transit_model)
            pos_reflect = -1 * np.sign(step[interacting]) * d_after + self.domain.barriers_position[barrier_index]
            pos_pass = self.domain.barriers_position[barrier_index] + np.sign(step[interacting]) * (modifiedStep)
            pos_new = np.where(crosses, pos_pass, pos_reflect)
            new_indices = np.where(crosses, newIndices[interacting], self.indices[interacting])
            self.position[interacting] = pos_new
            self.indices[interacting] = new_indices

    def getWalkerPositions(self):
        return self.position
