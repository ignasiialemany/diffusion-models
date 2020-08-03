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

    def locateIndex(self):
        # This one works but we will need to loop through the entire vector. Maybe vectorize is faster than loop?
        for i in range(len(self.domain.barriers_position) - 1):
            leftBarrier = self.domain.barriers_position[i]
            rightBarrier = self.domain.barriers_position[i + 1]
            index = np.where((self.position <= rightBarrier) & (self.position >= leftBarrier))[0]
            self.indices[index] = i

    def runAlgorithm(self, T, dT, transit_model):
        time = 0
        while (time <= T):
            self.allWalkersOneStep(dT, transit_model)
            time += dT
            if (time > T):
                self.allWalkersOneStep(time - T, transit_model)
                break

    def crossesBarrier(self, dx, P, D, transit_model):
        return transit_model(np.abs(dx), P, D)

    def allWalkersOneStep(self, dt, transit_model):
        for i in range(len(self.domain.barriers_position) - 1):

            leftBarrier = self.domain.barriers_position[i]
            rightBarrier = self.domain.barriers_position[i + 1]

            #Walkers that are in the compartment
            indexWalkersToStep = np.where(self.indices == i)[0]
            step = np.random.normal(0, 1, len(indexWalkersToStep)) * np.sqrt(self.domain.diffusivity[i] * 2 * dt)

            # Walkers that advance to the following step
            new_pos = self.position[indexWalkersToStep] + step

            rightWalkersCross = indexWalkersToStep[np.where((new_pos > rightBarrier))[0]]
            leftWalkersCross = indexWalkersToStep[np.where((new_pos< leftBarrier))[0]]


            # Analyze left cross/reflection or right cross/reflection
            if len(rightWalkersCross) + len(leftWalkersCross) != 0:

                self.position[indexWalkersToStep] += step

                dxLeft = leftBarrier - self.position[leftWalkersCross]
                dxRight = self.position[rightWalkersCross] - rightBarrier


                leftProbabilityCross = self.crossesBarrier(dxLeft, self.domain.permeability[i],
                                                           self.domain.diffusivity[i], transit_model)

                leftWalkerProbability = np.random.uniform(0, 1, len(dxLeft))
                rightProbabilityCross = self.crossesBarrier(dxRight, self.domain.permeability[i + 1],
                                                            self.domain.diffusivity[i], transit_model)
                rightWalkerProbability = np.random.uniform(0, 1, len(dxRight))

                leftParticlesReflect = np.where((leftProbabilityCross == 0) | (leftProbabilityCross < leftWalkerProbability))[0]
                rightParticlesReflect = np.where((rightProbabilityCross == 0) | (rightProbabilityCross < rightWalkerProbability))[0]

                self.position[leftWalkersCross[leftParticlesReflect]] = leftBarrier \
                                                                        +dxLeft[leftParticlesReflect]


                self.position[rightWalkersCross[rightParticlesReflect]] = rightBarrier\
                                                                          -dxRight[rightParticlesReflect]


                #leftWalkersCross = leftWalkersCross[np.in1d(range(len(leftWalkersCross)),leftParticlesReflect)]
                #rightWalkersCross = rightWalkersCross[np.in1d(range(len(rightWalkersCross)),rightParticlesReflect)]
                leftWalkersCross = np.delete(leftWalkersCross, leftParticlesReflect)
                rightWalkersCross = np.delete(rightWalkersCross, rightParticlesReflect)

                self.indices[leftWalkersCross] -= 1
                self.indices[rightWalkersCross] += 1

            else:

                self.position[indexWalkersToStep] = new_pos

    def getWalkerPositions(self):
        return self.position

    def computeIntersection(self, lower, upper, nextPos):
        if nextPos > upper:
            return upper
        else:
            return lower
