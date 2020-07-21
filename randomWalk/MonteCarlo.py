# This class will run the main algorithm.

from randomWalk.Walker import *
import numpy as np

class MonteCarlo:

    def __init__(self, nP):
        #Initialize particles
        self.particles = []
        self.nP = nP
        # Init the number of particles
        for i in range(self.nP):
            self.particles.append(Walker())

    def seedWalkersSpecificLocation(self,x0,domain):
        #Seeds walkers in a specific location
        self.domain = domain
        for i in range(len(self.particles)):
            self.particles[i].position=x0

    def seedWalkersInGeometry(self, domain):
        #Seed walkers randomly in the domain
        self.domain = domain
        for i in range(len(self.particles)):
            self.particles[i].position = (np.random.uniform(0, 1) * domain.length) + domain.xCoordinates[0]

    def locateIndex(self, pos):
        #Locates index cell from a certain position
        closestBarrier = min(self.domain.barriers_position,
                             key=lambda x: abs(x - pos))  # O(n) could be improved to O(lgN) with binary search
        if pos > closestBarrier:
            return list(self.domain.barriers_position).index(closestBarrier)
        else:
            return list(self.domain.barriers_position).index(closestBarrier) - 1

    def locateWalkers(self):
        # Locate walkers inside the domain
        # Total Complexity O(numberParticles*numberOfBarriers)
        for i in range(len(self.particles)):
            pos = self.particles[i].position
            self.particles[i].index = self.locateIndex(pos)

    def runAlgorithm(self, T,dT, transit_model):
        time=0
        while(time<=T):
            self.allWalkersOneStep(dT, transit_model)
            time += dT
            if (time > T):
                self.allWalkersOneStep(time-T, transit_model)
                break

    def checkWalkersLocation(self,x0):
        counterRight=0
        counterLeft=0
        counterx0=0
        for i in range(len(self.particles)):
            dx = (self.particles[i].position-x0)
            if(dx>0):
                counterRight+=1
            elif(dx<0):
                counterLeft+=1
            else:
                counterx0+=1

    def allWalkersOneStep(self,dt,transit_model):
        # 1 - Walkers that go out of the boundaries. Should I reflect them into the domain or create a buffer?
        for i in range(len(self.particles)):
            pos = self.particles[i].position
            indexCell = self.particles[i].index
            D = self.domain.diffusivity[indexCell]
            P = self.domain.permeability[indexCell]
            s = self.particles[i].stepWalker(D, dt)
            newIndex = self.locateIndex(s + pos)
            crosses = False
            if newIndex != indexCell:
                lowerBoundBarrier = self.domain.barriers_position[indexCell]
                upperBoundBarrier = self.domain.barriers_position[indexCell+1]
                barrier_pos = self.computeIntersection(lowerBoundBarrier,upperBoundBarrier,s+pos)
                walker_transit_probability = self.particles[i].transitProbability()
                distance_pos_to_barrier = abs(barrier_pos - pos)
                distance_barrier_to_target = abs((s + pos) -barrier_pos)
                probability_crossing = transit_model(distance_pos_to_barrier, P, D)
                if probability_crossing == 1:
                    crosses = True
                elif probability_crossing == 0:
                    crosses = False
                else:
                    crosses = walker_transit_probability <= probability_crossing
                if crosses:
                    self.particles[i].index = newIndex
                    self.particles[i].position = s + pos
                else:
                    self.particles[i].index = indexCell
                    if pos < barrier_pos:
                        self.particles[i].position = barrier_pos - distance_barrier_to_target
                    else:
                        self.particles[i].position = barrier_pos + distance_barrier_to_target
            else:
                self.particles[i].index = indexCell
                self.particles[i].position = s + pos


    def getWalkerPositions(self):
        finalPositions = []
        for i in range (len(self.particles)):
            finalPositions.append(self.particles[i].position)
        return finalPositions

    def computeIntersection(self,lower,upper,nextPos):
        if nextPos>upper:
            return upper
        else:
            return lower
