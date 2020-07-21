
import numpy as np

class Walker:

    def __init__(self):
        self.position=0
        self.index=0

    def stepWalker(self,diffusivity, dt):
        #TODO:Try constant step and see how permeablity model behaves
        step = np.random.normal(0,1)
        step = step * np.sqrt(diffusivity*2*dt)
        return step

    def transitProbability(self):
        return np.random.uniform(0,1)





