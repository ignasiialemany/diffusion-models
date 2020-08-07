#This is a class that will contain the 1D geometry.
import numpy as np

class Domain:

    def __init__(self,lengths,diffusivities,permeability):
        self.lengthVector = lengths
        self.length = sum(lengths)
        self.diffusivity = diffusivities
        self.permeability = permeability
        self.diffusivity=np.array(self.diffusivity)
        self.permeability = np.array(self.permeability)

        #When creating the domain those are computed
        self.barriers_position=[]
        self.numberOfCompartments = len(diffusivities)
        self.createDomain()

    def createDomain(self):
        #Coordinates in x
        self.barriers_position = np.concatenate((np.zeros(1), np.cumsum(self.lengthVector)))
        self.numberOfCompartments = len(self.lengthVector)
