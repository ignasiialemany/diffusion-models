from diffusion.mcrw import MonteCarlo
from diffusion.domain import Domain
from diffusion.mcrw.transit import Fieremans2010,HybridModel2022
import matplotlib.pyplot as plt
from scipy.io import savemat
import numpy as np

#Set the domain
domain = Domain(lengths=[200,5,120,5,200],diffusivities=[1.5,2.5,1.5,2.5,1.5],permeabilities=[0,0.02,0.02,0.02,0.02,0])

#Set time and algorithm
T = 100.
A = 0.2
v_func = lambda x,t: (0.)*x
#v_func = lambda x,t: A*2*np.pi*(1/100.)*np.cos(2*np.pi*(1/100.)*t)*x
algorithm = MonteCarlo(N=300000,velocity_function=v_func)
print(domain.barriers[2]+3,domain.barriers[3]-3)
algorithm.seed_in_interval(domain,domain.barriers[2]+3,domain.barriers[3]-3)
init_pos = algorithm.position
savemat("init_pos.mat",{'init_pos':init_pos})

#Run the algorithm
algorithm.run(T,0.005,HybridModel2022())
final_pos = algorithm.position
savemat('final_pos.mat',{'final_pos':final_pos})
bins = np.concatenate([np.arange(-200,-60,5),np.arange(-60,60,5),np.arange(60,200,5)])
plt.hist(final_pos,bins=bins,density=True,histtype='step')
plt.show()
