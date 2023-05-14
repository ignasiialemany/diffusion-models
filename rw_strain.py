from diffusion.mcrw import MonteCarlo
from diffusion.domain import Domain
from diffusion.mcrw.transit import Fieremans2010
import matplotlib.pyplot as plt
from scipy.io import savemat

domain = Domain(lengths=[200,5,120,5,200],diffusivities=[1.5,2.5,1.5,2.5,1.5],permeabilities=[0,0.02,0.02,0.02,0.02,0])
algorithm = MonteCarlo(N=100000)
algorithm.seed_in_interval(domain,domain.barriers[2]+3,domain.barriers[3]-3)
init_pos = algorithm.position
savemat("init_pos.mat",{'init_pos':init_pos})
algorithm.run(100.,0.01,Fieremans2010())
final_pos = algorithm.position
savemat('final_pos.mat',{'final_pos':final_pos})
plt.hist(final_pos,bins=100,density=True,histtype='step')
plt.show()
