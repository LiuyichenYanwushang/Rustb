import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt('BAND.dat')
k_nodes=[]
label=[]
f=open('KLABELS')
for i in f.readlines():
    k_nodes.append(float(i.split()[0]))
    label.append(i.split()[1])
fig,ax=plt.subplots()
ax.plot(data[:,0],data[:,1:],c='b')
for x in k_nodes:
    ax.axvline(x,c='k')
ax.set_xticks(k_nodes)
ax.set_xticklabels(label)
ax.set_xlim([0,k_nodes[-1]])
ax.set_ylim([-0.2,0.2])
fig.savefig('band.pdf')
