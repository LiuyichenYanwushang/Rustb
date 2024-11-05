import numpy as np
import matplotlib.pyplot as plt
import matplotlib
del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

data=np.loadtxt("band.txt")

#plt.scatter(range(len(data)),data,c="k")

x0=range(len(data))
y0=data
#y0-=(y0[int(len(x0)/2)]+y0[int(len(x0)/2)-1])/2
y0*=1000;

plt.hlines(y = 0,xmin = np.min(x0),xmax = np.max(x0), linestyles = "--",colors = "black",lw = 3)
#plt.plot(x0,y0,"o",ms = 8, c = "blue",alpha = 0.5)
nl=-1;
nh=1;

plt.plot(x0[:int(len(x0)/2) + nl],y0[:int(len(x0)/2) + nl],"o",ms = 8, c = "blue",alpha = 0.5)
plt.plot(x0[int(len(x0)/2) + nh:],y0[int(len(x0)/2) + nh:],"o",ms = 8, c = "blue",alpha = 0.5)
plt.plot(x0[int(len(x0)/2) + nl:int(len(x0)/2) + nh], y0[int(len(y0)/2) + nl:int(len(y0)/2) + nh], "o", c='red', ms = 8,alpha=0.5,label = "Corner modes")
font2 = {
         'weight': 'normal',
         'size': 40,
         }
font3 = {
         'weight': 'normal',
         'size': 25,
         }
plt.ylabel("E-E$_\mathrm{f}$(meV)",font2)
plt.xlabel("Energy Level", font2)
plt.xticks([])
#plt.yticks([-0.6,0,0.6], size = 40)
#plt.yticks([float(format(-20,".0f")),float(format(0,".0f")),float(format(20,".0f"))], size = 40)
plt.yticks([float(format(0,".0f"))], size = 40)
#plt.legend(loc = 'upper left', bbox_to_anchor=(-0.1,1.05), shadow = True, prop = font3, markerscale = 1.3)
plt.legend(loc = 'upper left', prop = font3, markerscale = 1.3)
plt.tick_params(axis='x',width = 5,length = 10)
plt.tick_params(axis='y',width = 5,length = 10)
ax = plt.gca()
ax.spines["bottom"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
ax.spines["right"].set_linewidth(3)
ax.spines["top"].set_linewidth(3)
n0=20
ax.set_xlim([int(len(x0)/2)-n0,int(len(x0)/2)+n0]);
ax.set_ylim([y0[int(len(x0)/2)-n0],y0[int(len(x0)/2)+n0]]);
plt.tight_layout()
plt.savefig("energy.pdf",dpi=1000)
plt.close()

evec=np.loadtxt("evec.txt");
(nsta,natom)=evec.shape;
print(nsta,natom)
evec=evec*nsta;
structure=np.loadtxt("structure.txt");


#for i in range(10):
#    fig,ax=plt.subplots()
#    ax.scatter(structure[:,0],structure[:,1],c="b",s=10)
#    ax.scatter(structure[:,0],structure[:,1],c="k",s=evec[6558+i,:]/5)
#    print(data[6558+i])
#    ax.set_aspect(1)
#    fig.savefig("structure_%d.pdf"%i)
#    plt.close()
nsta=int(len(data)/2);
s0=evec[nsta-6,:]*0
for i in range(nl,nh):
    s0+=evec[nsta+i,:]
#s0=evec[6558,:]+evec[6559,:]+evec[6560,:]+evec[6561,:]
A=np.sum(evec[0,:]);
s0=(s0-np.min(s0))/(np.max(s0)-np.min(s0));
#sc=plt.scatter(structure[:,0],structure[:,1],c=s0*0.,s=1,vmin=0,vmax=1,cmap="bwr")
#sc=plt.scatter(structure[:,0],structure[:,1],c=s0,s=(s0*20)**2,vmin=0,vmax=1,cmap="bwr")
sc=plt.scatter(structure[:,0],structure[:,1],c=s0*0.,s=10,vmin=0,vmax=1,cmap="RdPu")
sc=plt.scatter(structure[:,0],structure[:,1],c=s0,s=s0*50,vmin=0,vmax=1,cmap="RdPu")
cb = plt.colorbar(sc,fraction = 0.04,ticks=[ 0, 1],orientation='horizontal',pad=0.05)
#cb.ax.set_position([0.2, 0.2, 0.61, 0.1])
cb.ax.tick_params(labelsize=20)
plt.axis('equal')
plt.axis('off')
plt.savefig("structure_all.pdf",dpi=1000)
plt.close()
