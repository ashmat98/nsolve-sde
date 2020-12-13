import setpaths
from utils import get_RSH_harmonic_onlyka,rootsPolyOmega
import numpy as np
import style 
from matplotlib import pyplot as plt

EPS=1e-8


(b,o0,g0,ka,result, i_plots) = np.load(
    "dumps/color-plot-g0100-ka1000.0-o020.0-b1-dt1e-05.npy", allow_pickle=True)
N=3000000
dt=0.00001
skip=1
time = np.linspace(0,N//skip *dt, N//skip)


R,S,H,_ = get_RSH_harmonic_onlyka(o0,g0,b,ka)
SNR_omega = abs(H)/np.sqrt(R*S-H)
roots = rootsPolyOmega(g0,ka,b,o0)+EPS+EPS*1j
print(f"omega: {H/R:0.6f}")
# print(f"{alpha:0.4f}, {beta**2:0.4f}")
print(f"relaxation freq:  ",(np.real(roots)).round(4))
print(f"relaxation time:  ",1/(np.real(roots)).round(4))
print(f"rotational freq:",(np.imag(roots)).round(4))
print(f"rotational time:",1/(np.imag(roots)).round(4))

print(f"SNR: {SNR_omega:0.4f}")


print(f"g0_hat: {g0/ka:0.3f}  o0_hat: {o0/ka:0.3f}")
print(f"b:{b:0.1f}  g0:{g0:0.1f}  ka:{ka:0.1f}  o0:{o0:0.1f}    "\
f"$2 pi / omega$: {2*np.pi/(H/R+EPS):0.3f}s "\
f"SNR (expected):{SNR_omega:0.3f}  \n"\
f"root1    relaxation: {1/np.real(roots[0]):0.3f}s     rotation: {np.imag(roots[0]):0.3f}s^-1 \n"
f"root2    relaxation: {1/np.real(roots[1]):0.3f}s     rotation: {np.imag(roots[1]):0.3f}s^-1 \n"
f"root3    relaxation: {1/np.real(roots[2]):0.3f}s     rotation: {np.imag(roots[2]):0.3f}s^-1 ")




x,y,vx,vy,_,_ = result

crosprod=(-x[:,1:]*y[:,:-1] + y[:,1:]*x[:,:-1])/(EPS+np.sqrt(x[:,1:]**2+y[:,1:]**2)*np.sqrt(x[:,:-1]**2+y[:,:-1]**2))
phi = np.arcsin(crosprod).cumsum(axis=-1)
del crosprod

start = 0*100
TURNS = 0.0001
OMEGA = abs(H/R)
TURNS = OMEGA/2/np.pi* 1

rang = int(2*3.14/OMEGA/dt * TURNS)
path = 1
factor = 2.5
print(x.shape[1],rang, OMEGA,TURNS)


st=1

st1=100;print(rang, x.shape)


fig = plt.figure(figsize=(9,6), dpi=200)

plt.subplot(2,1,1)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.plot(time[::1000,None],phi[:,::1000].T, lw=0.5, alpha=0.9);
plt.plot([0, time[-1]], [0,time[-1]*H/R], c="k", ls="--", lw=2)
# plt.plot([0, time[-1]], [0,0], c="k", ls="--", lw=1)
plt.plot(time[::1000],phi[:,::1000].mean(axis=0), lw=3, c="b", ls="-.");

plt.xlabel("Time")
plt.ylabel("Angle ($rad$)",labelpad=-8)

N_plots = len(i_plots)
for j, path in enumerate(i_plots):
    ax = plt.subplot(2,4,4+j+1)
    cm = plt.get_cmap("Greys")
    
#     for i in range(rang-1):
#         ax.plot(x[path][start:][i:i+2],y[path][start:][i:i+2],  c=cm(0.2+0.8*i/rang))
    plt.title(f"{(phi[path,start+rang] - phi[path,start])/2/np.pi:0.1f} turns", fontsize=9)
#     ax.scatter(x[path][start:][:rang][::st],y[path][start:][:rang][::st], 
#                c=-np.linspace(0.4,0.8,rang)[::st], s=0.9, cmap="gray", alpha=0.8)
    
    x1 = x[path][start:][:rang][::st1]
    y1 = y[path][start:][:rang][::st1]
    for i in range(len(x1)-1):
        ax.plot(x1[i:i+2],y1[i:i+2],  c=cm(0.3+0.7*i/len(x1)), lw=1,solid_capstyle='round')
    
    ax.scatter([0],[0], c="red", s=10)

    ax.set_aspect("equal")
    ax.set_xlim(-factor*R**0.5,factor*R**0.5)
    ax.set_ylim(-factor*R**0.5,factor*R**0.5)  
    ax.set_xticks(np.arange(-int(factor), int(factor)+EPS, 1)*R**0.5)
    ax.set_xticklabels(["$-2\sigma$","$-\sigma$","$0$","$\sigma$","$2\sigma$"])
    ax.set_yticks(np.arange(-int(factor), int(factor)+EPS, 1)*R**0.5)
    if j == 0:
        ax.set_yticklabels(["$-2\sigma$","$-\sigma$","$0$","$\sigma$","$2\sigma$"])
    else:
        ax.set_yticklabels([])

plt.savefig(f"./figuregen/paths-g0{g0}-ka{ka}-o0{o0}-b{b}.pdf", bbox_inches = 'tight', pad_inches = 0.08)