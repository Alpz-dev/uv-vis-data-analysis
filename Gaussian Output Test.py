#! /usr/bin/env pytho
import numpy as np
import math
from matplotlib import pyplot as plt
from data import *


def read_es(file):
    energies=[]
    os_strengths=[]
    for line in file:

        if " Excited State " in line:
            print(line)
            energies.append(float(line.split()[6]))
            os_strengths.append(float(line.split()[8][2:]))
            print(line)
    return energies,os_strengths

def abs_max(f,lam,ref):
    a=1.3062974e8
    b=f/(1e7/3099.6)
    c=np.exp(-(((1/ref-1/lam)/(1/(1240/args.sd)))**2))
    return a*b*c
def spectrum(E,osc,sigma,x):
    gE=[]
    for Ei in x:
        tot=0
        for Ej,os in zip(E,osc):
            tot+=os*np.exp(-((((Ej-Ei)/sigma)**2)))
        gE.append(tot)
    return gE
def mpl_plot(xaxis,yaxis):
    plt.scatter(xaxis,yaxis,s=2,c="r")
    plt.plot(xaxis,yaxis,color="k")
    plt.xlabel("Energy (nm)")
    plt.ylabel("$\epsilon$ (L mol$^{-1}$ cm$^{-1}$)")
    return

energies, osc = read_es(open("test data/NiBT_sp_tddft_01.log"))

def spectrum(E,osc,sigma,x):
    gE=[]
    for Ei in x:
        tot=0
        for Ej,os in zip(E,osc):
            tot+=os*2*np.exp(-((((Ej-Ei)/sigma)**2)))
        gE.append(tot)
    return gE

x=np.linspace(0,1500, num=1000, endpoint=True)
sigma=20

gE=spectrum(energies,osc,sigma,x)



fig,ax=plt.subplots(figsize=(6,4))
data = import_data("test data/Ni-BT_3-4-2020.txt")


data.scale(0.5).plot(1)

ax.plot(x,gE,"--r")


for energy,osc_strength in zip(energies,osc):
    ax.plot((energy,energy),(0,osc_strength*2.5),c="r")

ax.set_xlabel("Wavelength (nm)",fontsize=16)
ax.xaxis.set_tick_params(labelsize=14,width=1.5)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
ax.set_xlim(300,800)
ax.set_ylim(0, 0.75)
ax.set_ylabel("Abs/Osc. Strength (AU)",fontsize=16)
plt.tight_layout()
plt.xlim((300, 800))
plt.ylim((0, 0.75))

plt.xlabel("Wavelength (nm)", fontweight = "bold", fontsize = 14)
plt.ylabel("Abs/Osc. Strength", fontweight= "bold", fontsize = 14)
plt.xticks([300, 400, 500, 600, 700, 800], weight = "bold", fontsize = 12)
plt.yticks([])
plt.legend(["A", "B"], loc = "best", fontsize = 14)

plt.show()




plt.show()


