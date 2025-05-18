#!/usr/bin/python

import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt

year=3.15*10**7
G = 6.67*10**(-11)
rho = 13
kB = 1.38*10**(-23)
GeV = 1.78*10**(-27)

resfactor=10.0
kfactor=1.0
distancetomatchtime=2.0*kfactor
dt=resfactor**2*10.0*year/distancetomatchtime
dr=resfactor*400.0

endtime=4500000000*year

mdm=10**float(sys.argv[1])*GeV
heatflow=float(sys.argv[2])*1000000000000.0
t0=float(sys.argv[3])
k0=float(sys.argv[4])
if k0>100:
    dt=dt*100/k0
alpha0=24*10**(-6)
rhoc=100/alpha0
alpha0=k0/rhoc
print("k = "+str(k0)+", T0 = "+str(t0))

def k(temp):
    return k0

def scaleradiussqr(temp):
    return 1/(G*(4/3)*np.pi*(rho*1000)*mdm/(kB*temp)/2)

def n(r,temp):
    return np.exp(-r**2/scaleradiussqr(temp))

def qnormalization(temp):
    return 4*np.pi*np.sqrt(np.pi/2)/(8*(1/scaleradiussqr(temp)**(3/2)))

def q(r,temp):
    return heatflow*n(r,temp)**2/qnormalization(temp)

def testnormalization(r,temp):
    return q(r,temp)*4*np.pi*r**2

# this factor encodes the latent heat
def cfactor(t):
    width=500
    return 247000/(np.sqrt(2*np.pi)*width)*np.exp(-(t-10000)**2/(2*width**2))
    return 0

yeararray=[]
temperaturearray=[]

ualltime=np.full(int(1200000/dr),t0)
radii=[dr*i for i in range(len(ualltime))]
for i in range(int(endtime/dt)):
    if ualltime[int(400000/dr)]>10000:
        print("Finished")
        print("Temperature at 400 km exceeds 10000 K after "+str(i*dt/year)+" years")
        with open("timehistogram5tw.txt","a") as outfile:
            outfile.write(str(i*dt/year)+"\n")
        break
    radialarray=[]
    temparray=ualltime
    for j in range(int(1200000/dr)-1):
        r=(j+1)*dr
        if j==0:
            endpoint=temparray[j]
        else:
            endpoint=temparray[j-1]

        testfactor=2
        tnew=temparray[j]+1/((rhoc + rho*1000*cfactor(temparray[j]))*r**2)*(dt/dr)*((r**2*(k(temparray[j])-k(endpoint))/(dr) + 2*r*k(temparray[j]))*(temparray[j+1]-temparray[j]) + k(temparray[j])*r**2*(temparray[j+1]-2*temparray[j]+endpoint)/dr) + q(r,min(10000,temparray[0]))*dt/(rhoc + rho*1000*cfactor(temparray[j]))

        radialarray.append(tnew)
    radialarray.append(t0)
    ualltime=radialarray

    if i % 1000==0:
        yeararray.append(i*dt/year)
        temperaturearray.append(ualltime[int(400000/dr)])
        print(i*dt/year)

yeararrayold=[]
dtold=5.0**2*10.0*year/(2.0*1.0)
for i in range(int(1000000000*year/dtold)):
    if i % 1000==0:
        yeararrayold.append(i*dtold/year)

with open("temperaturearray.txt","a") as outfile:
    for i in range(len(temperaturearray)):
        outfile.write(str(i*dt/year)+" ")
    outfile.write("\n")
    for temp in temperaturearray:
        outfile.write(str(temp)+" ")
    outfile.write("\n") 
