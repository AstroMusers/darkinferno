#!/usr/bin/python

# This script uses Asteria to compute the dark matter capture rate in Earth, then solves the spherical heat equation for an isothermal dark matter profile to compute the radius out to which the core reaches 10000 K
# The solution assumes the system is in steady state
# Because the dark matter distribution depends on the temperature of the Earth, this script iterates the solution 10 times, solving for the temperature profile assuming a baseline fiducial dark matter distribution, then updating the dark matter distribution and solving again

import asteria
import sys
import numpy as np
import scipy
from sympy.solvers import solve
from sympy import Symbol
from scipy.optimize import root_scalar

G=6.67*10**-11
kgperm3=1000
rho=13*kgperm3
k=100
Tboundary=5500
Tcore=5500
kB=1.38*10**-23
GeV=1.78*10**-27
joule=1.602*10**-10#in GeV

# inner core-outer core boundary
icradius=1200000
# radius within which Earth's core might actually be molten
innerradius=400000

# command line args: DM mass in GeV and heat injection (mass capture rate, assuming equilibrium) in terawatts
mdm=10**float(sys.argv[1])*GeV
sigma=10**float(sys.argv[2])

print("mdm = "+str(mdm/GeV)+", sigma = "+str(sigma))

####################
#setting up asteria capture package
REARTHasteria=6.38*10**6
MEARTHasteria=5.97*10**24
rhodmasteria=0.4
vasteria=270000
mnucasteria=0.94
asteria.set_elements([[0.32,55.8],[0.29,16],[0.15,24],[0.14,28],[0.017,40],[0.015,27]])
captureasteria=np.vectorize(asteria.multicapture)
capturerate=captureasteria(REARTHasteria,MEARTHasteria,rhodmasteria,vasteria,mnucasteria,sigma,mdm/GeV)

masscapturerate=capturerate*mdm/GeV*joule
heatflow=masscapturerate
print("mass-energy capture rate is "+str(heatflow)+" watts")

#done with Asteria section
####################

iter=1
while iter<10:
    iter+=1
    rx=np.sqrt(3*kB*Tcore/(2*np.pi*G*rho*mdm))

    def densityintegrand(x):
        return np.exp(-x**2/rx**2)**2*4*np.pi*x**2
    densitynormalization=scipy.integrate.quad(densityintegrand,0,5*rx)[0]/heatflow

    def tempprofile(x,const):
        return np.sqrt(2*np.pi)*rx**3 * scipy.special.erf(np.sqrt(2)*x/rx)/(16*densitynormalization*k*x) + const

    c=Symbol('c')
    constant=float(solve(tempprofile(icradius,c)-Tboundary,c)[0])

    def truetempprofile(x):
        return tempprofile(x,constant)

    def splicedtempprofile(x):
        if x<icradius:
            return truetempprofile(x)
        else:
            return Tboundary

#    print(splicedtempprofile(innerradius))
    Tcore=(Tcore+splicedtempprofile(innerradius/100000))/2

def rootfunc(r):
    return truetempprofile(r)-10000
if truetempprofile(1)<10000:
    root=1
else:
    root=root_scalar(rootfunc,bracket=[1,1200000],method='brentq').root

with open('parameterscan.txt', 'a') as f:
    f.write(str(sys.argv[1])+" "+str(sys.argv[2])+" "+str(root)+"\n")
