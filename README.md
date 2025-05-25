DarkInferno is a package for computing the temperature profile of the Earth's inner core due to heating from dark matter annihilation.

DarkInferno consists of two scripts. One, darkinfernoradius.py, computes the steady-state temperature profile for given values of the dark matter mass and dark matter-nucleon scattering cross section, and returns the radius out to which the inner core is melted. The other, darkinfernotimedependent.py, computes the temperature profile as a function of time, assuming a uniform initial temperature. It then outputs the temperature at 400 km, as a function of time, as a list.

DarkInferno uses the ASTERIA Python package to model dark matter capture in the Earth.

This repository contains both scripts, as well as a jupyter notebook demonstrating their usage. Two brief examples are shown below:


To compute the melting radius in steady-state, assuming a dark matter mass of 1 TeV (10^3 GeV) and a cross section of 10^-37 cm^2, run

```
python darkinfernoradius.py 3 -37
```

To compute the temperature at a radius of 400 km as a function of time, assuming 10^3 GeV dark matter mass, 5 TW heat injection, T0 = 5500 K, and k = 100 W/m/K, you can run the code below.

```
python darkinfernotimedependent.py 3 5 5500 100
```
