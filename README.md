# Introduction
DarkInferno is a package for computing the temperature profile of the Earth's inner core due to heating from dark matter annihilation.

DarkInferno consists of two scripts. One, darkinfernoradius.py, computes the steady-state temperature profile for given values of the dark matter mass and dark matter-nucleon scattering cross section, and returns the radial distance to which the inner core is melted. The other, darkinfernotimedependent.py, computes the temperature profile as a function of time, assuming a uniform initial temperature. It then outputs the temperature at 400 km, as a function of time, as a list. DarkInferno uses the [ASTERIA](https://zenodo.org/records/8150110) package to model dark matter capture in the Earth.


# Example use

Refer to the [jupyter notebook](https://github.com/AstroMusers/darkinferno/blob/main/darkinferno/darkinferno_example.ipynb) demonstrating the usee of both scripts.

To compute the melting radius in steady-state, assuming a dark matter mass of 1 TeV ($10^3$ GeV) and a cross section of $10^-37$ cm^2, you can run the code below.

```
python darkinfernoradius.py 3 -37
```

To compute the temperature at a radius of 400 km as a function of time, assuming $10^3$ GeV dark matter mass, 5 TW heat injection, $T_0$ = 5500 K, and k = 100 W/m/K, you can run the code below.

```
python darkinfernotimedependent.py 3 5 5500 100
```


# Acknowledgement

The development of DarkInferno has been supported by the McDonnell Center for Space Sciences at Washington University in St. Louis.
