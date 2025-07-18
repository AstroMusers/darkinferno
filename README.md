# darkinferno

[![arXiv](https://img.shields.io/badge/arXiv-2505.24070-b31b1b.svg)](https://arxiv.org/pdf/2505.24070)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15530200.svg)](https://doi.org/10.5281/zenodo.15530200)


DarkInferno is a package for computing the temperature profile within the core of rocky planets, including the Earth, due to heating from the annihilation of captured dark matter.

`DarkInferno` provides two scripts. `darkinfernoradius.py` computes the steady-state temperature profile for given values of the dark matter mass and dark matter-nucleon scattering cross section, and returns the radial distance to which the inner core is melted. `darkinfernotimedependent.py` computes the temperature profile as a function of time, assuming a uniform initial temperature. It then outputs the temperature at 400 km, as a function of time, as a list. The framework uses [ASTERIA](https://zenodo.org/records/8150110) to model dark matter capture in the Earth.


# Example use

Refer to the [jupyter notebook](https://github.com/AstroMusers/darkinferno/blob/main/darkinferno/darkinferno_example.ipynb) demonstrating the usee of both scripts.

To compute the melting radius in steady-state, assuming a dark matter mass of 1 TeV ($10^3$ GeV) and a cross section of $10^{-37}$ cm^2, you can run the code below.

```
python darkinfernoradius.py 3 -37
```

To compute the temperature at a radius of 400 km as a function of time, assuming $10^3$ GeV dark matter mass, 5 TW heat injection, $T_0$ = 5500 K, and k = 100 W/m/K, you can run the code below.

```
python darkinfernotimedependent.py 3 5 5500 100
```


# Acknowledgements

The development of DarkInferno has been supported by the McDonnell Center for Space Sciences at Washington University in St. Louis.
