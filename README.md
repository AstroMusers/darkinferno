# darkinferno

[![arXiv](https://img.shields.io/badge/arXiv-2505.24070-b31b1b.svg)](https://arxiv.org/pdf/2505.24070)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15530200.svg)](https://doi.org/10.5281/zenodo.15530200)


Can captured dark matter leave a thermal fingerprint in a planet's core? We explore that question with spherical heat-conduction models for Earth, the Moon, and a Mars reference case. For Earth and the Moon, we calculate where the modeled temperature crosses a chosen threshold. Demonstrating that an actual planetary core melts would require more than that threshold crossing. Mars gives us a reference case for comparing source localization, the response to a finite outer boundary, radial temperature profiles, and heat budgets.

Putting most of the heating source inside the modeled sphere does not mean the temperature has stopped depending on the outer boundary. The relevant ratio is $x=R/r_\chi$, where $R$ is the modeled outer radius and $r_\chi$ is the characteristic width of the dark-matter heating region. Two fractions help us keep those questions separate.

- $f_R(x)$ is the fraction of the total heating power deposited inside $R$.
- $g_R(x)$ is the Dirichlet-domain convergence fraction, defined as the central warming with temperature fixed at $R$ divided by its formal value as that model boundary moves infinitely far away.

The first fraction reaches 90% at $x\simeq1.25$, but the second does not reach 90% until $x\simeq6.27$. At the first boundary, the model recovers about 50% of the formal Dirichlet-limit warming. The resulting factor 25.1 in transition mass is specific to the Gaussian source, constant properties, and fixed-temperature spherical boundary. It is not a whole-planet limit.

The analytic profile also yields a prospective radial-shape ratio from any three model temperatures, $[T(r_1)-T(r_2)]/[T(r_2)-T(r_3)]$. Under the constant-property model, the boundary temperature, total deposited power, and conductivity cancel exactly. Current seismic observations do not directly supply these temperatures, so this ratio is not used by the CMB particle boundary.

We also implement an internal-boundary calorimetry benchmark. In steady state, heat deposited inside a spherical boundary contributes to the outward power crossing it. The modern preferred 13--15 TW inference at Earth's 3,480 km core--mantle boundary becomes lower than the 49 TW surface ceiling above the geometric crossover at 0.50 GeV. We report the particle constraint only above 5 GeV because evaporation is not modeled. The 15 TW curve is a fixed-geophysical-background benchmark, not a directly measured or evolution-independent ceiling.

The `darkinferno.constraints` module propagates this benchmark through Gordion capture, finite-age annihilation equilibrium, Gaussian source containment, and local energy deposition. Gordion implements the self-contained multiregime capture calculation and tests its rates against Asteria v1.0 reference values. DarkInferno does not install or import Asteria. For the reference thermal-relic annihilation cross section, complete local deposition, and masses at or above the 5 GeV evaporation cutoff, the particle-constraint example produces a model-conditioned spin-independent DM--nucleon boundary. The 15 TW CMB curve reaches cross sections 14--21% below the matched 49 TW surface result at representative masses from 10 GeV to 1 TeV. This differential gain is distinct from a claim of globally new parameter space relative to every other particle search.

`examples/plot_particle_constraint.py` writes the boundary, drift-ceiling existence results, and capture-systematics figure together with their numerical CSV table. The systematics calculation varies halo density over 0.3--0.6 GeV cm$^{-3}$, halo speed over 230--310 km s$^{-1}$, and homogeneous bulk Fe fraction over 0.27--0.37. Gordion does not represent radial density or layered composition, so the Fe envelope is a bulk-composition scenario rather than a layered-Earth uncertainty.

The same analysis quantifies an existence condition at the adopted core-drift ceiling. Across 5 GeV--1 PeV, some scattering interval remains excludable if 0.68--2.36% of annihilation energy is deposited locally. At that same drift ceiling, the annihilation coefficient required for 90% capture--annihilation equilibrium is $2.1\times10^{-35}$--$3.2\times10^{-32}$ cm$^3$ s$^{-1}$. These thresholds do not describe robustness at every point on the reference exclusion boundary.

We provide two scripts. `darkinfernoradius.py` uses Gordion to calculate how many dark-matter particles become gravitationally trapped, then estimates the region where a steady temperature profile exceeds a chosen threshold. `darkinfernotimedependent.py` follows the temperature after a specified heat source is switched on. It uses a numerically stable spherical heat-diffusion solver and includes the energy absorbed during a phase change through an effective heat capacity.

The particle-limit workflow returns a deterministic boundary under its stated assumptions. A separate radial-temperature example estimates source power from temperature measurements.

## Radial-temperature inference

`darkinferno.inference.infer_heat_source_power()` fits a source-power posterior to radial temperature measurements using the steady-state solver. The result is conditional on the selected thermal scenario and dark-matter mass, with a uniform prior in log10 source power and independent Gaussian temperature errors. Run `python examples/infer_heat_source.py --smoke` for a short synthetic demonstration using assumed temperatures. These data illustrate the inference procedure and support no dark-matter constraint.

For `solve_steady_state()` and `steady_temperature_profile()`, we select the
scenario conductivity only when `conductivity_w_m_k=None`. Explicit overrides
must be finite and positive in $\mathrm{W\,m^{-1}\,K^{-1}}$. A zero override
raises `ValueError` instead of silently selecting the default. Both APIs share
the same conductivity selection and validation.

Sampled profiles require a one-dimensional `radius_m` array of finite values
from zero through the thermal-domain radius, including both endpoints. We reject
masked coordinates before array conversion and validate radii before solving.
Valid query order is preserved without sorting or filling coordinates. Empty
arrays retain the existing empty sampled profile and computed solver summary.

Body properties and thermal assumptions are selected through named scenarios. The Earth model uses a pure-iron melting estimate at the pressure of the inner-core boundary. The lunar model imposes a 57 K gap between its initial temperature and its threshold over a nominal 240 km domain. It is not a self-consistent radial temperature model or a dark-matter detector forecast. Its small domain is deliberately used to reveal when a local-core heating calculation becomes unreliable.


# Installation

DarkInferno requires Python 3. Install its dependencies from the repository root with the following command.

```bash
python -m pip install -r requirements.txt
```

## Local paths

Set `DARKINFERNO_PATH` to override the checkout used by our path helper. When it is unset, `paths.py` uses this repository directory. The helper exposes its local `data/` and `visuals/` directories, which Git ignores.

```bash
export DARKINFERNO_PATH=/path/to/darkinferno
```

# Example use

To calculate the radius where the steady temperature profile crosses the chosen threshold, assuming a dark matter mass of 1 TeV ($10^3$ GeV) and a cross section of $10^{-37}$ cm$^2$, run the following command.

```bash
python darkinferno/darkinfernoradius.py 3 -37
```

Select the central lunar inner-core scenario and write a structured JSON Lines result with the following command.

```bash
python darkinferno/darkinfernoradius.py 3 -37 \
	--scenario moon-inner-core \
	--output results/moon-steady.jsonl
```

The steady-state command also accepts `--halo-density`, `--halo-speed`, and `--deposition-efficiency`. Without `--output`, Earth runs append to `parameterscan.txt`. Lunar runs append to a scenario-qualified JSON Lines file.

To compute the temperature at a radius of 400 km as a function of time, assuming $10^3$ GeV dark matter mass, 5 TW heat injection, $T_0$ = 5500 K, and k = 100 W/m/K, you can run the code below.

```bash
python darkinferno/darkinfernotimedependent.py 3 5 5500 100
```

For a short central-scenario lunar evolution with a 0.01 TW heat injection, run the following command.

```bash
python darkinferno/darkinfernotimedependent.py 3 0.01 1754 40 \
	--scenario moon-inner-core \
	--end-time-years 1e8 \
	--time-step-years 1e5 \
	--radial-cells 300 \
	--output results/moon-time.json
```

The time-dependent total deposited power is supplied directly and does not calculate capture. Its first positional argument controls the width of the captured dark-matter distribution. The solver represents melting by spreading the required phase-change energy over a temperature interval set by `--latent-width-k`, which defaults to 50 K. Use `--time-step-years`, `--radial-cells`, and `--latent-width-k` to test sensitivity to the time step, radial grid, and artificial melting interval. The lunar threshold-crossing time is not a physical result because it changes strongly when that interval changes.

# Scenarios

The available scenarios include the following choices.

- `earth-inner-core` uses a 5,500 K reference state and the 6,230 K pure-iron ICB melting estimate of [Anzellini et al. (2013)](https://doi.org/10.1126/science.1233514).
- `moon-inner-core` uses a 1,754 K reference state and a 1,811 K elemental-iron melting reference.
- `mars-core-reference` uses the approximately 1,830 km InSight seismic core scale from [Stahler et al. (2021)](https://doi.org/10.1126/science.abi7730) with imposed 2,000 K and constant-property assumptions. It is not a Martian melting model.

The lunar mean radius follows [Archinal et al. (2018)](https://doi.org/10.1007/s10569-017-9805-5), and its mass follows the gravity analysis of [Konopliv et al. (2001)](https://doi.org/10.1006/icar.2000.6573). The 240 km thermal-domain scale is informed by [Weber et al. (2011)](https://doi.org/10.1126/science.1199375). The uniform 1,754 K state is an imposed benchmark and is not a temperature evaluated at 240 km from the same interior model. The 1,811 K reference is likewise approximate because lunar pressure raises iron's melting temperature while sulfur and other light elements lower it. The Mars density, composition, temperature, and transport properties are reference assumptions rather than direct observations. Each structured result records its scenario, provenance, assumptions, and warnings.

# Heat-loss data and benchmarks

The `darkinferno.observations` module records heat-loss data and preserves how each value was obtained. The whole-body surface budgets and internal CMB benchmark differ in how they were inferred and which physical assumptions they require.

- Earth has a measurement-derived global heat-loss interval of 45--49 TW from [Davies (2013)](https://doi.org/10.1002/ggge.20271).
- Earth's core--mantle boundary has a preferred modern 13--15 TW outward heat-flow range from [Frost et al. (2022)](https://doi.org/10.1029/2021GC009764), within a broader historical literature reviewed by [Lay et al. (2008)](https://doi.org/10.1038/ngeo.2007.44). `RadialHeatFlowConstraint.total_dm_power_ceiling_w()` corrects its upper endpoint for the Gaussian power fraction deposited inside the core. Treating 15 TW as a DM benchmark holds the conventional present-day geophysical background fixed.
- The Moon has a regionally inferred mantle power of 0.28--0.41 TW from Apollo 15 and 17 analyses by [Siegler and Smrekar (2014)](https://doi.org/10.1002/2013JE004453). It is not a global surface measurement.
- Mars has a model-inferred surface heat flux of 14--25 mW/m$^2$, equivalent to 2.0--3.6 TW for the adopted radius, from [Parro et al. (2017)](https://doi.org/10.1038/srep45629). InSight HP3 did not reach the depth required for its intended direct planetary heat-flow measurement.

Assigning the entire measured Earth surface luminosity to dark matter gives a conservative whole-body ceiling on anomalous steady power. The inferred CMB luminosity instead defines a fixed-background benchmark because central dark-matter heating can alter the conventional core energy terms used to infer it. These intervals are not interchangeable confidence regions. A statistical CMB constraint requires coupled thermal evolution and a likelihood for conventional heating and transport.

# Moon and Mars model status

Lunar output is an estimate under stated assumptions, not a dark-matter exclusion limit. The following limitations are particularly important.

- Gordion treats the body as having homogeneous whole-body elemental abundances and does not model radial density or composition.
- The current lunar elemental abundances are an assumed, normalized bulk-silicate mixture. They have not been validated as a whole-Moon composition.
- The imposed 1,754 K temperature at the 240 km boundary does not come from a self-consistent radial temperature model of the Moon.
- The calculation assumes that annihilation removes captured particles as quickly as capture supplies them and that the selected energy fraction becomes local heat. The time required to reach this balance is not calculated.
- The escape of thermalized dark matter from the Moon and the time required for dark matter to exchange energy with lunar material are not calculated.
- The Moon's 2.38 km/s escape speed is below the range used to calibrate Gordion's reflection treatment.
- Source containment does not guarantee a boundary-independent temperature. At 1 TeV, the lunar domain contains more than 99.9% of the Gaussian source but recovers only 70.8% of the central warming obtained with an infinitely distant boundary.
- At low mass, the Gaussian heating region can extend beyond the modeled core. Quantitative work must calculate gravity and the dark-matter distribution throughout the body and follow heat transport through the whole core or body.
- The time-dependent lunar threshold-crossing time is rejected as a physical result because it is strongly sensitive to the artificial temperature interval used to represent melting.

A quantitative lunar constraint requires a validated whole-Moon composition, a capture calculation tested at the Moon's low escape speed, particle escape and capture--annihilation balance calculations, a self-consistent radial temperature profile, realistic melting and heat transport, and a model for a measurable signal such as seismic structure or surface heat flow.

The Mars reference similarly does not provide a particle exclusion. Its InSight-scale domain is observationally motivated, but the fixed boundary temperature and material properties are imposed. Current Martian heat flow is model-inferred, and current seismology constrains elastic structure rather than directly measuring the three temperatures needed for the radial fingerprint.

# Figures

The standalone examples generate publication-resolution figures from the solver APIs. They write 300 dpi PNG files by default with the following commands.

```bash
python examples/plot_earth.py
python examples/plot_moon.py
python examples/plot_comparison.py
python examples/plot_particle_constraint.py
python examples/plot_intermediates.py
python examples/plot_convergence.py
```

These figures use white backgrounds with black axes by default. Pass
`--typeplotback dark` for black backgrounds with white axes. Dark figure names
have a `_dark` suffix, so existing light figures are not reused or overwritten.
The full-results launcher forwards the same option to every plotting script.

By default, the scripts write to `figures/`. Use `--output-dir PATH` to select another location, or `--typefileplot pdf` to write vector PDFs instead of PNGs. Every output contains one plot. Its two-digit filename prefix gives the order of that variable within the Earth, Moon, comparison, intermediate-variable, or convergence sequence. Before calculating a plot, each script checks whether that output already exists. Existing outputs are reported and skipped independently. The Earth and Moon sequences contain steady radial and illustrative time-dependent temperature plots. Their shared threshold response appears only in the comparison sequence. Mars joins the body-independent spatial-heating, finite-domain, radial-fingerprint, and observed-heat-budget figures. The comparison sequence also contrasts Earth's surface and core--mantle radial calorimeters. The scale plots mark where less than 90% of the Gaussian power lies inside the modeled region. The convergence sequence tests the radial grid, time step, and artificial melting interval.

All executable scripts accept `--typeverb 0`, `--typeverb 1`, or `--typeverb 2` for critical-only, nominal, or verbose narration. The default is nominal. Verbose thermal evolutions also report regular time-step progress.

Run any script with `--help` for a description of its arguments.

Run the complete DarkInferno results workflow, including its Earth and Moon thermal models, rocky-body comparisons, convergence plots, intermediate variables, particle-constraint boundaries, and numerical table, with this command:

```bash
python examples/reproduce_cappiello_daylan_2025.py
```

The workflow writes figures to `visuals/cappiello_daylan_2025/` by default. The 2025 paper used Asteria for capture. This reproducer uses Gordion, whose implementation is tested against Asteria v1.0 reference capture rates.

# Paper

The ApJ manuscript is in `paper/` and uses AASTeX 7. Build instructions and the bibliography are provided alongside `paper/main.tex`.

# Tests

Run the command-line and physics regression tests with the following command.

```bash
python -m unittest discover -s tests -v
```


# Acknowledgements

The development of DarkInferno has been supported by the McDonnell Center for Space Sciences at Washington University in St. Louis.
