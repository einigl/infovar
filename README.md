# InfoVar

[![PyPI version](https://badge.fury.io/py/infovar.svg)](https://badge.fury.io/py/infovar)
[![Documentation status](https://readthedocs.org/projects/infovar/badge/?version=latest)](https://infovar.readthedocs.io/en/latest/?badge=latest)
![](./coverage.svg)

The `infovar` Python package provides tools to efficiently study the informativity of variables on data of interest.


## Context

The informativity of a variable or set of variables is defined here as the ability of these variables, if known, to reduce the uncertainty we have about a quantity of interest. This uncertainty can be defined in several ways, for example in the sense of Shannon's information theory.

This is a ubiquitous problem in science in general, with very concrete applications in climatology, economics, psychology, sociology, and astrophysics, to name a few. Consequently, `InfoVar` has been designed to be very general.

This package provides tools for quantifying the statistical dependence (e.g., mutual information, but other metrics are available) between continuous numerical data and estimating the associated error as well as the influence of the latter on the order of variables in terms of importance.

## Installation

*(optional)* Create a virtual environment and activate it:

```shell
python -m venv .venv
source .venv/bin/activate
```

**Note 1:** to deactivate the virtual env :

```shell
deactivate
```

**Note 2:** To delete the virtual environment:

```shell
rm -r .venv
```

### From PyPI (recommanded)

To install `infovar`:

```shell
pip install infovar
```

### From local package

To get the source code:

```shell
git clone git@github.com:einigl/infovar.git
```

To install `infovar`:

```shell
pip install -e .
```


## Get started

To get started, check out the Jupyter notebooks provided in the `examples` folder.


## Tests

To test, run:

```shell
pytest --cov && coverage-badge -o coverage.svg -f
```

## Documentation

```bash
cd docs
sphinx-apidoc -o . ../infovar
make html
```

Outputs are in `docs/_build/html`.


## Features

### Statistics

In this project, we propose to measure the statistical dependence of variables based on the mutual information. Other metrics can also be used, such as the conditional differential entropy, which is closely related to mutual information, or canonical correlation coefficient.

Mutual information and conditional differential entropy are estimated nonparametrically using [Greg Ver Steeg's implementation](http://www.isi.edu/~gregv/npeet.html). More details are given in the `assessment` directory, which evaluates the properties of each available statistics and provides further mathematical context and references.

If you're interested in other metrics, it's possible to add and use them.

### Uncertainty on estimations

Uncertainty in the estimation of the above statistics can arise from various sources:
- the variance of the estimator,
- statistical fluctuations of samples from the same distribution.

To account for these uncertainties and to be able to compare different values properly, we propose implementations of several approaches, based on bootstrapping or subsampling.

### Estimation for different range of values

The heart of `InfoVar` lies in the fact that the informativity of a variable on a quantity of interest can vary according to the selected range of value of this quantity.

For example, if we're interested in house prices in California (see `examples/california-housing`), among a set of variables, geographical location (latitude, longitude) appears to be the most important pair of variables. However, if we restrict ourselves to the 10% most expensive homes, it appears that the number of rooms in the house becomes most useful. This type of observation is important, for example, from a data analysis point of view, but also in a variable selection context.

More generally, taking into account these variations as a function of ranges of values of the variable of interest enables more refined analysis of phenomena. To help you understand, here are a few examples of possible applications.

Determining factors on ...

**... student's grades as a function of the grade obtained.**.
- *Data of interest:* student marks on an exam.
- *Variables:* time spent working at home, missed lessons, parents' income, etc.

**... number of species in forests.**
- *Data of interest:* number of species.
- *Variables:* forest age, humidity, distance to nearest town, number of visitors per day, etc.

**... on the number of medals a country has won at the Olympic Games.**
- *Data of interest:* number of medals won by each country in each of the last 10 editions of the games.
- *Variables:* amount invested by the national Olympic committee, population, per capita income, unemployment rate, etc.

It is also possible to perform the same analysis, but according to the value range of *another* variable.

**...the average annual temperature in a city as a function of altitude.**
- *Data of interest:* average temperature.
- *Variables:* duration of sunshine, percentage of vegetated land, altitude.

**... the number of medals won by a country at the Olympic Games as a function of its population.**
- *Data of interest:* number of medals won by each country in each of the last 10 editions of the games.
- *Variables:* amount invested by the national Olympic committee, population, per capita income, unemployment rate.

The `InfoVar` allows you to perform sensitivity analysis in two ways:
1. Define rigid intervals for the data that varies (example: houses priced below $150k, between $150 and $350k and above $350k).
2. Define a sliding window and calculate the evolution of the statistics almost continuously.

In case 1 (discrete case), the `DiscreteHandler` class provides all the important functions for calculating, storing and accessing results. In case 2 (continuous case), the `ContinuousHandler` class is used. The notebooks in `examples` give an example of the use of each of these two classes.


## Associated packages

[**A&A papers repository**](https://github.com/einigl/informative-obs-paper): Reproduce the results in Einig et al. (2024, 2025)

[**IRAM 30m EMIR informative observables**](https://github.com/einigl/iram-30m-emir-obs-info): Informativity of molecular lines to estimate astrophysical parameters.


## References

[1] Einig, L & Palud, P. & Roueff, A. & Pety, J. & Bron, E. & Le Petit, F. & Gerin, M. & Chanussot, J. & Chainais, P. & Thouvenin, P.-A. & Languignon, D. & Bešlić, I. & Coudé, S. & Mazurek, H. & Orkisz, J. H. & G. Santa-Maria, M. & Ségal, L. & Zakardjian, A. & Bardeau, S. & Demyk, K. & de Souza Magalhẽs, V. & Javier R. Goicoechea & Gratier, P. & V. Guzmán, V. & Hughes, A. & Levrier, F. & Le Bourlot, J. & Darek C. Lis & Liszt, H. S. & Peretto, N. & Roueff, E & Sievers, A. (2024).
**Quantifying the informativity of emission lines to infer physical conditions in giant molecular clouds. I. Application to model predictions.** *Astronomy & Astrophysics.*
10.xxxx/xxxx-xxxx/xxxxxxxxx.

[2] Einig, L et al (2024, in prep.).
**Quantifying the informativity of emission lines to infer physical conditions in giant molecular clouds. II. Training robust models from selected observations.** *Astronomy & Astrophysics.*
10.xxxx/xxxx-xxxx/xxxxxxxxx.
