# Bayesian Nonparametric Vector Autoregressive Models via a Logit Stick-breaking Prior: 
## an Application to Child Obesity

Code for reproducing the simulation studies.

## How to install

### Prerequisites

The build system relies on

- `python3-dev`
- `pybind11`
- `make`
- `protobuf` and `protoc`
- `gsl`

Which should all be installed on your machine.


You should also 

1) Download the stan-math library somewhere in your computer and export and environmental variable `STAN_ROOT_DIR` with the path of the installation.

2) After installing `gsl`, export the `GSL_HOME` environmental variable with the path of the root of GSL.


### Installation

Then, from a terminal, simply execute

```
make compile_protos
make generate_pybind
```

Warning: you might need some tweaking with the makefile! It is very specific for Mac-OS, but should work as it is also on Linux. I hope it works on Windows as well but would bet my money against it...

### Reproducing the simulation studies

Just run the three jupyter notebooks.
