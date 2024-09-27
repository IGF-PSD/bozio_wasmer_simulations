# Simulations of reforms to employer contributions from the "Bozio-Wasmer" report

This directory contains all the programs used to simulate the reforms presented in the "Bozio-Wasmer" report and their outcomes.

## Objectives

This package is built to quantify different outcomes of employer contributions :
* Empirically :
  - To quantify budget costs (static and "first-round" after application of the employment effects) associated with each reform
  - To estimate employment effects of the reform
* Theorically :
  - To estimate incentives to increase wages
  - To estimate the impact of adding the "prime de partage de la valeur" to the basis for reducing contributions

## Organization

The repository is organized as follow :
* the `data` folder contains the data used in the simulations:
  - the inventory of [Rural Revitalization Zones (ZRR)](https://www.data.gouv.fr/fr/datasets/zones-de-revitalisation-rurale-zrr/)
  - the inventory of [Zones de Restructuration de la Défense (ZRD)](https://www.data.gouv.fr/fr/datasets/les-zones-de-restructuration-de-la-defense-zrd/)
  - distributions of individuals by salary bracket
* the `docs` folder contains all the package documentation and assumptions underlying the simulation model
* the `logs` folder contains the logging files associated with the simulators
* the `notebooks` folder contains illustrative notebooks for reproducing report results
* the `outputs` folder contains program outputs.
* the `parameters` folder contains default simulation parameters.
* the `scripts` folder contains a script for running simulations of current contributions and reductions, and of the reform scenarios listed in the `config.yaml` file at the root.

## Installation

### Package and dependencies

```bash
git clone <toolbox_repo_url>
pip install -e igf_toolbox_python

git clone <repo_url>
pip install -e bozio_wasmer_simulations
```

The package is then usable as any other python package.

### Parametrisation

File in the `config.yaml` file :
```yaml
CASD :
  PROJET : "INGENFI"
``` 

### Documentation

To visualize the documentation :

```
mkdocs build --port 5000
```

## Usage

Here's an example of how to use the functions in the package:

```python
from bozio_wasmer_simulations import CaptationMarginaleSimulator

# Create a simulator object
simulator = CaptationMarginaleSimulator()

# Define a dictionary of reform parameters
reform_params = {
    'TYPE': 'fillon',
    'PARAMS': {
        'PLAFOND': 2.7,
        'TAUX_50_SALARIES_ET_PLUS': 0.35,
        'TAUX_MOINS_DE_50_SALARIES': 0.354
    }
}

# Simulate a reform
data_simul = simulator.simulate_reform(name='my_reform', reform_params=reform_params, year=2022, simulation_step_smic=0.1, simulation_max_smic=4)
``` 

## License

The package is licensed under the MIT License.