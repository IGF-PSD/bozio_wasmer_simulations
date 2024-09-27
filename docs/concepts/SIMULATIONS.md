# Methodology underlying the estimation of the effects of reforms

The main goal of this package is to simulate different payroll tax reductions and compare labor cost before and after changing the tax reduction rules applicable in 2023.

## Data

Simulations are iterated on the "Base Tous Salariés, Postes" of the  Déclarations Annuelles de Données Sociales (DADS, see [documentation](https://www.insee.fr/fr/metadonnees/source/serie/s1998)).

## Simulation module

We use version 166.1.5 of the [`openfisca-france`](https://github.com/openfisca/openfisca-france) package to simulate all the employers contributions and payroll tax reductions applicable to each employee. These simulations make it possible to deduce a labor cost associated with each individual. 
A reform module is then used to simulate a set of alternative payroll tax reductions and deduce a new labor cost. These new reductions can take the form of :
- a "Fillon" payroll tax reduction with parameters $\left( rate_{max}, ceiling \right)$ : \(rate = \frac{rate_{max}}{ceiling -1} \times \left( ceiling \times \frac{Gross \;Minimum \; Wage}{Gross \; Salary} -1 \right)\)
- a "Convexe Fillon" payroll tax reduction with parameters $\left( rate_{max}, ceiling, exponent \right)$ : \(rate = rate_{max} \times \left[ \frac{ceiling \times \frac{Gross \;Minimum \; Wage}{Gross \; Salary} -1}{ceiling -1}  \right]^{exponent}\)
- a "Family" or "Illness" payroll tax reduction with parameters $\left( rate_{max}, ceiling \right)$: \(rate = \mathbb{1}_{Gross \; Salary \leq ceiling} \times rate_{max}\)


## Perimeter 

With the default parameters, simulations are iterated on the main employment period [1] of private sector employees (excluding those of "particuliers employeurs" - individual employers), aged 18 to 64 and employed in mainland France. These observations are then re-weighted on the basis of ACOSS distribution of employees by gross wage bins.

The following tax reductions have been simulated:
- TODE, 
- Apprentice, 
- Trainee, 
- Young Innovative Company (JEI),
- Defense Restructuring Zone (ZRD),
- Rural Revitalization Zone (ZRR). 

However, the following tax reduction is ignored:
- "Zone Franche Urbaine (ZFU)"

By default, employees to whom one of these tax reductions applies are not affected by a simulated reform.

## Warnings

Compared with version 166.1.5 of the [`openfisca-france`](https://github.com/openfisca/openfisca-france) package, `chomage_employeur` contributions have been modified so that they always amount to 4.5% of the social security contribution base.
> This correction, currently implemented in this package's code, will be included in a future PR on `openfisca-france`. Hence, using a later version of `openfisca-france` may lead to over-corrections.


***

[1]: *the continuum of employment contracts with the highest net salary for each employee*
