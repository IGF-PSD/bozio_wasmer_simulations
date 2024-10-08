# Importation des modules
# Modules de base
import json
import os
from pathlib import Path
from typing import List, Union

import pandas as pd
# Openfisca
# Openfisca-france
from openfisca_france import FranceTaxBenefitSystem
# Importation de modules ad hoc
from bozio_wasmer_simulations.simulation.empirical.preprocessing import \
    preprocess_simulated_variables
from bozio_wasmer_simulations.simulation.empirical.reform import \
    create_and_apply_structural_reform_ag
# Modules de package
from bozio_wasmer_simulations.simulation.theoretical.base import TheoreticalSimulator
from tqdm import tqdm

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

# Importation des paramètres de simulation
with open(
    os.path.join(FILE_PATH.parents[3], "parameters/simulation.json")
) as json_file:
    params = json.load(json_file)


# Classe permettant de simuler la réintégration de la PPV à l'assiette des allègements
class PPVReintegrationSimulator(TheoreticalSimulator):
    """
    A class for simulating the reintegration of the PPV in the base for exemptions.

    This class inherits from TheoreticalSimulator and provides methods for initializing a case with the PPV exempted or reintegrated,
    simulating variables of interest, simulating a reform, simulating multiple reforms, calculating the implicit tax rate,
    and building the dataset.

    Attributes:
        logger (logging.Logger):
            A logger for logging messages.

    """

    # Initialisation
    def __init__(
        self,
        log_filename: os.PathLike = os.path.join(
            FILE_PATH.parents[3], "logs/ppv_simulation.log"
        ),
    ) -> None:
        """
        Constructs all the necessary attributes for the ReformSimulation object.

        Args:
            log_filename (os.PathLike, optional): The path to the log file. Defaults to os.path.join(FILE_PATH.parents[3], 'logs/ppv_simulation.log').

        """
        # Initialisation du simulateur
        super().__init__(log_filename=log_filename)

    # Fonction auxiliaire d'extraction des noms de scénarios
    def scenarios_names(self, scenarios: Union[str, List[str], dict]) -> List[str]:
        """
        Extracts the names of the scenarios.

        Args:
            scenarios (Union[str, List[str], dict]):
                The scenarios to extract the names from.

        Returns:
            (List[str]): The names of the scenarios.

        Raises:
            ValueError: If the type of scenarios is not 'dict', 'list', or 'str'.
        """
        # Extraction des noms des scenarios
        if isinstance(scenarios, str):
            scenarios_names = [scenarios]
        elif isinstance(scenarios, dict):
            scenarios_names = [name.lower() for name in scenarios.keys()]
        elif isinstance(scenarios, list):
            scenarios_names = [name.lower() for name in scenarios]
        else:
            raise ValueError(
                f"Invalid type for scenarios : {type(scenarios)}. Should be in 'dict', 'list' or 'str'"
            )

        return scenarios_names

    # Méthode auxiliaire de construction du cas de base dans le cadre exonéré
    def init_case_ppv_exoneree(
        self,
        year: int,
        simulation_step_smic: float,
        simulation_max_smic: float,
        ppv: float,
    ) -> None:
        """
        Initializes a case with the PPV exempted.

        Args:
            year (int):
                The year for which the simulation is being performed.
            simulation_step_smic (float):
                The step size for the simulation, as a multiple of the SMIC value.
            simulation_max_smic (float):
                The maximum value for the simulation, as a multiple of the SMIC value.
            ppv (float):
                The amount of the PPV.

        Returns:
            None
        """
        # Initialisation du cas de base
        self.init_base_case(
            year=year,
            simulation_step_smic=simulation_step_smic,
            simulation_max_smic=simulation_max_smic,
        )
        # Modification des paramètres de PPV
        self.base_case["individus"]["individu_1"]["prime_partage_valeur_exoneree"][
            2022
        ] = ppv
        self.base_case["individus"]["individu_1"]["prime_partage_valeur_non_exoneree"][
            2022
        ] = 0

    # Méthode auxiliaire de construction du cas de base dans le cadre réintégré
    # /!\
    # 'axes' = [[
    #         {'count' : simulation_count, 'name' : 'salaire_de_base', 'min' : value_smic, 'max' : simulation_max, 'period' : 2022},
    #         {'count' : simulation_count, 'name' : 'assiette_allegement', 'min' : value_smic+montant_ppv, 'max' : simulation_max+montant_ppv, 'period' : 2022}
    #     ]]
    # N'est peut-être pas équivalent pour l'année 2022 à mettre 'prime_partage_valeur_non_exoneree' = ppv car elle n'est intégrée aux primes non exonérées qu'à compter du 1er juillet
    def init_case_ppv_reintegree(
        self,
        year: int,
        simulation_step_smic: float,
        simulation_max_smic: float,
        ppv: float,
    ) -> None:
        """
        Initializes a case with the PPV reintegrated.

        Args:
            year (int):
                The year for which the simulation is being performed.
            simulation_step_smic (float):
                The step size for the simulation, as a multiple of the SMIC value.
            simulation_max_smic (float):
                The maximum value for the simulation, as a multiple of the SMIC value.
            ppv (float):
                The amount of the PPV.
        """
        # Initialisation du cas de base
        self.init_base_case(
            year=year,
            simulation_step_smic=simulation_step_smic,
            simulation_max_smic=simulation_max_smic,
        )
        # Modification des paramètres de PPV
        self.base_case["individus"]["individu_1"]["prime_partage_valeur_exoneree"][
            2022
        ] = 0
        self.base_case["individus"]["individu_1"]["prime_partage_valeur_non_exoneree"][
            2022
        ] = ppv

    # Fonction auxiliaire de preprocessing du salaire de base
    def _preprocess_salaire_de_base(
        self, data: pd.DataFrame, year: int
    ) -> pd.DataFrame:
        """
        Preprocesses the gross salary.

        Expresses the gross salary as a proportion of the SMIC.

        Args:
            data (pd.DataFrame):
                The input data.
            year (int):
                The year for which the data is being processed.

        Returns:
            (pd.DataFrame): The preprocessed data.
        """
        # Expression du salaire en proportion du SMIC
        data["salaire_de_base_prop_smic"] = data["salaire_de_base"] / self.value_smic(
            year=year
        )

        return data

    # Fonction auxiliaire de preprocessing du salaire super brut
    def _preprocess_salaire_super_brut(
        self,
        data: pd.DataFrame,
        scenarios: Union[str, List[str], dict],
        simulation_case: str,
    ) -> pd.DataFrame:
        """
        Preprocesses the gross salary.

        Calculates the gross salary for each scenario and drops the unnecessary columns.

        Args:
            data (pd.DataFrame):
                The input data.
            scenarios (Union[str, List[str], dict]):
                The scenarios to simulate.
            simulation_case (str):
                The simulation case ('exoneree' or 'reintegree').

        Returns:
            (pd.DataFrame): The preprocessed data.
        """
        # Parcours des scénarios
        for scenario in self.scenarios_names(scenarios=scenarios):
            # Création du salaire super_brut
            data[f"{simulation_case}_salaire_super_brut_{scenario}"] = (
                data[f"{simulation_case}_salaire_super_brut"]
                + data[f"{simulation_case}_allegement_general"]
                - data[f"{simulation_case}_new_allegement_{scenario}"]
            )
            # Suppression du nouvel allègement
            data.drop(
                f"{simulation_case}_new_allegement_{scenario}", axis=1, inplace=True
            )
        # Suppression de l'allègement général
        data.drop(f"{simulation_case}_allegement_general", axis=1, inplace=True)

        return data

    # Fonction auxiliaire de simulation des variables d'intérêt
    def core_simulation(
        self,
        year: int,
        simulation_step_smic: float,
        simulation_max_smic: float,
        simulation_case: str,
        ppv: float,
    ) -> pd.DataFrame:
        """
        Simulates the variables of interest.

        Initializes the simulation case, initializes the tax-benefit system,
        simulates the variables, postprocesses the simulated variables,
        and preprocesses the gross salary.

        Args:
            year (int):
                The year for which the simulation is being performed.
            simulation_step_smic (float):
                The step size for the simulation, as a multiple of the SMIC value.
            simulation_max_smic (float):
                The maximum value for the simulation, as a multiple of the SMIC value.
            simulation_case (str):
                The simulation case ('exoneree' or 'reintegree').
            ppv (float):
                The amount of the PPV.

        Returns:
            (pd.DataFrame): The simulated data.

        Raises:
            ValueError: If the simulation case is not 'exoneree' or 'reintegree'.
        """
        # Initialisation du cas de simulation
        if simulation_case == "exoneree":
            self.init_case_ppv_exoneree(
                year=year,
                simulation_step_smic=simulation_step_smic,
                simulation_max_smic=simulation_max_smic,
                ppv=ppv,
            )
        elif simulation_case == "reintegree":
            self.init_case_ppv_reintegree(
                year=year,
                simulation_step_smic=simulation_step_smic,
                simulation_max_smic=simulation_max_smic,
                ppv=ppv,
            )
        else:
            raise ValueError(
                f"Invalid value for 'simulation_case' : {simulation_case}. Should be in ['exoneree', 'reintegree']"
            )

        # Initialisation du système socio-fiscal
        tax_benefit_system = FranceTaxBenefitSystem()
        # Extraction des variables à simuler
        list_var_simul = params["VARIABLES_PPV"]
        # Simulation
        data_simul = self.base_case_simulation(
            tax_benefit_system=tax_benefit_system,
            year=year,
            list_var_simul=list_var_simul,
        )
        # Post-processing des variables simulées
        # Ajout de la prime partage de la valeur exonérée
        data_simul["prime_partage_valeur_exoneree"] = self.base_case["individus"][
            "individu_1"
        ]["prime_partage_valeur_exoneree"][year]
        # Postprocessing
        data_simul = preprocess_simulated_variables(data=data_simul)

        # Retraitement du salaire de base
        data_simul = self._preprocess_salaire_de_base(data=data_simul, year=year)
        # Logging
        self.logger.info("Successfully preprocessed simulated variables")

        return data_simul[
            [
                "salaire_de_base",
                "salaire_de_base_prop_smic",
                "salaire_super_brut",
                "allegement_general",
                "salaire_net",
            ]
        ].add_prefix(prefix=f"{simulation_case}_")

    # Fonction auxiliaire de simulation théorique des réformes
    def simulate_reform(
        self,
        name: str,
        reform_params: dict,
        year: int,
        simulation_step_smic: float,
        simulation_max_smic: float,
        simulation_case: str,
        ppv: float,
    ) -> pd.DataFrame:
        """
        Simulates a reform.

        Initializes the simulation case, initializes the tax-benefit system,
        applies the reform, simulates the variables, and preprocesses the gross salary.

        Args:
            name (str):
                The name of the reform.
            reform_params (dict):
                The parameters of the reform.
            year (int):
                The year for which the simulation is being performed.
            simulation_step_smic (float):
                The step size for the simulation, as a multiple of the SMIC value.
            simulation_max_smic (float):
                The maximum value for the simulation, as a multiple of the SMIC value.
            simulation_case (str):
                The simulation case ('exoneree' or 'reintegree').
            ppv (float):
                The amount of the PPV.

        Returns:
            (pd.DataFrame): The simulated data.

        Raises:
            ValueError: If the simulation case is not 'exoneree' or 'reintegree'.
        """
        # Initialisation du cas de simulation
        if simulation_case == "exoneree":
            self.init_case_ppv_exoneree(
                year=year,
                simulation_step_smic=simulation_step_smic,
                simulation_max_smic=simulation_max_smic,
                ppv=ppv,
            )
        elif simulation_case == "reintegree":
            self.init_case_ppv_reintegree(
                year=year,
                simulation_step_smic=simulation_step_smic,
                simulation_max_smic=simulation_max_smic,
                ppv=ppv,
            )
        else:
            raise ValueError(
                f"Invalid value for 'simulation_case' : {simulation_case}. Should be in ['exoneree', 'reintegree']"
            )

        # Initialisation des paramètres du système sociofiscal
        tax_benefit_system = FranceTaxBenefitSystem()

        # Application de la réforme
        reformed_tax_benefit_system = create_and_apply_structural_reform_ag(
            tax_benefit_system=tax_benefit_system, dict_params=reform_params
        )

        # Logging
        self.logger.info("Successfully updated the tax-benefit system")

        # Extraction du type de la réforme
        reform_type = reform_params["TYPE"]

        # Itération de la simulation
        data_simul = self.base_case_simulation(
            tax_benefit_system=reformed_tax_benefit_system,
            year=year,
            list_var_simul=["salaire_de_base", f"new_allegement_{reform_type}"],
        )

        # Renomination de la variable simulée pour correspondre au nom du scénario
        data_simul.rename(
            {f"new_allegement_{reform_type}": f"new_allegement_{name.lower()}"},
            axis=1,
            inplace=True,
        )

        # Retraitement du salaire de base
        data_simul = self._preprocess_salaire_de_base(data=data_simul, year=year)

        return data_simul.add_prefix(prefix=f"{simulation_case}_")

    # Fonction auxiliaire de simulation de plusieurs réformes théoriques
    def iterate_reform_simulations(
        self,
        scenarios: Union[str, List[str], dict],
        year: int,
        simulation_step_smic: float,
        simulation_max_smic: float,
        simulation_case: str,
        ppv: float,
    ) -> pd.DataFrame:
        """
        Simulates multiple reforms.

        Iterates over the scenarios and simulates each reform.
        Concatenates the simulated data for all reforms.

        Args:
            scenarios (Union[str, List[str], dict]):
                The scenarios to simulate.
            year (int):
                The year for which the simulation is being performed.
            simulation_step_smic (float):
                The step size for the simulation, as a multiple of the SMIC value.
            simulation_max_smic (float):
                The maximum value for the simulation, as a multiple of the SMIC value.
            simulation_case (str):
                The simulation case ('exoneree' or 'reintegree').
            ppv (float):
                The amount of the PPV.

        Returns:
            (pd.DataFrame): The simulated data for all reforms.
        """
        # Initialisation de la liste résultat
        list_data_simul = []
        # Itération sur les scénarii référencés dans le jeu de données de paramètres
        for i, scenario in tqdm(enumerate(self.scenarios_names(scenarios=scenarios))):
            # Itération des réformes
            data_simul = self.simulate_reform(
                name=scenario,
                reform_params=scenarios[scenario.upper()],
                year=year,
                simulation_step_smic=simulation_step_smic,
                simulation_max_smic=simulation_max_smic,
                simulation_case=simulation_case,
                ppv=ppv,
            )
            # Ajout à la liste résultat
            if i > 0:
                list_data_simul.append(
                    data_simul.drop(
                        [
                            f"{simulation_case}_salaire_de_base",
                            f"{simulation_case}_salaire_de_base_prop_smic",
                        ],
                        axis=1,
                    )
                )
            else:
                list_data_simul.append(data_simul)
        # Concaténation
        data_simul = pd.concat(list_data_simul, axis=1, join="outer")

        return data_simul

    # Fonction auxiliaire de calcul du taux de cotisation implicite
    def calculate_taux_cotisation_implicite(
        self,
        data: pd.DataFrame,
        scenarios: Union[str, List[str], dict],
        simulation_case: str,
    ) -> pd.DataFrame:
        """
        Calculates the implicit tax rate.

        Calculates the implicit tax rate for each scenario and drops the unnecessary columns.

        Args:
            data (pd.DataFrame):
                The input data.
            scenarios (Union[str, List[str], dict]):
                The scenarios to simulate.
            simulation_case (str):
                The simulation case ('exoneree' or 'reintegree').

        Returns:
            (pd.DataFrame): The data with the implicit tax rate calculated.
        """
        # Parcours des scénarios
        for scenario in self.scenarios_names(scenarios=scenarios):
            # Calcul du taux de cotisations implicite
            data[f"{simulation_case}_taux_cotisation_implicite_{scenario}"] = (
                data[f"{simulation_case}_salaire_super_brut_{scenario}"]
                - data[f"{simulation_case}_salaire_net"]
            ) / data[f"{simulation_case}_salaire_super_brut_{scenario}"]
            # Suppression du salaire super_brut
            data.drop(
                f"{simulation_case}_salaire_super_brut_{scenario}", axis=1, inplace=True
            )

        return data

    # Fonction de construction du jeu de données
    def build(
        self,
        scenarios: dict,
        year: int,
        simulation_step_smic: float,
        simulation_max_smic: float,
        simulation_case: str,
        ppv: float,
    ) -> pd.DataFrame:
        """
        Builds the dataset.

        Simulates the core variables and the reform variables, preprocesses the gross salary,
        calculates the implicit tax rate, and returns the dataset.

        Args:
            scenarios (dict):
                The scenarios to simulate.
            year (int):
                The year for which the simulation is being performed.
            simulation_step_smic (float):
                The step size for the simulation, as a multiple of the SMIC value.
            simulation_max_smic (float):
                The maximum value for the simulation, as a multiple of the SMIC value.
            simulation_case (str):
                The simulation case ('exoneree' or 'reintegree').
            ppv (float):
                The amount of the PPV.

        Returns:
            (pd.DataFrame): The dataset.
        """
        # Construction du scénario de base et concaténration des simulations des scénarios de réforme
        data_simul = pd.concat(
            [
                self.core_simulation(
                    year=year,
                    simulation_step_smic=simulation_step_smic,
                    simulation_max_smic=simulation_max_smic,
                    simulation_case=simulation_case,
                    ppv=ppv,
                ),
                self.iterate_reform_simulations(
                    scenarios=scenarios,
                    year=year,
                    simulation_step_smic=simulation_step_smic,
                    simulation_max_smic=simulation_max_smic,
                    simulation_case=simulation_case,
                    ppv=ppv,
                ).drop(
                    [
                        f"{simulation_case}_salaire_de_base",
                        f"{simulation_case}_salaire_de_base_prop_smic",
                    ],
                    axis=1,
                ),
            ],
            axis=1,
            join="outer",
        )
        # Construction des salaires super_bruts associés à chaque scénario
        data_simul = self._preprocess_salaire_super_brut(
            data=data_simul, scenarios=scenarios, simulation_case=simulation_case
        )

        # Calcul du taux de cotisations implicite
        # Pour le cas de base
        data_simul[f"{simulation_case}_taux_cotisation_implicite"] = (
            data_simul[f"{simulation_case}_salaire_super_brut"]
            - data_simul[f"{simulation_case}_salaire_net"]
        ) / data_simul[f"{simulation_case}_salaire_super_brut"]
        # Suppression du salaire super brut
        data_simul.drop(f"{simulation_case}_salaire_super_brut", axis=1, inplace=True)
        # Pour les réformes
        data_simul = self.calculate_taux_cotisation_implicite(
            data=data_simul, scenarios=scenarios, simulation_case=simulation_case
        )
        # Suppression du salaire net
        data_simul.drop(f"{simulation_case}_salaire_net", axis=1, inplace=True)

        return data_simul
