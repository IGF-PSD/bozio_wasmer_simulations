# Importation des modules
# Modules de base
import json
import os
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Modules graphiques
import seaborn as sns
from matplotlib.ticker import PercentFormatter
# Openfisca
# Openfisca-core
from openfisca_core.simulations import SimulationBuilder
from openfisca_core.taxbenefitsystems import TaxBenefitSystem
# Openfisca-france
from openfisca_france import FranceTaxBenefitSystem
from bozio_wasmer_simulations.simulation.empirical.preprocessing import \
    preprocess_simulated_variables
from bozio_wasmer_simulations.simulation.empirical.reform import \
    create_and_apply_structural_reform_ag
# Modules ad hoc
from bozio_wasmer_simulations.utils.utils import _init_logger
from tqdm import tqdm

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

# Importation des paramètres de simulation
with open(
    os.path.join(FILE_PATH.parents[3], "parameters/simulation.json")
) as json_file:
    params = json.load(json_file)


# Classe de simulation théorique
class TheoreticalSimulator:
    """
    A class for performing theoretical simulations.

    This class provides methods for initializing a base case, simulating the base case, and plotting the results.

    Attributes:
        logger (logging.Logger):
            A logger for logging messages.

    """

    # Initialisation
    def __init__(
        self,
        log_filename: Optional[os.PathLike] = os.path.join(
            FILE_PATH.parents[3], "logs/theoretical_simulation.log"
        ),
    ) -> None:
        """
        Constructs all the necessary attributes for the EmpiricalSimulator object.

        Args:
            log_filename (os.PathLike, optional): The path to the log file. Defaults to os.path.join(FILE_PATH.parents[3], 'logs/empirical_simulation.log').

        """
        # Initialisation du logger
        self.logger = _init_logger(filename=log_filename)

    # Fonction auxiliaire de calcul de la valeur du SMIC
    def value_smic(self, year: int) -> float:
        """
        Calculates the value of the SMIC for the given year.

        Args:
            year (int):
                The year for which the SMIC value is calculated.

        Returns:
            (float): The value of the SMIC for the given year.
        """
        # Initialisation du système socio-fiscal contenant les valeurs de SMIC en paramètres
        tax_benefit_system = FranceTaxBenefitSystem()
        value_smic = sum(
            [
                tax_benefit_system.get_parameters_at_instant(
                    instant=f"{year}-{month}"
                ).marche_travail.salaire_minimum.smic.smic_b_mensuel
                for month in [str(m).zfill(2) for m in range(1, 13)]
            ]
        )
        # Logging
        self.logger.info(f"The SMIC value computed for {year} is {value_smic} €")

        return value_smic

    # Initialisation d'un cas de base pour réaliser une simulation
    def init_base_case(
        self, year: int, simulation_step_smic: float, simulation_max_smic: float
    ) -> None:
        """
        Initializes a base case for simulation.

        Args:
            year (int):
                The year for which the simulation is performed.
            simulation_step_smic (float):
                The step size for the simulation, as a multiple of the SMIC value.
            simulation_max_smic (float):
                The maximum value for the simulation, as a multiple of the SMIC value.
        """
        # Initialisation du système socio-fiscal contenant les valeurs de SMIC en paramètres
        tax_benefit_system = FranceTaxBenefitSystem()
        # Extraction de la valeur moyenne de SMIC sur l'année
        value_smic = self.value_smic(year=year)
        # Calcul de la valeur maximale de la simulation et de la valeur du pas
        simulation_max = simulation_max_smic * value_smic
        simulation_step = simulation_step_smic * value_smic
        # Calcul du nombre d'observations dans la simulation entre le min (1 SMIC) et le max avec le pas spécifié
        simulation_count = ceil((simulation_max - value_smic) / simulation_step) + 1
        # Définition des caractéristiques de l'individu
        self.base_case = {
            "individus": {
                "individu_1": {
                    "effectif_entreprise": {year: 200},
                    "depcom_entreprise": {year: "93001"},
                    "contrat_de_travail_debut": {year: "2009-03-16"},
                    "heures_remunerees_volume": {year: 1820},
                    "prime_exceptionnelle_pouvoir_achat": {year: 0},
                    "quotite_de_travail": {year: 12},
                    "prime_partage_valeur_exoneree": {year: 0},
                    "prime_partage_valeur_non_exoneree": {year: 0},
                    "age": {year: 40},
                    "secteur_activite_employeur": {
                        year: "non_agricole"
                    },  # {year : TypesSecteurActivite.non_agricole},
                    "exoneration_cotisations_employeur_tode_eligibilite": {year: False},
                    "choix_exoneration_cotisations_employeur_agricole": {year: False},
                    "travailleur_occasionnel_agricole": {year: False},
                    "zone_restructuration_defense": {year: False},
                    "zone_revitalisation_rurale": {year: False},
                    "categorie_salarie": {
                        year: "prive_non_cadre"
                    },  # {year : TypesCategorieSalarie.prive_non_cadre},
                    "contrat_de_travail": {
                        year: "temps_plein"
                    },  # {year : TypesContratDeTravail.temps_plein},
                    "contrat_de_travail_fin": {year: "2099-12-31"},
                    "contrat_de_travail_type": {
                        year: "cdi"
                    },  # {year : TypesContrat.cdi},
                    "salarie_regime_alsace_moselle": {year: False},
                    #'salaire_de_base'
                    "remuneration_apprenti": {year: 0},
                    "apprentissage_contrat_debut": {year: "1970-01-01"},
                    "apprenti": {year: False},
                    "stage_duree_heures": {year: 0},
                    "stage_gratification": {year: 0},
                    "taux_versement_transport": {year: 0.032},
                    "taux_accident_travail": {year: 0.0212},
                }
            },
            "menages": {
                "menage_1": {
                    "personne_de_reference": ["individu_1"],
                    "depcom": {year: "93001"},
                },
            },
            "familles": {"famille_1": {"parents": ["individu_1"]}},
            "foyers_fiscaux": {"foyer_fiscal_1": {"declarants": ["individu_1"]}},
            "axes": [
                [
                    {
                        "count": simulation_count,
                        "name": "salaire_de_base",
                        "min": value_smic,
                        "max": simulation_max,
                        "period": year,
                    }
                ]
            ],
        }

        # Logging
        self.logger.info("Successfully initialized a test case")

    # Fonction auxilaire d'itération d'une simulation sur un cas
    def base_case_simulation(
        self, tax_benefit_system: TaxBenefitSystem, year: int, list_var_simul: List[str]
    ) -> pd.DataFrame:
        """
        Performs a simulation on the base case.

        Args:
            tax_benefit_system (TaxBenefitSystem):
                The tax-benefit system to use for the simulation.
            year (int):
                The year for which the simulation is performed.
            list_var_simul (List[str]):
                A list of variables to simulate.

        Returns:
            (pd.DataFrame): A dataframe containing the results of the simulation.
        """
        # Initialisation des paramètres de la simulation
        simulation_builder = SimulationBuilder()
        simulation = simulation_builder.build_from_entities(
            tax_benefit_system, self.base_case
        )
        # Initialisation du dictionnaire résultat
        dict_simul = {}
        # Itération sur la liste des variables à simuler
        for variable in list_var_simul:
            dict_simul[variable] = simulation.calculate_add(variable, year)
            # Logging
            self.logger.info(f"Successfully simulated {variable} for period {year}")
        # Conversion en dataFrame
        data_simul = pd.DataFrame(dict_simul)

        return data_simul

    # Fonction auxiliaire de tracé des graphiques
    def plot(
        self,
        data: pd.DataFrame,
        x: str,
        hue: Union[str, List[str]],
        x_label: Optional[Union[str, None]] = None,
        y_label: Optional[Union[str, None]] = None,
        hue_label: Optional[Union[str, None]] = None,
        labels: Optional[Dict[str, str]] = {},
        export_key: Optional[Union[os.PathLike, None]] = None,
        show: Optional[bool] = True,
    ) -> None:
        """
        Plots the results of the simulation.

        Args:
            data (pd.DataFrame):
                The data to plot.
            x (str):
                The variable to use for the x-axis.
            hue (Union[str, List[str]]):
                The variable(s) to use for the hue.
            x_label (Optional[Union[str, None]], optional):
                The label for the x-axis. Defaults to None.
            y_label (Optional[Union[str, None]], optional):
                The label for the y-axis. Defaults to None.
            hue_label (Optional[Union[str, None]], optional):
                The label for the hue. Defaults to None.
            labels (Optional[Dict[str, str]], optional):
                A dictionary of labels to apply to the data. Defaults to {}.
            export_key (Optional[Union[os.PathLike, None]], optional):
                The path to save the plot to. Defaults to None.
            show (Optional[bool], optional):
                Whether to display the plot. Defaults to True.

        Returns:
            None
        """
        # Conversion des arguments en liste
        if isinstance(hue, str):
            hue = [hue]

        # Création des noms à partir des labels
        id_name = x_label if (x_label is not None) else x
        var_name = hue_label if (hue_label is not None) else "Variable"
        value_name = y_label if (y_label is not None) else "Valeur"

        # Réorganisation du jeu de données
        data_graph = pd.melt(
            frame=data,
            id_vars=x,
            value_vars=hue,
            var_name=var_name,
            value_name=value_name,
        ).rename({x: id_name}, axis=1)
        # Application des labels
        data_graph[var_name] = (
            data_graph[var_name].map(labels).fillna(data_graph[var_name])
        )

        # Initialisation de la figure
        fig, ax = plt.subplots()
        # Construction du graphique
        sns.lineplot(data=data_graph, x=id_name, y=value_name, hue=var_name)
        # Formattage de l'axe des ordonnées
        if all(["_prop_" in var_hue for var_hue in hue]):
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        # Exportation
        if export_key is not None:
            plt.savefig(export_key, bbox_inches="tight")

        # Logging
        self.logger.info(f"Successfully build graph")

        if show:
            plt.show()
        else:
            plt.close("all")


class TheoreticalSimulation(TheoreticalSimulator):
    """
    A class for performing theoretical simulations of reforms.

    This class inherits from TheoreticalSimulator and provides methods for simulating variables of interest,
    simulating a reform, and simulating multiple reforms.

    Attributes:
        logger (logging.Logger):
            A logger for logging messages.

    """

    # Initialisation
    def __init__(
        self,
        log_filename: Optional[os.PathLike] = os.path.join(
            FILE_PATH.parents[3], "logs/theoretical_simulation.log"
        ),
    ) -> None:
        """
        Constructs all the necessary attributes for the ReformSimulation object.

        Args:
            log_filename (os.PathLike, optional): The path to the log file. Defaults to os.path.join(FILE_PATH.parents[3], 'logs/theoretical_simulation.log').

        """
        # Initialisation du simulateur
        super().__init__(log_filename=log_filename)

    # Fonction auxiliaire de retraitement de l'assiette d'allègements
    def _preprocess_assiette_allegement(
        self, data: pd.DataFrame, year: int, list_var: List[str]
    ) -> pd.DataFrame:
        """
        Preprocesses the allegement base.

        Expresses all quantities as a proportion of the allegement base.
        Expresses the allegement base as a proportion of the SMIC.

        Args:
            data (pd.DataFrame):
                The input data.
            year (int):
                The year for which the data is being processed.
            list_var (List[str]):
                A list of variables to process.

        Returns:
            (pd.DataFrame): The preprocessed data.
        """
        # Expression de l'ensemble des grandeurs en proportion de l'assiette d'allègements
        list_var_prop = np.setdiff1d(list_var, ["assiette_allegement"]).tolist()
        data[[f"{var}_prop_assiette" for var in list_var_prop]] = data[
            list_var_prop
        ].divide(other=data["assiette_allegement"], axis=0)
        # Expression de l'assiette d'allègements en proportion du SMIC
        data["assiette_allegement_prop_smic"] = data[
            "assiette_allegement"
        ] / self.value_smic(year=year)

        # Logging
        self.logger.info(f"Successfully preprocessed 'assiette_allegement'")

        return data

    # Fonction auxiliaire de simulation des variables d'intérêt
    def core_simulation(
        self, year: int, simulation_step_smic: float, simulation_max_smic: float
    ) -> pd.DataFrame:
        """
        Simulates the variables of interest.

        Initializes the simulation case, initializes the tax-benefit system,
        simulates the variables, postprocesses the simulated variables,
        and preprocesses the allegement base.

        Args:
            year (int):
                The year for which the simulation is being performed.
            simulation_step_smic (float):
                The step size for the simulation, as a multiple of the SMIC value.
            simulation_max_smic (float):
                The maximum value for the simulation, as a multiple of the SMIC value.

        Returns:
            (pd.DataFrame): The simulated data.
        """
        # Initialisation du cas de simulation
        self.init_base_case(
            year=year,
            simulation_step_smic=simulation_step_smic,
            simulation_max_smic=simulation_max_smic,
        )
        # Initialisation du système socio-fiscal
        tax_benefit_system = FranceTaxBenefitSystem()
        # Extraction des variables à simuler
        list_var_simul = params["VARIABLES"]
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
        # Logging
        self.logger.info("Successfully preprocessed simulated variables")
        # Retraitement de l'assiette d'allègements
        data_simul = self._preprocess_assiette_allegement(
            data=data_simul, year=year, list_var=data_simul.columns.tolist()
        )

        return data_simul

    # Fonction auxiliaire de simulation théorique des réformes
    def simulate_reform(
        self,
        name: str,
        reform_params: dict,
        year: int,
        simulation_step_smic: float,
        simulation_max_smic: float,
    ) -> pd.DataFrame:
        """
        Simulates a reform.

        Initializes the simulation case, initializes the tax-benefit system,
        applies the reform, simulates the variables, and preprocesses the allegement base.

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

        Returns:
            (pd.DataFrame): The simulated data.
        """
        # Initialisation du cas de simulation
        if not hasattr(self, "base_case"):
            self.init_base_case(
                year=year,
                simulation_step_smic=simulation_step_smic,
                simulation_max_smic=simulation_max_smic,
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
            list_var_simul=["assiette_allegement", f"new_allegement_{reform_type}"],
        )

        # Renomination de la variable simulée pour correspondre au nom du scénario
        data_simul.rename(
            {f"new_allegement_{reform_type}": f"new_allegement_{name.lower()}"},
            axis=1,
            inplace=True,
        )

        # Retraitement de l'assiette d'allègements
        data_simul = self._preprocess_assiette_allegement(
            data=data_simul,
            year=year,
            list_var=["assiette_allegement", f"new_allegement_{name.lower()}"],
        )

        return data_simul

    # Fonction auxiliaire de simulation de plusieurs réformes théoriques
    def iterate_reform_simulations(
        self,
        scenarios: dict,
        year: int,
        simulation_step_smic: float,
        simulation_max_smic: float,
    ) -> pd.DataFrame:
        """
        Simulates multiple reforms.

        Iterates over the scenarios and simulates each reform.
        Concatenates the simulated data for all reforms.

        Args:
            scenarios (dict):
                The scenarios to simulate.
            year (int):
                The year for which the simulation is being performed.
            simulation_step_smic (float):
                The step size for the simulation, as a multiple of the SMIC value.
            simulation_max_smic (float):
                The maximum value for the simulation, as a multiple of the SMIC value.

        Returns:
            (pd.DataFrame): The simulated data for all reforms.
        """
        # Initialisation de la liste résultat
        list_data_simul = []
        # Itération sur les scénarii référencés dans le jeu de données de paramètres
        for i, scenario in tqdm(enumerate(scenarios.keys())):
            # Itération des réformes
            data_simul = self.simulate_reform(
                name=scenario,
                reform_params=scenarios[scenario],
                year=year,
                simulation_step_smic=simulation_step_smic,
                simulation_max_smic=simulation_max_smic,
            )
            # Ajout à la liste résultat
            if i > 0:
                list_data_simul.append(
                    data_simul.drop(
                        ["assiette_allegement", "assiette_allegement_prop_smic"], axis=1
                    )
                )
            else:
                list_data_simul.append(data_simul)
        # Concaténation
        data_simul = pd.concat(list_data_simul, axis=1, join="outer")

        return data_simul