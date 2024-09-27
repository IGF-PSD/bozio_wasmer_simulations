# Importation des modules
# Module de base
import json
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
# Modules de la toolbox
from igf_toolbox.stats_des.control_secret_stat import \
    SecretStatEstimator
# Openfisca
# Openfisca-france
from openfisca_france import FranceTaxBenefitSystem
from bozio_wasmer_simulations.simulation.empirical.base import CoreSimulation
from bozio_wasmer_simulations.simulation.empirical.preprocessing import \
    preprocess_simulated_variables
from bozio_wasmer_simulations.simulation.empirical.reform import \
    create_and_apply_structural_reform_ag
# Modules du package
from bozio_wasmer_simulations.simulation.theoretical.base import TheoreticalSimulator
from tqdm import tqdm

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

# Importation des paramètres de simulation
with open(
    os.path.join(FILE_PATH.parents[3], "parameters/simulation.json")
) as json_file:
    params = json.load(json_file)


# Classe permettant la réalisation des simulation de taux marginaux
class CaptationMarginaleSimulator(TheoreticalSimulator, CoreSimulation):
    """
    A class for simulating marginal capture rates.

    This class inherits from TheoreticalSimulator and CoreSimulation and provides methods for calculating the marginal capture rate,
    preprocessing the gross salary, building the columns for the DADS data, preprocessing the DADS data for simulation,
    simulating the marginal capture rate, simulating a reform, simulating multiple reforms, building the weights for the simulation,
    building the dataset, and calculating a synthetic rate.

    Attributes:
    -----------
        logger : logging.Logger
            A logger for logging messages.

    Methods:
    --------
        _calculate_taux_captation_marginal(data, name)
            Calculates the marginal capture rate.

        _preprocess_salaire_de_base(data, year, name)
            Preprocesses the gross salary.

        columns_dads(year)
            Returns the columns to keep from the DADS data.

        preprocess_dads_simulation(year)
            Preprocesses the DADS data for simulation.

        simulate(year, simulation_step_smic, simulation_max_smic)
            Simulates the marginal capture rate.

        simulate_reform(name, reform_params, year, simulation_step_smic, simulation_max_smic)
            Simulates a reform.

        iterate_reform_simulations(scenarios, year, simulation_step_smic, simulation_max_smic)
            Simulates multiple reforms.

        build_weights_simulation(data_simul, year, simulation_max_smic, list_var_groupby)
            Builds the weights for the simulation.

        build(year_data, year_simul, simulation_step_smic, simulation_max_smic, scenarios, data)
            Builds the dataset.

        build_taux_synthetique(data, elasticite, names, weights)
            Calculates a synthetic rate.
    """

    # Initialisation
    def __init__(
        self,
        project: str,
        log_filename: Optional[os.PathLike] = os.path.join(
            FILE_PATH.parents[3], "logs/captation_marginale_simulation.log"
        ),
    ) -> None:
        """
        Constructs all the necessary attributes for the ReformSimulation object.

        Args:
        -----
            project (str): The name of the CASD project
            log_filename (os.PathLike, optional): The path to the log file. Defaults to os.path.join(FILE_PATH.parents[3], 'logs/captation_marginale_simulation.log').

        Returns:
        --------
            None
        """
        # Initialisation du simulateur
        TheoreticalSimulator.__init__(log_filename=log_filename)
        CoreSimulation.__init__(project=project, log_filename=log_filename)


    # Fonction auxiliaire de calcul du taux de captation marginal
    def _calculate_taux_captation_marginal(
        self, data: pd.DataFrame, name: Union[str, None]
    ) -> pd.DataFrame:
        """
        Calculates the marginal capture rate.

        Args:
        -----
            data : pd.DataFrame
                The input data.
            name : Union[str, None]
                The name of the scenario.

        Returns:
        --------
            pd.DataFrame
                The input data with the marginal capture rate calculated.
        """
        # Calcul des \delta S_NET / \delta Coût du travail
        if (name is None) | (name == ""):
            data["taux_captation_marginal"] = (
                data["salaire_net"].shift(-1) - data["salaire_net"]
            ) / (data["salaire_super_brut"].shift(-1) - data["salaire_super_brut"])
            data["taux_captation_marginal"] = data["taux_captation_marginal"].fillna(
                method="ffill"
            )
        else:
            data[f"taux_captation_marginal_{name}"] = (
                data["salaire_net"].shift(-1) - data["salaire_net"]
            ) / (
                data[f"salaire_super_brut_{name}"].shift(-1)
                - data[f"salaire_super_brut_{name}"]
            )
            data[f"taux_captation_marginal_{name}"] = data[
                f"taux_captation_marginal_{name}"
            ].fillna(method="ffill")

        return data

    # Fonction auxiliaire de preprocessing du salaire de base
    def _preprocess_salaire_de_base(
        self, data: pd.DataFrame, year: int, name: str
    ) -> pd.DataFrame:
        """
        Preprocesses the gross salary.

        Expresses the gross salary as a proportion of the SMIC and drops the unnecessary columns.

        Args:
        -----
            data : pd.DataFrame
                The input data.
            year : int
                The year for which the data is being processed.
            name : str
                The name of the scenario.

        Returns:
        --------
            pd.DataFrame
                The preprocessed data.
        """
        # Expression du salaire en proportion du SMIC
        data["salaire_de_base_prop_smic"] = data["salaire_de_base"] / self.value_smic(
            year=year
        )
        # Liste des variables à conserver
        list_var_keep = (
            ["salaire_de_base", "salaire_de_base_prop_smic", "taux_captation_marginal"]
            if ((name is None) | (name == ""))
            else [
                "salaire_de_base",
                "salaire_de_base_prop_smic",
                f"taux_captation_marginal_{name}",
            ]
        )
        # Suppression des variables inutiles
        data.drop(
            np.setdiff1d(data.columns.tolist(), list_var_keep), axis=1, inplace=True
        )

        return data

    # Fonction auxiliaire de construction des colonnes des DADS
    def columns_dads(self, year: int) -> List[str]:
        """
        Returns the columns to keep from the DADS data.

        Args:
        -----
            year : int
                The year for which the data is being processed.

        Returns:
        --------
            List[str]
                The columns to keep from the DADS data.
        """
        # Liste des variables à conserver lors de l'import
        return params["DADS"]["COLONNES_CAPTATION_MARGINALE"]

    # Fonction auxiliaire de preprocessing des DADS en vue d'une branchement avec openfisca
    def preprocess_dads_simulation(self, year: int) -> None:
        """
        Preprocesses the DADS data for simulation.

        Args:
        -----
            year : int
                The year for which the data is being processed.
        """
        # Simulation du SMIC proratisé
        # Construction des variables d'intérêt
        # Conversion de la de la date de début de contrat de travail en datetime
        self.data_dads["date_fin_contrat"] = (
            pd.to_datetime(f"{year}-01-01", format="%Y-%m-%d")
            + pd.to_timedelta(arg=self.data_dads["datfin"], unit="D")
        ).dt.strftime("%Y-%m-%d")
        self.data_dads["contrat_de_travail_fin"] = np.where(
            self.data_dads["datfin"] < 360,
            self.data_dads["date_fin_contrat"],
            "2099-12-31",
        )
        # Conersion en string
        self.data_dads["date_debut_contrat"] = (
            pd.to_datetime(self.data_dads["date_debut_contrat"], format="%Y-%m-%d")
            .dt.strftime("%Y-%m-%d")
            .fillna("1970-01-01")
        )
        # Renomination de certaines variables
        self.data_dads.rename(
            {
                "date_debut_contrat": "contrat_de_travail_debut",
                "nbheur": "heures_remunerees_volume",
                "brut_s": "salaire_de_base",
            },
            axis=1,
            inplace=True,
        )
        # Ajout de la rémunération de l'apprenti
        self.data_dads["remuneration_apprenti"] = 0
        # Expression du salaire de base en fonction du SMIC
        self.data_dads["salaire_de_base_prop_smic"] = self.data_dads[
            "salaire_de_base"
        ] / self.value_smic(year=year)

        # Suppression des colonnes inutiles
        self.data_dads.drop(
            np.setdiff1d(
                self.columns_dads(year=year) + ["date_fin_contrat"],
                ["ident_s", "siren", "eqtp", "brut_s", "date_debut_contrat", "nbheur"],
            ).tolist(),
            axis=1,
            inplace=True,
        )

        # Logging
        self.logger.info(
            "Successfully preprocessed data_dads to connect it with openfisca"
        )

    # Fonction auxiliaire de simulation du taux marginal
    def simulate(
        self, year: int, simulation_step_smic: float, simulation_max_smic: float
    ) -> pd.DataFrame:
        """
        Simulates the marginal capture rate.

        Initializes the simulation case, initializes the tax-benefit system,
        simulates the variables, postprocesses the simulated variables,
        calculates the marginal capture rate, and preprocesses the gross salary.

        Args:
        -----
            year : int
                The year for which the simulation is being performed.
            simulation_step_smic : float
                The step size for the simulation, as a multiple of the SMIC value.
            simulation_max_smic : float
                The maximum value for the simulation, as a multiple of the SMIC value.

        Returns:
        --------
            pd.DataFrame
                The simulated data.
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
        list_var_simul = params["VARIABLES_CAPTATION_MARGINALE"]
        # Simulation
        data_simul = self.base_case_simulation(
            tax_benefit_system=tax_benefit_system,
            year=year,
            list_var_simul=list_var_simul,
        )
        # Retraitement des variables simulées
        data_simul = preprocess_simulated_variables(data=data_simul)
        # Calcul du taux marginal
        data_simul = self._calculate_taux_captation_marginal(data=data_simul, name=None)
        # Retraitement du salaire de base
        data_simul = self._preprocess_salaire_de_base(
            data=data_simul, year=year, name=None
        )

        return data_simul

    # Fonction auxiliaire de simulation d'une réforme
    def simulate_reform(
        self,
        name: str,
        reform_params: dict,
        year: int,
        simulation_step_smic: float,
        simulation_max_smic: float,
    ):
        """
        Simulates a reform.

        Initializes the simulation case, initializes the tax-benefit system,
        applies the reform, simulates the variables, postprocesses the simulated variables,
        calculates the marginal capture rate, and preprocesses the gross salary.

        Args:
        -----
            name : str
                The name of the reform.
            reform_params : dict
                The parameters of the reform.
            year : int
                The year for which the simulation is being performed.
            simulation_step_smic : float
                The step size for the simulation, as a multiple of the SMIC value.
            simulation_max_smic : float
                The maximum value for the simulation, as a multiple of the SMIC value.

        Returns:
        --------
            pd.DataFrame
                The simulated data.
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
            list_var_simul=params["VARIABLES_CAPTATION_MARGINALE"]
            + [f"new_allegement_{reform_type}"],
        )

        # Retraitement des variables simulées
        data_simul = preprocess_simulated_variables(data=data_simul)

        # Construction du nouveau salaire super brut
        data_simul[f"salaire_super_brut_{name}"] = (
            data_simul[
                [
                    "salaire_super_brut",
                    "allegement_general",
                    "allegement_cotisation_maladie",
                    "allegement_cotisation_allocations_familiales",
                ]
            ].sum(axis=1)
            - data_simul[f"new_allegement_{reform_type}"]
        )

        # Calcul du taux marginal
        data_simul = self._calculate_taux_captation_marginal(data=data_simul, name=name)

        # Retraitement du salaire de base
        data_simul = self._preprocess_salaire_de_base(
            data=data_simul, year=year, name=name
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
        -----
            scenarios : dict
                The scenarios to simulate.
            year : int
                The year for which the simulation is being performed.
            simulation_step_smic : float
                The step size for the simulation, as a multiple of the SMIC value.
            simulation_max_smic : float
                The maximum value for the simulation, as a multiple of the SMIC value.

        Returns:
        --------
            pd.DataFrame
                The simulated data for all reforms.
        """
        # Initialisation de la liste résultat
        list_data_simul = []
        # Itération sur les scénarii référencés dans le jeu de données de paramètres
        for i, scenario in tqdm(enumerate(scenarios.keys())):
            # Itération des réformes
            data_simul = self.simulate_reform(
                name=scenario.lower(),
                reform_params=scenarios[scenario],
                year=year,
                simulation_step_smic=simulation_step_smic,
                simulation_max_smic=simulation_max_smic,
            )
            # Ajout à la liste résultat
            if i > 0:
                list_data_simul.append(
                    data_simul.drop(
                        ["salaire_de_base", "salaire_de_base_prop_smic"], axis=1
                    )
                )
            else:
                list_data_simul.append(data_simul)
        # Concaténation
        data_simul = pd.concat(list_data_simul, axis=1, join="outer")

        return data_simul

    # Fonction auxiliaire de construction des poids
    def build_weights_simulation(
        self,
        data_simul: pd.DataFrame,
        year: int,
        simulation_max_smic: float,
        list_var_groupby: Optional[List[str]] = ["salaire_de_base_prop_smic"],
    ):
        """
        Builds the weights for the simulation.

        Args:
        -----
            data_simul : pd.DataFrame
                The simulated data.
            year : int
                The year for which the data is being processed.
            simulation_max_smic : float
                The maximum value for the simulation, as a multiple of the SMIC value.
            list_var_groupby : Optional[List[str]], optional
                The variables to group by, by default ['salaire_de_base_prop_smic']

        Returns:
        --------
            Tuple[pd.DataFrame, pd.DataFrame]
                The descriptive statistics and the secret statistics.
        """
        # Construction du jeu de données data_dads s'il n'est pas déjà en argument
        if not hasattr(self, "data_dads"):
            self.build_data_dads(year=year)

        # Création de tranches de salaires
        self.data_dads["salaire_de_base_prop_smic"] = pd.to_numeric(
            pd.cut(
                x=self.data_dads["salaire_de_base_prop_smic"],
                bins=data_simul["salaire_de_base_prop_smic"].tolist(),
                labels=data_simul["salaire_de_base_prop_smic"].tolist()[:-1],
                include_lowest=True,
            )
        )
        # Restriction aux salaires inférieurs à 4 SMIC
        self.data_dads = self.data_dads.loc[
            self.data_dads["salaire_de_base_prop_smic"] <= simulation_max_smic
        ]

        # Initialisation de l'estimateur de statistiques descriptives
        estimator = SecretStatEstimator(
            data_source=self.data_dads, 
            list_var_groupby=list_var_groupby, 
            list_var_of_interest=["eqtp", "salaire_de_base"], 
            var_individu="ident_s", 
            var_entreprise="siren", 
            var_weights="weights", 
            threshold_secret_stat_effectif_individu=5, 
            threshold_secret_stat_effectif_entreprise=3, 
            threshold_secret_stat_contrib_individu=0.8, 
            threshold_secret_stat_contrib_entreprise=0.85, 
            strategy='total'
        )
        # Construction de la statistique descriptive et du contrôle du secret statistique
        data_stat_des, data_secret_stat = estimator.estimate_secret_stat(
            iterable_operations=["sum"], 
            include_total=False, 
            drop=False, 
            fill_value=np.nan, 
            nest=False
        )

        return data_stat_des, data_secret_stat

    # Méthode construisant le jeu de données avec les variables simulées
    def build(
        self,
        year_data: int,
        year_simul: int,
        simulation_step_smic: float,
        simulation_max_smic: float,
        scenarios: Optional[Union[dict, None]] = None,
        data: Optional[Union[pd.DataFrame, None]] = None,
    ) -> pd.DataFrame:
        """
        Builds the dataset.

        Loads the data, preprocesses it, adds weights, simulates the variables,
        simulates the reforms, builds the weights, and returns the dataset.

        Args:
        -----
            year_data : int
                The year of the data.
            year_simul : int
                The year for which the simulation is being performed.
            simulation_step_smic : float
                The step size for the simulation, as a multiple of the SMIC value.
            simulation_max_smic : float
                The maximum value for the simulation, as a multiple of the SMIC value.
            scenarios : Optional[Union[dict, None]], optional
                The scenarios to simulate, by default None
            data : Optional[Union[pd.DataFrame, None]], optional
                The data, by default None

        Returns
        -------
            pd.DataFrame
                The dataset.
        """
        # Chargement du jeu de données
        self.build_data_dads(data=data, year=year_data)
        # Preprocessing
        self.preprocess_dads_simulation(year=year_data)
        # Ajout des poids
        self.add_weights(year_data=year_data, year_simul=year_simul)
        # Simulation des variables
        data_simul = self.simulate(
            year=year_simul,
            simulation_step_smic=simulation_step_smic,
            simulation_max_smic=simulation_max_smic,
        )
        # Itération des réformes
        if scenarios is not None:
            data_simul = pd.concat(
                [
                    data_simul,
                    self.iterate_reform_simulations(
                        scenarios=scenarios,
                        year=year_simul,
                        simulation_step_smic=simulation_step_smic,
                        simulation_max_smic=simulation_max_smic,
                    ).drop(["salaire_de_base", "salaire_de_base_prop_smic"], axis=1),
                ],
                axis=1,
                join="outer",
            )
        # Construction des poids
        data_stat_des, data_secret_stat = self.build_weights_simulation(
            data_simul=data_simul,
            year=year_simul,
            simulation_max_smic=simulation_max_smic,
        )
        # Concaténation avec les EQTP et la masse salariale par tranche
        data_simul = pd.concat(
            [
                data_simul.set_index("salaire_de_base_prop_smic"),
                data_stat_des.drop(
                    ["secret_stat_primary", "secret_stat_secondary"], axis=1
                ),
            ],
            axis=1,
            join="inner",
        ).reset_index()

        # Logging
        self.logger.info("Successfully build simulated DataFrame")

        return data_simul

    # Fonction auxiliaire calculant un taux syntéhtique
    def build_taux_synthetique(
        self,
        data: pd.DataFrame,
        elasticite: int,
        names: List[str],
        weights: Optional[List[str]] = ["eqtp_sum", "salaire_de_base_sum"],
    ) -> pd.DataFrame:
        """
        Calculates a synthetic rate.

        Args:
        -----
            data : pd.DataFrame
                The input data.
            elasticite : int
                The elasticity.
            names : List[str]
                The names of the scenarios.
            weights : Optional[List[str]], optional
                The weights, by default ['eqtp_sum', 'salaire_de_base_sum']

        Returns:
        --------
            pd.DataFrame
                The synthetic rate.

        Raises:
        -------
            ValueError
                If the input data does not contain the necessary columns.
        """
        # Vérification que les les colonnes nécessaires sont bien présentes dans le jeu de données
        missing_columns = np.setdiff1d(
            weights
            + ["taux_captation_marginal"]
            + [f"taux_captation_marginal_{name.lower()}" for name in names],
            data.columns.tolist(),
        ).tolist()
        if missing_columns != []:
            # Logging
            self.logger.error(
                f"Given DataFrame should contain {missing_columns} as columns"
            )
            # Erreur
            raise ValueError(
                f"Given DataFrame should contain {missing_columns} as columns"
            )

        # Initialisation du jeu de données résultat
        data_res = pd.DataFrame(data=0, index=weights, columns=names)
        # Complétion du jeu de données
        for weight in weights:
            for name in names:
                data_res.loc[weight, name] = (
                    elasticite
                    * (
                        data[f"taux_captation_marginal_{name.lower()}"]
                        / data["taux_captation_marginal"]
                        - 1
                    )
                    .multiply(other=data[weight])
                    .sum()
                    / data[weight].sum()
                )

        return data_res