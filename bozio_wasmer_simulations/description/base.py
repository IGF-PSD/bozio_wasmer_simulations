# Importation des modules
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
# Modules de la toolbox
from igf_toolbox.stats_des.control_secret_stat import \
    SecretStatEstimator
from bozio_wasmer_simulations.datasets.loaders import load_dads, load_fare
from bozio_wasmer_simulations.description.datasets import build_data_evol_emploi
# Modules du package
from bozio_wasmer_simulations.utils.utils import _init_logger
from tqdm import tqdm

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

# Importation des paramètres de description des simulations
with open(
    os.path.join(FILE_PATH.parents[2], "parameters/description.json")
) as json_file:
    description_params = json.load(json_file)

# Importation des paramètres d'élasticités
with open(
    os.path.join(FILE_PATH.parents[2], "parameters/elasticite.json")
) as json_file:
    elasticite_params = json.load(json_file)


# Importation de la classe
class DescriptionBuilder:
    """
    A class used to build a dataset for description and calculate employment effects.


    Attributes:
        logger (logging.Logger):
            A logger instance.
        project (str):
            The name of the CASD project

    """

    # Initialisation
    def __init__(
        self,
        project: str,
        log_filename: Optional[os.PathLike] = os.path.join(
            FILE_PATH.parents[2], "logs/description.log"
        ),
    ) -> None:
        """
        Constructs all the necessary attributes for the DescriptionBuilder object.

        Args:
            project (str): The name of the CASD project
            log_filename (os.PathLike, optional): The path to the log file. Defaults to os.path.join(FILE_PATH.parents[2], 'logs/description.log').

        """
        # Initialisation du projet CASD
        self.project = project
        # Initialisation du logger
        self.logger = _init_logger(filename=log_filename)

    # Fonction auxiliaire d'extraction des noms de scénarios
    def scenarios_names(self, scenarios: Union[str, List[str], dict]) -> List[str]:
        """
        Extracts the names of scenarios.

        Args:
            scenarios (Union[str, List[str], dict]): The scenarios.

        Returns:
            (List[str]): The names of the scenarios.

        Raises:
            ValueError: If the type of scenarios is not 'dict', 'list' or 'str'.
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

    # Construction du jeu de données sur lesquel itérer
    def build_data_simulation(
        self,
        year: int,
        scenarios: Union[str, List[str], dict],
        taux_bascule_vm: Optional[Union[float, None]] = None,
    ) -> None:
        """
        Builds the simulation data.

        Args:
            year (int): The year.
            scenarios (Union[str, List[str], dict]): The scenarios.
            taux_bascule_vm (Optional[Union[float, None]], optional): The rate of switching "versement mobilité" to another tax base. Defaults to None.

        Returns:
            None
        """
        # Disjonction de cas suivant qu'un jeu de données est ou non fourni en argument
        if (
            os.path.exists(
                os.path.join(FILE_PATH.parents[2], f"data/data_simulation_{year}.csv")
            )
            is not None
        ):
            self._build_data_simulation_from_dataframe(year=year, scenarios=scenarios)
        else:
            self._init_data_simulation(
                year=year, scenarios=scenarios, taux_bascule_vm=taux_bascule_vm
            )

    # Fonction auxiliaire de construction des données DADS à partir d'un DataFrame
    def _build_data_simulation_from_dataframe(
        self, year: int, scenarios: Union[str, List[str], dict]
    ) -> None:
        """
        Builds the simulation data from a DataFrame.

        Args:
            year (int): The year.
            scenarios (Union[str, List[str], dict]): The scenarios.

        Returns:
            None
        """
        # Liste des variables de coût du travail
        list_var_ct = np.concatenate(
            [
                [
                    f"diff_ct_{name}",
                    f"evol_ct_{name}",
                    f"evol_ms_{name}",
                    f"salaire_super_brut_{name}",
                ]
                for name in self.scenarios_names(scenarios=scenarios)
            ]
        ).tolist()
        # Importation des données
        self.data_simulation = pd.read_csv(
            os.path.join(FILE_PATH.parents[2], f"data/data_simulation_{year}.csv"),
            usecols=description_params["USECOLS"] + list_var_ct,
        )
        # Logging
        self.logger.info(
            "Successfully initialized data_simulation for DescriptionBuilder"
        )

    # Fonction auxiliaire d'initialisation du jeu de données de simulation en les réalisant intégralement
    def _init_data_simulation(
        self,
        year: int,
        scenarios: dict,
        taux_bascule_vm: Optional[Union[float, None]] = None,
    ) -> None:
        """
        Initializes the simulation data.

        Args:
            year (int): The year.
            scenarios (dict): The scenarios.
            taux_bascule_vm (Optional[Union[float, None]], optional): The rate of switching to VM. Defaults to None.

        Returns:
            None
        """
        # Simulation des cotisations et allègements
        # Initialisation
        core_simulation = CoreSimulation()
        # Itération sur les années de config
        core_simulation.build(year_data=year, year_simul=year)

        # Simulation de réformes
        reform_simulation = ReformSimulation()
        # Itération
        reform_simulation.build(
            scenarios=scenarios,
            taux_bascule_vm=taux_bascule_vm,
            data=core_simulation.self.data_simulation,
        )

        # Liste des variables de coût du travail
        list_var_ct = np.concatenate(
            [
                [
                    f"diff_ct_{name}",
                    f"evol_ct_{name}",
                    f"evol_ms_{name}",
                    f"salaire_super_brut_{name}",
                ]
                for name in self.scenarios_names(scenarios=scenarios)
            ]
        ).tolist()

        # Ajout du jeu de données
        self.data_simulation = reform_simulation.data_simulation[list_var_ct]

        # Logging
        self.logger.info(
            "Successfully initialized data_simulation for DescriptionBuilder"
        )

    # Fonction auxiliaire d'ajout au jeu de données des variables utiles relatives aux variables de coûts du travail
    def add_labor_costs_variables(self, scenarios: Union[str, List[str], dict]) -> None:
        """
        Adds labor costs variables to the simulation data.

        Args:
            scenarios (Union[str, List[str], dict]): The scenarios.

        Returns:
            None
        """
        # Ajout du salaire brut
        self.data_simulation["salaire_brut"] = self.data_simulation[
            ["salaire_de_base", "remuneration_apprenti"]
        ].sum(axis=1)
        # Suppresion des colonnes inutiles
        self.data_simulation.drop(
            ["salaire_de_base", "remuneration_apprenti"], axis=1, inplace=True
        )
        # Ajout du montant de cotisations sociales
        self.data_simulation["cotisations_sociales"] = (
            self.data_simulation["salaire_super_brut"]
            - self.data_simulation["salaire_brut"]
        )
        # Ajout de tranches de salaire brut exprimé en multiples du SMIC
        self.data_simulation["tranche_salaire_brut_smic"] = (
            np.floor(self.data_simulation["salaire_brut_smic"] * 10) / 10
        )

        # Calcul de tranches d'évolution du cout du travail et de la masse salariale
        for name in tqdm(self.scenarios_names(scenarios=scenarios)):
            # Ajout du montant de cotisations sociales
            self.data_simulation[f"cotisations_sociales_{name}"] = (
                self.data_simulation[f"salaire_super_brut_{name}"]
                - self.data_simulation["salaire_brut"]
            )
            # Suppression du salaire super brut
            self.data_simulation.drop(
                f"salaire_super_brut_{name}", axis=1, inplace=True
            )
            # Création de tranches d'évolution de la masse salariale
            self.data_simulation[f"tranche_evol_ct_{name}"] = (
                np.floor(self.data_simulation[f"evol_ct_{name}"] * 100) / 100
            )
            # Création de tranches d'évolution du cout du travail
            self.data_simulation[f"tranche_evol_ms_{name}"] = (
                np.floor(self.data_simulation[f"evol_ms_{name}"] * 100) / 100
            )

    # Fonction auxiliaire de création des effets emplois
    def add_employment_effect(
        self,
        scenarios: Union[str, List[str], dict],
        name_elasticite: str,
        y0_elasticite: Optional[Union[float, None]] = None,
        seuil_pallier_elasticite_smic: Optional[Union[float, None]] = None,
        pallier_elasticite: Optional[Union[float, None]] = None,
        type_elasticite: Optional[Union[str, None]] = None,
    ) -> None:
        """
        Adds employment effect to the simulation data.

        Args:
            scenarios (Union[str, List[str], dict]): The scenarios.
            name_elasticite (str): The name of the elasticity.
            y0_elasticite (Optional[Union[float, None]], optional): The initial value of the elasticity. Defaults to None.
            seuil_pallier_elasticite_smic (Optional[Union[float, None]], optional): The threshold of the elasticity in SMIC. Defaults to None.
            pallier_elasticite (Optional[Union[float, None]], optional): The step of the elasticity. Defaults to None.
            type_elasticite (Optional[Union[str, None]], optional): The type of the elasticity. Defaults to None.

        Returns:
            None
        """
        # Construction des élasticités
        # Si un des éléments n'est pas retouvé, on recherche les paramètres correspondant au nom parmi ceux prédéfinis
        if (
            (y0_elasticite is None)
            | (seuil_pallier_elasticite_smic is None)
            | (pallier_elasticite is None)
            | (type_elasticite is None)
        ):
            dict_elasticite = elasticite_params["ELASTICITES"][name_elasticite]
        # Construction du dictionnaire à partir des paramètres en argument
        else:
            dict_elasticite = {
                "y0_elasticite": y0_elasticite,
                "seuil_pallier_elasticite_smic": seuil_pallier_elasticite_smic,
                "pallier_elasticite": pallier_elasticite,
                "type_elasticite": type_elasticite,
            }
        # Parcours des scénarios
        for name in self.scenarios_names(scenarios=scenarios):
            # Identification de la colonne d'évolution
            col_evol = (
                f"evol_ct_{name}"
                if dict_elasticite["type_elasticite"] == "indiv"
                else f"evol_ms_{name}"
            )
            # Ajout des effets emploi
            self.data_simulation = build_data_evol_emploi(
                data_source=self.data_simulation,
                col_new_ct=f"salaire_super_brut_{name}",
                col_evol=col_evol,
                y0_elasticite=dict_elasticite["y0_elasticite"],
                seuil_pallier_elasticite_smic=dict_elasticite[
                    "seuil_pallier_elasticite_smic"
                ],
                pallier_elasticite=dict_elasticite["pallier_elasticite"],
                name_elasticite=name_elasticite,
                keep_elast=False,
                to_concat=True,
                col_ct="salaire_super_brut",
            )
            # Suppression de la quotité de travail s'il s'agit d'une élasticité "entreprise"
            if dict_elasticite["type_elasticite"] == "firm":
                self.data_simulation.drop(
                    f"quotite_de_travail_{name_elasticite}_{name}", axis=1, inplace=True
                )

    # Fonction auxiliaire d'ajout des effets emploi de l'ensemble des élasticités en paramètres
    def add_employment_effects(
        self,
        scenarios: Union[str, List[str], dict],
        elasticites_names: Optional[Union[List[str], None]] = None,
    ) -> None:
        """
        Adds employment effects to the simulation data.

        Args:
            scenarios (Union[str, List[str], dict]): The scenarios.
            elasticites_names (Optional[Union[List[str], None]], optional): The names of the elasticities. Defaults to None.

        Returns:
            None
        """
        # Initialisation de la slite des élasticités à parcourir
        elasticites_names = (
            elasticite_params["ELASTICITES"].keys()
            if elasticites_names is None
            else elasticites_names
        )
        # Parcours des élasticités en paramètres
        for name_elasticite in tqdm(elasticites_names):
            self.add_employment_effect(
                scenarios=scenarios, name_elasticite=name_elasticite
            )
        # Suppression des variables d'évolution du cout du travail et de la masse salariale après la création des tranches
        self.data_simulation.drop(
            np.concatenate(
                [
                    [f"evol_ct_{name}", f"evol_ms_{name}"]
                    for name in self.scenarios_names(scenarios=scenarios)
                ]
            ).tolist(),
            axis=1,
            inplace=True,
        )

    # Fonction auxiliaire de combinaison des effets emplois individuels et au niveau de l'entreprise
    def combine_firm_indiv_effect(
        self,
        scenarios: dict,
        name_indiv_elasticite: str,
        name_firm_elasticite: str,
        add_weights: bool,
    ) -> None:
        """
        Combines firm and individual employment effects.

        Args:
            scenarios (dict): The scenarios.
            name_indiv_elasticite (str): The name of the individual elasticity.
            name_firm_elasticite (str): The name of the firm elasticity.
            add_weights (bool): Whether to add weights.

        Returns:
            None
        """
        # Parcours des scénarios
        for name in self.scenarios_names(scenarios=scenarios):
            # Ajout de l'effet emploi
            self.data_simulation[
                f"effet_emploi_{name_indiv_elasticite}_{name_firm_elasticite}_{name}"
            ] = self.data_simulation[
                [
                    f"effet_emploi_{name_indiv_elasticite}_{name}",
                    f"effet_emploi_{name_firm_elasticite}_{name}",
                ]
            ].sum(
                axis=1
            )
            # Ajout des poids
            if add_weights:
                # Reconstitution des quotités individus et entreprise
                self.data_simulation[
                    f"quotite_de_travail_{name_indiv_elasticite}_{name_firm_elasticite}_{name}"
                ] = (
                    self.data_simulation["quotite_de_travail"]
                    + self.data_simulation[
                        f"effet_emploi_{name_indiv_elasticite}_{name_firm_elasticite}_{name}"
                    ]
                )
                # Création des poids idoines
                self.data_simulation[
                    f"weights_{name_indiv_elasticite}_{name_firm_elasticite}_{name}"
                ] = np.maximum(
                    self.data_simulation["weights"]
                    * self.data_simulation[
                        f"quotite_de_travail_{name_indiv_elasticite}_{name_firm_elasticite}_{name}"
                    ]
                    / self.data_simulation["quotite_de_travail"],
                    0,
                )
                # Suppression de la quotité de travail
                self.data_simulation.drop(
                    f"quotite_de_travail_{name_indiv_elasticite}_{name_firm_elasticite}_{name}",
                    axis=1,
                    inplace=True,
                )
            # Suppression de la quotité de travail individuelle et de l'effet emploi individuel
            self.data_simulation.drop(
                [
                    f"quotite_de_travail_{name_indiv_elasticite}_{name}",
                    f"effet_emploi_{name_indiv_elasticite}_{name}",
                ],
                axis=1,
                inplace=True,
            )

    # Fonction auxiliaire d'itération des combinaisons d'effets emploi au niveau de l'individu et de l'entreprise
    def combine_firm_indiv_effects(
        self,
        scenarios: dict,
        name_firm_elasticite: str,
        elasticite_names: Optional[Union[List[str], None]] = None,
    ) -> None:
        """
        Combines firm and individual employment effects for all scenarios.

        Args:
            scenarios (dict): The scenarios.
            name_firm_elasticite (str): The name of the firm elasticity.
            elasticite_names (Optional[Union[List[str], None]], optional): The names of the elasticities. Defaults to None.

        Returns:
            None
        """
        # Initialisation de la liste des élasticités à parcourir
        list_elasticite_names = (
            list(elasticite_params["ELASTICITES"].keys())
            if elasticite_names is None
            else elasticite_names
        )
        # Initialisation de la liste des élasticités à combiner
        list_elasticite_combine = []
        # Parcours des élasticités en paramètres afin de déterminer celles au niveau individuel qu'il faut combiner et  s'il faut ou non y ajouter des poids
        for name_elasticite in list_elasticite_names:
            if (
                elasticite_params["ELASTICITES"][name_elasticite]["type_elasticite"]
                == "indiv"
            ) & elasticite_params["ELASTICITES"][name_elasticite]["combine"]:
                list_elasticite_combine.append(
                    {
                        "name_indiv_elasticite": name_elasticite,
                        "add_weights": elasticite_params["ELASTICITES"][
                            name_elasticite
                        ]["add_weights"],
                    }
                )
            elif (
                elasticite_params["ELASTICITES"][name_elasticite]["type_elasticite"]
                == "indiv"
            ):
                # Suppression des variables inutiles
                self.data_simulation.drop(
                    [
                        f"quotite_de_travail_{name_elasticite}_{name}"
                        for name in self.scenarios_names(scenarios=scenarios)
                    ],
                    axis=1,
                    inplace=True,
                )
        # Parcours des élasticités individuelles et ajout des combinaisons
        for elasticite_combine in tqdm(list_elasticite_combine):
            self.combine_firm_indiv_effect(
                scenarios=scenarios,
                name_indiv_elasticite=elasticite_combine["name_indiv_elasticite"],
                name_firm_elasticite=name_firm_elasticite,
                add_weights=elasticite_combine["add_weights"],
            )
        # Suppression des effets emploi au niveau de l'entreprise associés à chaque scénario
        self.data_simulation.drop(
            [
                f"effet_emploi_{name_firm_elasticite}_{name}"
                for name in self.scenarios_names(scenarios=scenarios)
            ],
            axis=1,
            inplace=True,
        )

    # Fonction auxiliaire de chargement et de retraitement des données DADS
    def build_data_dads(self, year: int) -> None:
        """
        Builds the DADS data.

        Args:
            year (int): The year.

        Returns:
            None
        """
        # Filtre sur les lignes (sélection des postes principaux de l'année du millésime)
        filter_dads = [("annee", "==", f"{year}"), ("pps", "==", "1")]

        # Chargement des données
        self.data_dads = load_dads(
            project=self.project,
            year=year,
            columns=description_params["DADS"]["COLONNES"],
            filters=filter_dads,
        )

        # Suppression des colonnes inutiles
        self.data_dads.drop(["annee", "pps"], axis=1, inplace=True)

        # Suppression des duplicats par SIREN
        self.data_dads.drop_duplicates(subset=["siren"], keep="first", inplace=True)
        # Conversion en numériques
        self.data_dads["siren"] = pd.to_numeric(
            self.data_dads["siren"], errors="coerce"
        )
        # Suppression des Nan
        self.data_dads.dropna(subset=["siren"], how="any", inplace=True)
        # Ajout d'une variante excluant l'intérim et le nettoyage
        # Interim
        self.data_dads["a17_interim_nettoyage"] = np.where(
            self.data_dads["apen"].isin(["7820Z"]), "interim", self.data_dads["a17"]
        )
        # Nettoyage
        self.data_dads["a17_interim_nettoyage"] = np.where(
            self.data_dads["apen"].isin(["8121Z", "8122Z", "8129A", "8129B"]),
            "nettoyage",
            self.data_dads["a17_interim_nettoyage"],
        )

    # Fonction auxiliaire de chargement et de retraitement des données FARE
    def build_data_fare(self, year: int) -> None:
        """
        Builds the FARE data.

        Args:
            year (int): The year.

        Returns:
            None
        """
        # Filtre sur les unités légales
        filter_fare = [("diff_ul", "==", "1")]
        # Chargement des données
        self.data_fare = load_fare(
            project=self.project,
            year=year,
            columns=description_params["FARE"]["COLONNES"],
            filters=filter_fare,
        )

        # Suppression de diff_ul
        self.data_fare.drop("diff_ul", axis=1, inplace=True)
        # Suppression des duplicates par SIREN
        self.data_fare.drop_duplicates(subset=["siren"], keep="first", inplace=True)
        # Conversion en types numériques et suppression des Nan
        self.data_fare["siren"] = pd.to_numeric(
            self.data_fare["siren"], errors="coerce"
        )
        self.data_fare.dropna(subset=["siren"], how="any", inplace=True)

    # Fonction auxiliaire d'appariement des données
    def merge_data_simulation_dads_fare(
        self, year: Optional[Union[int, None]] = None
    ) -> None:
        """
        Merges the simulation, DADS, and FARE data.

        Args:
            year (Optional[Union[int, None]], optional): The year. Defaults to None.

        Returns:
            None
        """
        # Vérification que l'ensemble des jeux de données à apparier sont en attribut de la classe
        # data_dads
        if not hasattr(self, "data_dads"):
            self.build_data_dads(year=year)
        # data_fare
        if not hasattr(self, "data_fare"):
            self.build_data_fare(year=year)

        # Appariements
        # Ajout des données DADS
        self.data_simulation = pd.merge(
            left=self.data_simulation,
            right=self.data_dads,
            on="siren",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        # Vérification de la qualité de l'appariement
        _serie_merge_check = self.data_simulation["_merge"].value_counts()
        # Logging
        self.logger.info(
            f"Merge data_simulation <- data_dads : both : {_serie_merge_check.loc['both']}, left_only : {_serie_merge_check.loc['left_only']}, right_only : {_serie_merge_check.loc['right_only']}"
        )
        # Suppression de la colonne d'appariement
        self.data_simulation.drop("_merge", axis=1, inplace=True)

        # Ajout des données FARE
        self.data_simulation = pd.merge(
            left=self.data_simulation,
            right=self.data_fare,
            on="siren",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        # Vérification de la qualité de l'appariement
        _serie_merge_check = self.data_simulation["_merge"].value_counts()
        # Logging
        self.logger.info(
            f"Merge data_simulation <- data_fare : both : {_serie_merge_check.loc['both']}, left_only : {_serie_merge_check.loc['left_only']}, right_only : {_serie_merge_check.loc['right_only']}"
        )
        # Suppression de la colonne d'appariement
        self.data_simulation.drop("_merge", axis=1, inplace=True)

    # Fonction de construction du jeu de données de description
    def build(
        self,
        year: int,
        scenarios: dict,
        name_firm_elasticite: str,
        elasticite_names: Optional[Union[List[str], None]] = None,
    ) -> None:
        """
        Builds the description dataset.

        Args:
            year (int): The year.
            scenarios (dict): The scenarios.
            name_firm_elasticite (str): The name of the firm elasticity.
            elasticite_names (Optional[Union[List[str], None]], optional): The names of the elasticities. Defaults to None.

        Returns:
            None
        """
        # Construction des données de simulation
        self.build_data_simulation(year=year, scenarios=scenarios)
        # Ajout des variables de cout du travail
        self.add_labor_costs_variables(scenarios=scenarios)
        # Ajout des effets emploi
        self.add_employment_effects(
            scenarios=scenarios, elasticites_names=elasticite_names
        )
        # Combinaisons des effets individuels et au niveau de l'entreprise
        self.combine_firm_indiv_effects(
            scenarios=scenarios, name_firm_elasticite=name_firm_elasticite
        )
        # Ajout des variables auxiliaires issues des DADS et de FARE
        self.merge_data_simulation_dads_fare(year=year)

    # Fonction d'itération des statistiques descriptives
    def stat_des(
        self,
        year: int,
        list_var_groupby: List[str],
        list_var_of_interest: List[str],
        output_folder_path: os.PathLike,
        export_filename: str,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], None]:
        """
        Calculates descriptive statistics.

        Args:
            year (int): The year.
            list_var_groupby (List[str]): The variables to group by.
            list_var_of_interest (List[str]): The variables of interest.
            output_folder_path (os.PathLike): The path to the output folder.
            export_filename (str): The name of the export file.

        Returns:
            None
        """
        # Initialisation de l'estimateur de statistiques descriptives
        estimator = SecretStatEstimator(
            data_source=self.data_simulation, 
            list_var_groupby=list_var_groupby, 
            list_var_of_interest=list_var_of_interest, 
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
            include_total=True, 
            drop=True, 
            fill_value=np.nan, 
            nest=False
        )
        
        # Si un chemin et une nom de fichier sont fournis, le jeu de données est exporté
        if (output_folder_path is not None) & (export_filename is not None):
            # Initialisation des chemins d'export des statistiques descriptives et du contrôle du secret statistique
            path_stat_des = os.path.join(
                FILE_PATH.parents[2],
                "outputs",
                f"{datetime.today().strftime('%Y-%m-%d')}/stat_des/{output_folder_path}",
            )
            path_secret_stat = os.path.join(
                FILE_PATH.parents[2],
                "outputs",
                f"{datetime.today().strftime('%Y-%m-%d')}/secret_stat/{output_folder_path}",
            )
            # Création du chemin s'il n'existe pas
            if (not os.path.exists(path_stat_des)) | (
                not os.path.exists(path_secret_stat)
            ):
                os.makedirs(path_stat_des)
                os.makedirs(path_secret_stat)
            # Exportation
            data_stat_des.to_excel(f"{path_stat_des}/stat_des_{export_filename}.xlsx")
            data_secret_stat.to_excel(
                f"{path_secret_stat}/secret_stat_{export_filename}.xlsx"
            )
        # Sinon ils sont retournés
        else:
            return data_stat_des, data_secret_stat

    # Fonction auxiliaire calculant les effets de premiers tours agrégés par scénario
    def first_round_effects(
        self,
        data: pd.DataFrame,
        variable: str,
        name_elasticite: str,
        scenarios: Union[str, List[str], dict],
    ) -> pd.DataFrame:
        """
        Calculates first round effects.

        Args:
            data (pd.DataFrame): The data.
            variable (str): The variable.
            name_elasticite (str): The name of the elasticity.
            scenarios (Union[str, List[str], dict]): The scenarios.

        Returns:
            (pd.DataFrame): The first round effects.
        """
        # Variables manquantes
        missing_variables = np.setdiff1d(
            [variable]
            + [
                f"{variable}_{name}"
                for name in self.scenarios_names(scenarios=scenarios)
            ],
            data.columns.tolist(),
        ).tolist()
        # Si la variable est affectée par les différents scénarios
        if missing_variables == []:
            # Parcours des différentes scénarii
            # Effets statiques
            stat_des_0t = pd.Series(
                data=[(data[variable] * data["weights"]).sum()]
                + [
                    (data[f"{variable}_{name}"] * data["weights"]).sum()
                    for name in self.scenarios_names(scenarios=scenarios)
                ],
                index=["Actuel"] + self.scenarios_names(scenarios=scenarios),
                name=f"{variable} (statiques)",
            )
            # Effets de premier tour
            stat_des_1t = pd.Series(
                data=[(data[variable] * data[f"weights"]).sum()]
                + [
                    (
                        data[f"{variable}_{name}"]
                        * data[f"weights_{name_elasticite}_{name}"]
                    ).sum()
                    for name in self.scenarios_names(scenarios=scenarios)
                ],
                index=["Actuel"] + self.scenarios_names(scenarios=scenarios),
                name=f"{variable} (1er tour)",
            )
            stat_des = pd.concat(
                [stat_des_0t.to_frame(), stat_des_1t.to_frame()], axis=1, join="outer"
            )
            # Différence par rapport à la situation actuelle
            stat_des = pd.concat(
                [
                    stat_des,
                    stat_des.subtract(other=stat_des.loc["Actuel"]).add_prefix(
                        "variation_"
                    ),
                ],
                axis=1,
            )

        else:
            # Parcours des différentes scénarii
            # Effets statiques
            stat_des_0t = pd.Series(
                data=[(data[variable] * data["weights"]).sum()]
                * (len(self.scenarios_names(scenarios=scenarios)) + 1),
                index=["Actuel"] + self.scenarios_names(scenarios=scenarios),
                name=f"{variable} (statiques)",
            )
            # Effets de premier tour
            stat_des_1t = pd.Series(
                data=[(data[variable] * data[f"weights"]).sum()]
                + [
                    (data[variable] * data[f"weights_{name_elasticite}_{name}"]).sum()
                    for name in self.scenarios_names(scenarios=scenarios)
                ],
                index=["Actuel"] + self.scenarios_names(scenarios=scenarios),
                name=f"{variable} (1er tour)",
            )
            stat_des = pd.concat(
                [stat_des_0t.to_frame(), stat_des_1t.to_frame()], axis=1, join="outer"
            )
            # Différence par rapport à la situation actuelle
            stat_des = pd.concat(
                [
                    stat_des,
                    stat_des.subtract(other=stat_des.loc["Actuel"]).add_prefix(
                        "variation_"
                    ),
                ],
                axis=1,
            )

        return stat_des