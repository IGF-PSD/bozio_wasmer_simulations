# Importation des modules
# Modules de base
import json
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from openfisca_core.simulations import SimulationBuilder
# Importation d'Openfisca
# Openfisca-core
from openfisca_core.taxbenefitsystems import TaxBenefitSystem
# Openfisca-france
from openfisca_france import FranceTaxBenefitSystem
from bozio_wasmer_simulations.datasets.loaders import load_dads
from bozio_wasmer_simulations.description.datasets import build_data_evol_ct
# Importation de modules ad hoc
from bozio_wasmer_simulations.simulation.empirical.preprocessing import (
    preprocess_dads_openfisca_ag, preprocess_simulated_variables)
from bozio_wasmer_simulations.simulation.empirical.reform import \
    create_and_apply_structural_reform_ag
from bozio_wasmer_simulations.simulation.empirical.weights import add_weights_eqtp_accos
# Importation de modules ad hoc
from bozio_wasmer_simulations.utils.utils import _init_logger
from tqdm import tqdm

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Importation des paramètres de simulation
with open(
    os.path.join(FILE_PATH.parents[3], "parameters/simulation.json")
) as json_file:
    params = json.load(json_file)


# Création d'un simulateur
class EmpiricalSimulator:
    """
    A class used to simulate empirical data.

    Attributes
    ----------
        logger : logging.Logger
            A logger instance.

    Methods
    -------
        iterate_simulation(data, tax_benefit_system, year, list_var_simul, list_var_exclude, inplace):
            Iterates a simulation.
        simulate_smic_proratise(data, year, list_var_exclude, inplace):
            Simulates the prorated minimum wage.
    """

    # Initialisation
    def __init__(
        self,
        log_filename: Optional[os.PathLike] = os.path.join(
            FILE_PATH.parents[3], "logs/empirical_simulation.log"
        ),
    ) -> None:
        """
        Constructs all the necessary attributes for the EmpiricalSimulator object.

        Args:
        -----
            log_filename (os.PathLike, optional): The path to the log file. Defaults to os.path.join(FILE_PATH.parents[3], 'logs/empirical_simulation.log').

        Returns:
        --------
            None
        """
        # Initialisation du logger
        self.logger = _init_logger(filename=log_filename)

    # Fonction d'itération d'une simulation
    def iterate_simulation(
        self,
        data: pd.DataFrame,
        tax_benefit_system: TaxBenefitSystem,
        year: int,
        list_var_simul: List[str],
        list_var_exclude: Optional[List[str]] = [],
        inplace: Optional[bool] = True,
    ) -> pd.DataFrame:
        """
        Iterates a simulation.

        Args:
        -----
            data (pd.DataFrame): The data to simulate.
            tax_benefit_system (FranceTaxBenefitSystem): The tax benefit system.
            year (int): The year of the simulation.
            list_var_simul (List[str]): The list of variables to simulate.
            list_var_exclude (Optional[List[str]], optional): The list of variables to exclude. Defaults to [].
            inplace (Optional[bool], optional): Whether to perform the simulation in place. Defaults to True.

        Returns:
        --------
            pd.DataFrame: The simulated data.
        """
        # Disjonction de cas suivant la nécessité de réaliser une copie indépendante du jeu de données
        if inplace:
            data_res = data
        else:
            data_res = data.copy()

        # Initialisation des paramètres de la simulation
        simulation = SimulationBuilder().build_default_simulation(
            tax_benefit_system, len(data_res)
        )
        # Ajout de l'ensemble des données
        # /!\ On ajout 'smic_proratisé' aux variables à exclure de l'imputation pour contourner l'écueil de la mauvaise transition entre valeurs mensuelles et annuelles # + ['smic_proratise']
        # Finalement retiré car les rému restent divisées par 12 et ne sont pas intersectées avec la durée du contrat
        # Il s'agit sans doute d'un point à améliorer dans le package
        for caracteristic in np.setdiff1d(data_res.columns, list_var_exclude):
            try:  # if not (caracteristic in ['id', 'siren']) :
                simulation.set_input(
                    caracteristic, year, data_res[caracteristic].to_numpy()
                )
                # logging
                self.logger.info(
                    f"Successfully initialized {caracteristic} in the french tax benefit system"
                )
            except Exception as e:
                # Logging
                self.logger.warning(
                    f"Cannot initialize {caracteristic} in the french tax benefit system : {e}"
                )
                pass
        # Ajout des cotisations et des allègements généraux
        for var_simul in tqdm(list_var_simul):
            data_res[var_simul] = simulation.calculate_add(var_simul, year)
            # Logging
            self.logger.info(f"Successfully simulated {var_simul} for period {year}")

        return data_res

    # Fonction de simulation du SMIC proratisé
    def simulate_smic_proratise(
        self,
        data: pd.DataFrame,
        year: int,
        list_var_exclude: Optional[List[str]] = [],
        inplace: Optional[bool] = True,
    ) -> pd.DataFrame:
        """
        Simulates the prorated minimum wage.

        Args:
        -----
            data (pd.DataFrame): The data to simulate.
            year (int): The year of the simulation.
            list_var_exclude (Optional[List[str]], optional): The list of variables to exclude. Defaults to [].
            inplace (Optional[bool], optional): Whether to perform the simulation in place. Defaults to True.

        Returns:
        --------
            pd.DataFrame: The simulated data.
        """
        # Initialisation des paramètres du système sociofiscal français
        tax_benefit_system = FranceTaxBenefitSystem()

        # Simulation du SMIC proratisé pour l'année des données
        data = self.iterate_simulation(
            data=data,
            tax_benefit_system=tax_benefit_system,
            year=year,
            list_var_simul=["smic_proratise"],
            list_var_exclude=list_var_exclude,
            inplace=inplace,
        )

        return data


# Classe permettant de construire les données nécessaires à la simulation
class CoreSimulation(EmpiricalSimulator):
    """
    A class used to build the core simulation data.

    Methods
    -------
        zonage_zrr():
            Imports the rural revitalization zones (ZRR) zoning.
        zonage_zrd():
            Imports the defense restructuring zones (ZRD) zoning.
        columns_dads(year):
            Builds the columns for the DADS data.
        build_data_dads(year, data):
            Builds the DADS data.
        _build_data_dads_from_dataframe(data, year):
            Builds the DADS data from a DataFrame.
        _init_data_dads(year):
            Imports and preprocesses the DADS data.
        preprocess_dads_simulation(year):
            Preprocesses the DADS data for simulation.
        add_weights(year_data, year_simul):
            Adds weights to the DADS data.
        simulate(year):
            Simulates the data.
        build(year_data, year_simul, data):
            Builds the simulation data.
    """

    # Initialisation
    def __init__(
        self,
        project: str,
        log_filename: Optional[os.PathLike] = os.path.join(
            FILE_PATH.parents[3], "logs/core_simulation.log"
        ),
    ) -> None:
        """
        Constructs all the necessary attributes for the CoreSimulation object.

        Args:
        -----
            log_filename (os.PathLike, optional): The path to the log file. Defaults to os.path.join(FILE_PATH.parents[3], 'logs/core_simulation.log').

        Returns:
        --------
            None
        """
        # Initialisation du projet CASD
        self.project = project
        # Initialisation du logger
        super().__init__(log_filename=log_filename)

    # Fonction auxiliaire d'importation du zonage des Zones de Revitalisation Rurales (ZRR)
    @property
    def zonage_zrr(self) -> List[str]:
        """
        Imports the rural revitalization zones (ZRR) zoning.

        Returns:
        --------
            List[str]: The list of rural revitalization zones.
        """
        # Importation des données
        data_zonage_zrr = pd.read_excel(
            os.path.join(
                FILE_PATH.parents[3], "data/diffusion-zonages-zrr-cog2021.xls"
            ),
            skiprows=5,
            dtype={"CODGEO": str},
        )
        # Sélection des données
        list_zonage_zrr = data_zonage_zrr.loc[
            data_zonage_zrr["ZRR_SIMP"].isin(
                ["C - Classée en ZRR", "P - Commune partiellement classée en ZRR"]
            ),
            "CODGEO",
        ].tolist()

        return list_zonage_zrr

    # Fonction auxiliaire d'importation du zonage des Zones de Restructuration de la Défense (ZRD)
    @property
    def zonage_zrd(self) -> List[str]:
        """
        Imports the defense restructuring zones (ZRD) zoning.

        Returns:
        --------
            List[str]: The list of defense restructuring zones.
        """
        # Importation des données
        list_zonage_zrd = pd.read_excel(
            os.path.join(FILE_PATH.parents[3], "data/diffusion-zonages-zrd-2020.xls"),
            skiprows=5,
            dtype={"CODGEO": str},
        )["CODGEO"].tolist()

        return list_zonage_zrd

    # Fonction auxiliaire de construction des colonnes des DADS
    # @property
    def columns_dads(self, year: int) -> List[str]:
        """
        Builds the columns for the DADS data.

        Args:
        -----
            year (int): The year.

        Returns:
        --------
            List[str]: The list of columns.
        """
        # Liste des variables à conserver lors de l'import
        columns = params["DADS"]["COLONNES"]
        # Ajout des primes de partage de la valeur si on se trouve en 2022
        if year == 2022:
            columns += params["DADS"]["COLONNES_2022"]
        return columns

    # Fonction auxilaire de construction des données DADS
    def build_data_dads(
        self, year: int, data: Optional[Union[pd.DataFrame, None]] = None
    ) -> None:
        """
        Builds the DADS data.

        Args:
        -----
            year (int): The year.
            data (Optional[Union[pd.DataFrame, None]], optional): The data. Defaults to None.

        Returns:
        --------
            None
        """
        if data is not None:
            self._build_data_dads_from_dataframe(data=data, year=year)
        else:
            self.data_dads = self._init_data_dads(year=year)

    # Fonction auxiliaire de construction des données DADS à partir d'un DataFrame
    def _build_data_dads_from_dataframe(self, data: pd.DataFrame, year: int) -> None:
        """
        Builds the DADS data from a DataFrame.

        Args:
        -----
            data (pd.DataFrame): The data.
            year (int): The year.

        Returns:
        --------
            None
        """
        # Vérification que l'ensemble des variables attendues sont dans le jeu de données
        # Variables manquantes
        missing_variables = np.setdiff1d(
            self.columns_dads(year=year), data.columns.tolist()
        ).tolist()
        if missing_variables == []:
            self.data_dads = data
            # Logging
            self.logger.info("Successfully build data_dads")
        else:
            # Logging
            self.logger.error(
                f"Given DataFrame should contain {missing_variables} as columns"
            )
            # Erreur
            raise ValueError(
                f"Given DataFrame should contain {missing_variables} as columns"
            )

    # Fonction auxiliaire d'importation et de retraitement des DADS
    def _init_data_dads(self, year: int) -> pd.DataFrame:
        """
        Imports and preprocesses the DADS data.

        Args:
        -----
            year (int): The year.

        Returns:
        --------
            pd.DataFrame: The preprocessed DADS data.
        """
        # Filtre sur les lignes (sélection des postes principaux de l'année du millésime)
        filter_dads = [("annee", "==", f"{year}"), ("pps", "==", "1")]

        # Chargement des données
        data_dads = load_dads(
            project=self.project, year=year, columns=self.columns_dads(year=year), filters=filter_dads
        )

        # Construction de l'âge
        data_dads["age"] = -(data_dads["annee_naiss"].subtract(other=year))
        # Complétion des Nan
        data_dads["age"] = data_dads["age"].fillna(year - 1970)

        # Restriction sur le champ du secteur privé et aux salariés âgés de 18 à 64 ans, en france métropolitaine
        data_dads = data_dads.loc[
            (~data_dads["domempl_empl"].isin(params["CHAMP"]["DOMEMPL_EXCLUDE"]))
            & (data_dads["age"] >= int(params["CHAMP"]["AGE_MIN"]))
            & (data_dads["age"] <= int(params["CHAMP"]["AGE_MAX"]))
            & (~data_dads["dept"].isin(params["CHAMP"]["DEPT_EXCLUDE"]))
        ]

        # Construction d'un identifiant
        data_dads.reset_index(drop=True, inplace=True)
        data_dads.reset_index(drop=False, names="id", inplace=True)

        # Logging
        self.logger.info("Successfully build data_dads")

        return data_dads

    # Fonction auxiliaire de preprocessing des DADS en vue d'une branchement avec openfisca
    def preprocess_dads_simulation(self, year: int) -> None:
        """
        Preprocesses the DADS data for simulation.

        Args:
        -----
            year (int): The year.

        Returns:
        --------
            None
        """
        # Preprocessing pour les allègements généraux
        self.data_dads = preprocess_dads_openfisca_ag(
            data_dads=self.data_dads,
            year=year,
            list_zonage_zrr=self.zonage_zrr,
            list_zonage_zrd=self.zonage_zrd,
        )

        # Suppression des variables inutiles pour les simulations
        self.data_dads.drop(
            np.setdiff1d(
                self.columns_dads(year=year) + ["pcs_2", "date_fin_contrat"],
                params["PREPROCESSING"]["KEEP"],
            ),
            axis=1,
            inplace=True,
            errors="ignore",
        )

        # Logging
        self.logger.info(
            "Successfully preprocessed data_dads to connect it with openfisca"
        )

    # Fonction auxiliaire d'ajout de poids
    def add_weights(self, year_data: int, year_simul: int) -> None:
        """
        Adds weights to the DADS data.

        Args:
        -----
            year_data (int): The year of the data.
            year_simul (int): The year of the simulation.

        Returns:
        --------
            None
        """
        # Simulation du SMIC proratisé
        # Simulation
        self.data_dads = self.simulate_smic_proratise(
            data=self.data_dads, year=year_data, list_var_exclude=[], inplace=True
        )

        # Si l'année des données ne coincide pas avec l'année des simulations, on met à jour les salaires pour qu'il corresponde au même niveau de SMIC
        if year_data != year_simul:
            # Renomination de la colonne simulée
            self.data_dads.rename(
                {"smic_proratise": f"smic_proratise_{year_data}"}, axis=1, inplace=True
            )
            # Simulation du SMIC proratisé pour l'année de simulation
            self.data_dads = self.simulate_smic_proratise(
                data=self.data_dads, year=year_simul, list_var_exclude=[], inplace=True
            )
            # Correction des salaires
            # Salaire en proportion du SMIC
            self.data_dads["salaire_brut_smic"] = (
                self.data_dads[["salaire_de_base", "remuneration_apprenti"]].sum(axis=1)
                / self.data_dads[f"smic_proratise_{year_data}"]
            )
            # Actualisation des réumnérations
            self.data_dads["salaire_de_base"] = np.where(
                self.data_dads["salaire_de_base"] > 0,
                self.data_dads["salaire_brut_smic"] * self.data_dads["smic_proratise"],
                0,
            )
            self.data_dads["remuneration_apprenti"] = np.where(
                self.data_dads["remuneration_apprenti"] > 0,
                self.data_dads["salaire_brut_smic"] * self.data_dads["smic_proratise"],
                0,
            )
            # Suppression du SMIC proratisé initialement calculé
            self.data_dads.drop(f"smic_proratise_{year_data}", axis=1, inplace=True)
        # Recréation d'un salaire brut
        self.data_dads["brut_s"] = self.data_dads[
            ["salaire_de_base", "remuneration_apprenti"]
        ].sum(axis=1)
        # Ajout des poids
        self.data_dads = add_weights_eqtp_accos(
            data_dads=self.data_dads,
            year=year_simul,
            var_eqtp="eqtp",
            var_sal_brut="brut_s",
            var_smic_proratise="smic_proratise",
        )
        # Suppression de la colonne de salaire brut
        self.data_dads.drop("brut_s", axis=1, inplace=True)

        # Logging
        self.logger.info("Successfully added accoss weights to data_dads")

    # Fonction auxiliaire de simulation
    def simulate(self, year: int) -> None:
        """
        Simulates the data.

        Args:
        -----
            year (int): The year.

        Returns:
        --------
            None
        """
        # Le salaire de base  et smic_proratisé sont des variables mensuelles dans Openfisca et les DADS sont des variables annuelles
        # Les deux variables ayant l'attribut set_input=set_input_divide_by_period mais smic_proratisé est calculé en tenant compte de la durée du contrat
        # Si on simule d'abord un smic proratisé et qu'on en crée une variable annuelle, on divisera par 12 les deux grandeurs, alors qu'il faudrait les intersecter les deux avec la durée du contrat
        # Le rapport smic_proratise/salaire_de_base ou salaire_de_base/smic_proratise reste alors juste.
        # Simulation du SMIC proratisé
        if "smic_proratise" not in self.data_dads.columns:
            self.data_dads = self.simulate_smic_proratise(
                data=self.data_dads, year=year, list_var_exclude=[], inplace=True
            )
        # Liste des variables à simuler
        list_var_simul = np.setdiff1d(params["VARIABLES"], ["smic_proratise"])
        # Initialisation des paramètres du système sociofiscal
        tax_benefit_system = FranceTaxBenefitSystem()
        # Itération de la simulation
        self.data_dads = self.iterate_simulation(
            data=self.data_dads,
            tax_benefit_system=tax_benefit_system,
            year=year,
            list_var_simul=list_var_simul,
            list_var_exclude=[],
            inplace=True,
        )
        # Retraitement des variables simulées
        self.data_dads = preprocess_simulated_variables(data=self.data_dads)
        # Renomination de la quotité de travail pour pallier la mauvaise gestion annuel/mensuel de la variable dans openfisca
        self.data_dads.rename({"eqtp": "quotite_de_travail"}, axis=1, inplace=True)

        # Logging
        self.logger.info(
            f"Successfully simulated {list_var_simul} on data_dads observations"
        )

    # Méthode construisant le jeu de données avec les variables simulées
    def build(
        self,
        year_data: int,
        year_simul: int,
        data: Optional[Union[pd.DataFrame, None]] = None,
    ) -> pd.DataFrame:
        """
        Builds the simulation data.

        Args:
        -----
            year_data (int): The year of the data.
            year_simul (int): The year of the simulation.
            data (Optional[Union[pd.DataFrame, None]], optional): The data. Defaults to None.

        Returns:
        --------
            pd.DataFrame: The simulation data.
        """
        # Chargement du jeu de données
        self.build_data_dads(data=data, year=year_data)
        # Preprocessing
        self.preprocess_dads_simulation(year=year_data)
        # Ajout des poids
        self.add_weights(year_data=year_data, year_simul=year_simul)
        # Simulation des variables
        self.simulate(year=year_simul)

        # Construction de chiffres cadres aggrégés
        aggregated_numbers = (
            self.data_dads[
                [
                    "salaire_de_base",
                    "remuneration_apprenti",
                    "salaire_super_brut",
                    "salaire_super_brut_hors_allegements",
                    "exonerations_et_allegements",
                    "exoneration_cotisations_employeur_apprenti",
                    "exoneration_cotisations_employeur_tode",
                    "exoneration_cotisations_employeur_zrd",
                    "exoneration_cotisations_employeur_zrr",
                    "exoneration_cotisations_employeur_jei",
                    "exoneration_cotisations_employeur_stagiaire",
                    "allegement_general",
                    "allegement_cotisation_maladie",
                    "allegement_cotisation_allocations_familiales",
                    "versement_transport",
                    "prime_partage_valeur_exoneree",
                ]
            ]
            .multiply(other=self.data_dads["weights"], axis=0)
            .sum()
        )
        # Logging
        self.logger.info("Successfully build simulated DataFrame")
        self.logger.info(aggregated_numbers.to_string())

        return self.data_dads


class ReformSimulation(EmpiricalSimulator):
    """
    A class used to build and simulate reform data.

    This class inherits from EmpiricalSimulator and provides methods to build and simulate reform data.
    It includes methods to build simulation data, simulate reforms, and iterate over multiple reform simulations.

    Attributes
    ----------
        log_filename : os.PathLike, optional
            The path to the log file for the simulation. Default is a path in the logs directory.

    Methods
    -------
        build_data_simulation(data, path)
            Builds the simulation data. If data is provided, it uses _build_data_simulation_from_dataframe.
            If not, it reads the data from the provided path.

        _build_data_simulation_from_dataframe(data)
            Builds the simulation data from a DataFrame. Checks if the DataFrame contains all the required variables.

        simulate_reform(name, reform_params, year, taux_bascule_vm)
            Simulates a reform. Applies the reform to the simulation data and calculates new variables.

        iterate_reform_simulations(scenarios, year, taux_bascule_vm)
            Iterates over multiple reform simulations. Simulates each reform in the provided scenarios.

        build(scenarios, year, taux_bascule_vm, data, path)
            Builds the reform simulation data. Calls build_data_simulation and iterate_reform_simulations.
    """

    # Initialisation
    def __init__(
        self,
        log_filename: Optional[os.PathLike] = os.path.join(
            FILE_PATH.parents[3], "logs/reform_simulation.log"
        ),
    ) -> None:
        """
        Constructs all the necessary attributes for the ReformSimulation object.

        Args:
        -----
            log_filename (os.PathLike, optional): The path to the log file. Defaults to os.path.join(FILE_PATH.parents[3], 'logs/reform_simulation.log').

        Returns:
        --------
            None
        """
        # Initialisation du simulateur
        super().__init__(log_filename=log_filename)

    # Fonction auxilaire de construction des données contenant les résultats d'une simulation à partir de laquelle dériver une réforme
    def build_data_simulation(
        self,
        data: Optional[Union[pd.DataFrame, None]] = None,
        path: Optional[Union[os.PathLike, None]] = None,
    ) -> None:
        """
        Builds the simulation data.

        If data is provided, it uses _build_data_simulation_from_dataframe to build the simulation data.
        If not, it reads the data from the provided path.

        Args:
        -----
            data : pd.DataFrame, optional
                The data to use for building the simulation data. If None, the data is read from the path.
            path : os.PathLike, optional
                The path to the data file. Used if data is None.

        Returns:
        --------
            None
        """
        if data is not None:
            self._build_data_simulation_from_dataframe(data=data)
        else:
            self.data_simulation = pd.read_csv(path)

    # Fonction auxiliaire de construction des données de simulation à partir d'un DataFrame
    def _build_data_simulation_from_dataframe(self, data: pd.DataFrame) -> None:
        """
        Builds the simulation data from a DataFrame.

        Checks if the DataFrame contains all the required variables. If it does, it sets the data as the simulation data.
        If it doesn't, it logs an error and raises a ValueError.

        Args:
        -----
        data : pd.DataFrame
            The DataFrame to use for building the simulation data.

        Returns:
        --------
        None
        """
        # Vérification que l'ensemble des variables attendues sont dans le jeu de données
        # Variables manquantes
        missing_variables = np.setdiff1d(
            params["VARIABLES"], data.columns.tolist()
        ).tolist()
        if missing_variables == []:
            self.data_simulation = data
            # Logging
            self.logger.info("Successfully build data_simulation")
        else:
            # Logging
            self.logger.error(
                f"Given DataFrame should contain {missing_variables} as columns"
            )
            # Erreur
            raise ValueError(
                f"Given DataFrame should contain {missing_variables} as columns"
            )

    # Fonction auxiliaire d'itération d'une simulation de réforme
    def simulate_reform(
        self,
        name: str,
        reform_params: dict,
        year: int,
        taux_bascule_vm: Optional[Union[float, None]] = None,
    ) -> None:
        """
        Simulates a reform.

        Applies the reform to the simulation data and calculates new variables. The new variables are added to the simulation data.

        Args
        ----
            name : str
                The name of the reform.
            reform_params : dict
                The parameters of the reform.
            year : int
                The year for the simulation.
            taux_bascule_vm : float, optional
                The rate of the "versement mobilité" (VM) switch. If provided, new variables are calculated for the VM switch.

        Returns
        -------
            None
        """
        # Simulation du SMIC proratisé
        if "smic_proratise" not in self.data_simulation.columns:
            self.data_simulation = self.simulate_smic_proratise(
                data=self.data_simulation, year=year, list_var_exclude=[], inplace=True
            )

        # Initialisation des paramètres du système sociofiscal
        tax_benefit_system = FranceTaxBenefitSystem()

        # Application de la réforme
        reformed_tax_benefit_system = create_and_apply_structural_reform_ag(
            tax_benefit_system=tax_benefit_system, dict_params=reform_params
        )

        # Extraction du type de la réforme
        reform_type = reform_params["TYPE"]
        # Itération de la simulation
        self.data_simulation = self.iterate_simulation(
            data=self.data_simulation,
            tax_benefit_system=reformed_tax_benefit_system,
            year=year,
            list_var_simul=[f"new_allegement_{reform_type}"],
            list_var_exclude=params["REFORM"]["VAR_EXCLUDE"],
            inplace=True,
        )

        # Renomination de la variable simulée pour correspondre au nom du scénario
        self.data_simulation.rename(
            {f"new_allegement_{reform_type}": f"new_allegement_{name.lower()}"},
            axis=1,
            inplace=True,
        )
        # Somme des exonérations et allègements
        self.data_simulation[f"exonerations_et_allegements_{name.lower()}"] = (
            self.data_simulation[
                ["exonerations", f"new_allegement_{name.lower()}"]
            ].sum(axis=1)
        )
        # Calcul du salaire brut avec la réforme
        self.data_simulation[f"salaire_super_brut_{name.lower()}"] = (
            self.data_simulation["salaire_super_brut_hors_allegements"]
            - self.data_simulation[f"exonerations_et_allegements_{name.lower()}"]
            + self.data_simulation["prime_partage_valeur_exoneree"]
        )
        # Construction des variables de variation du coût du travail
        self.data_simulation = build_data_evol_ct(
            data_source=self.data_simulation,
            col_new_ct=f"salaire_super_brut_{name.lower()}",
            col_ct="salaire_super_brut",
            to_concat=True,
        )
        # Logging
        self.logger.info(
            f"Successfully built variables related to labor costs variations inducted by the reform"
        )

        # Construction d'un nouveau coût du travail associé à la bascule d'une fraction du versement transport
        if taux_bascule_vm is not None:
            self.data_simulation[f"salaire_super_brut_vm_{name.lower()}"] = (
                self.data_simulation["salaire_super_brut_hors_allegements"]
                + self.data_simulation["versement_transport"] * (1 - taux_bascule_vm)
                - self.data_simulation[f"exonerations_et_allegements_{name.lower()}"]
                + self.data_simulation["prime_partage_valeur_exoneree"]
            )
            # Construction des variables de variation du coût du travail
            self.data_simulation = build_data_evol_ct(
                data_source=self.data_simulation,
                col_new_ct=f"salaire_super_brut_vm_{name.lower()}",
                col_ct="salaire_super_brut",
                to_concat=True,
            )
            # Logging
            self.logger.info(
                f"Successfully built variables related to labor costs variations inducted by the reform and the vm"
            )

    # Fonction auxiliaire de d'itération de l'ensemble des simulations de réformes
    def iterate_reform_simulations(
        self,
        scenarios: dict,
        year: int,
        taux_bascule_vm: Optional[Union[float, None]] = None,
    ) -> None:
        """
        Iterates over multiple reform simulations.

        Simulates each reform in the provided scenarios. The results are added to the simulation data.

        Args:
        -----
            scenarios : dict
                The scenarios to simulate. Each scenario is a dictionary of reform parameters.
            year : int
                The year for the simulation.
            taux_bascule_vm : float, optional
                The rate of the value-added tax (VM) switch. If provided, new variables are calculated for the VM switch.

        Returns:
        --------
            None
        """
        # Itération sur les scénarii référencés dans le jeu de données de paramètres
        for scenario in tqdm(scenarios.keys()):
            # Itération des réformes
            self.simulate_reform(
                name=scenario,
                reform_params=scenarios[scenario],
                year=year,
                taux_bascule_vm=taux_bascule_vm,
            )

    # Fonction de construction du jeu de données
    def build(
        self,
        scenarios: dict,
        year: int,
        taux_bascule_vm: Optional[Union[float, None]] = None,
        data: Optional[Union[pd.DataFrame, None]] = None,
        path: Optional[Union[os.PathLike, None]] = None,
    ) -> pd.DataFrame:
        """
        Builds the reform simulation data.

        Calls build_data_simulation and iterate_reform_simulations to build the simulation data.

        Args:
        -----
            scenarios : dict
                The scenarios to simulate. Each scenario is a dictionary of reform parameters.
            year : int
                The year for the simulation.
            taux_bascule_vm : float, optional
                The rate of the value-added tax (VM) switch. If provided, new variables are calculated for the VM switch.
            data : pd.DataFrame, optional
                The data to use for building the simulation data. If None, the data is read from the path.
            path : os.PathLike, optional
                The path to the data file. Used if data is None.

        Returns:
        --------
            pd.DataFrame
                The simulation data.
        """
        # Ajout du jeu de données de simulations
        self.build_data_simulation(data=data, path=path)
        # Itération des simulations de réformes
        self.iterate_reform_simulations(
            scenarios=scenarios, year=year, taux_bascule_vm=taux_bascule_vm
        )

        return self.data_simulation
