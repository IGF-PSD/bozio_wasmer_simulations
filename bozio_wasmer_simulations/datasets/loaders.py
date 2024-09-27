# Importaion des modules
# Modules de base
import os
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Importation du loader
from bozio_wasmer_simulations.datasets.base import Loader


# Fonction de construction de la base DADS pour une année
def load_dads(
    project: str, year: int, columns: List[str], filters: Optional[List[Tuple[str, str, str]]] = None
):
    """
    Loads the DADS data for a given year.

    Args:
    ----------
        project (str) : The name of the CASD project
        year (int): The year of the data to load.
        columns (List[str]): The columns to load.
        filters (Optional[List[Tuple[str, str, str]]], optional): The filters to apply. Each filter is a tuple of (column, operator, value). Defaults to None.

    Returns:
    ----------
        pd.DataFrame: The loaded data.

    Raises:
    ----------
        ValueError: If the data is not available for the given year.
    """
    # Initialisation du loader
    loader = Loader()
    # Distinction selon l'année
    if year < 2018:
        # Variables à conserver lors de l'import
        columns = [e.upper() for e in columns]
        # Initialisation de la liste résultat
        list_data_dads = []
        # Chemin d'accès aux données
        table_path = f"\\\casd.fr\\casdfs\\Projets\\{project}\\Data\\DADS_DADS Postes_{year}\\Régions"
        # Importation des différentes tables
        for i in tqdm([24, 27, 28, 32, 44, 52, 53, 75, 76, 84, 93, 94, 97, 99]):
            # Nom du jeu de données
            table_name = f"post{i}.sas7bdat"
            # Importation des données
            list_data_dads.append(
                loader.load(
                    path=os.path.join(table_name, table_path),
                    columns=columns,
                    filters=filters,
                )
            )
        # Concaténation des données
        data_dads = pd.concat(list_data_dads, axis=0, ignore_index=True)

    elif (year >= 2018) & (year < 2020):
        # Variables à conserver lors de l'import
        columns = [e.upper() for e in columns]
        # Initialisation de la liste résultat
        list_data_dads = []
        # Chemin d'accès aux données
        table_path = f"\\\casd.fr\\casdfs\\Projets\\{project}\\Data\\DADS_DADS Postes_{year}"
        # Importation des différentes tables
        for i in tqdm(range(1, 5)):
            # Nom du jeu de données
            table_name = f"post_{i}.sas7bdat"
            # Importation des données
            list_data_dads.append(
                loader.load(
                    path=os.path.join(table_name, table_path),
                    columns=columns,
                    filters=filters,
                )
            )
        # Concaténation des données
        data_dads = pd.concat(list_data_dads, axis=0, ignore_index=True)
    # Chemin d'accès aux données
    elif (year >= 2020) & (year < 2022):
        # Variables à conserver lors de l'import
        columns = [e.lower() for e in columns]
        # Chemin
        table_path = f"\\\casd.fr\\casdfs\\Projets\\{project}\\Data\\DADS_DADS Postes_{year}\\Format parquet"
        # Chargement des données
        data_dads = loader.load(path=table_path, columns=columns, filters=filters)
    elif year == 2022:
        # Variables à conserver lors de l'import
        columns = [e.lower() for e in columns]
        # Chemin
        table_path = f"\\\casd.fr\\casdfs\\Projets\\{project}\\Data\\DADS_DADS Postes_{year}"
        # Enumération des fichiers
        list_files = os.listdir(table_path)

        # Restriction aux fichiers parquet relatifs à l'année 2022
        data_dads = pd.concat(
            (
                loader.load(
                    path=f"{table_path}\\{file}", columns=columns, filters=filters
                )
                for file in list_files
                if ((file.endswith(".parquet")) & (str(year) in file))
            ),
            axis=0,
            join="outer",
            ignore_index=True,
        )
    else:
        raise ValueError(f"Data not available for year : {year}")
        # data_dads = pd.read_parquet(table_path, columns=columns, filters=filters)
    return data_dads


# Fonction de construction de la base FARE pour une année
def load_fare(
    project : str, year: int, columns: List[str], filters: Optional[List[Tuple[str, str, str]]] = None
):
    """
    Loads the FARE data for a given year.

    Args:
    ----------
        project (str) : The name of the CASD project
        year (int): The year of the data to load.
        columns (List[str]): The columns to load.
        filters (Optional[List[Tuple[str, str, str]]], optional): The filters to apply. Each filter is a tuple of (column, operator, value). Defaults to None.

    Returns:
    ----------
        pd.DataFrame: The loaded data.
    """
    # Distinction du chemin selon l'année
    # Les données postérieures à 2021 n'étant pas disponibles, ce millésime est retenu en dernier ressort
    if year < 2022:
        # Chemin d'accès aux données
        table_path = f"\\\casd.fr\\casdfs\\Projets\\{project}\\Data\\Statistique annuelle d'entreprise_FARE_{year}"
        # Nom du jeu de données
        table_name = f"FARE{year}METH{year}.sas7bdat"
    else:
        # Chemin d'accès aux données
        table_path = f"\\\casd.fr\\casdfs\\Projets\\{project}\\Data\\Statistique annuelle d'entreprise_FARE_2021"
        # Nom du jeu de données
        table_name = f"FARE2021METH2021.sas7bdat"
    # Initialisation du loader
    loader = Loader()
    # Chargement des données
    data_fare = loader.load(
        path=os.path.join(table_path, table_name), columns=columns, filters=filters
    )

    return data_fare