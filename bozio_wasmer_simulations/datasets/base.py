# Importation des modules
# Modules de base
import os
from typing import List, Optional, Tuple, Union

import pandas as pd
import pyreadstat


# Classe de chargement des données
class Loader:
    """
    A class used to load data from 'csv', 'sas7bdat' and 'parquet' file formats.

    """

    # Initialisation
    def __init__(self) -> None:
        """
        Constructs all the necessary attributes for the Loader object.
        """
        pass

    # Fonction auxiliaire de déduction d'un masque à partir de filtres
    @staticmethod
    def _build_mask_from_filters(
        data: pd.DataFrame, filters: Union[List[Tuple], List[List[Tuple]]]
    ) -> pd.DataFrame:
        """
        Builds a mask from filters for a given dataframe.

        Args:
            data (pd.DataFrame):
                The dataframe to build the mask for.
            filters (Union[List[Tuple], List[List[Tuple]]]):
                The filters to apply. Each filter is a tuple of (column, operator, value).
                If filters is a list of tuples, all filters are applied in conjunction.
                If filters is a list of lists of tuples, each list is applied in disjunction.

        Returns:
            (pd.DataFrame): The mask that can be used to filter the dataframe.
        """
        # Si filters est une liste de tuples
        if all(isinstance(i, tuple) for i in filters) and isinstance(filters, list):
            # Initialisation de la série résultat
            mask = pd.Series(True, index=data.index)
            # Parcours des filters
            for conjonction_filter in filters:
                col, op, val = conjonction_filter
                mask &= eval(f"data['{col}'] {op} {val}")
        # Si filters est une liste de liste de tuples
        elif all(
            isinstance(i, list) and all(isinstance(j, tuple) for j in i)
            for i in filters
        ) and isinstance(filters, list):
            # Initialisation de la série résultat
            mask = pd.Series(False, index=data.index)
            # Parcours des filtres
            for disjonction_filter in filters:
                if isinstance(disjonction_filter, tuple):
                    col, op, val = disjonction_filter
                    mask |= eval(f"data['{col}'] {op} {val}")
                elif isinstance(disjonction_filter, list):
                    submask = pd.Series(True, index=data.index)
                    for conjonction_filter in disjonction_filter:
                        col, op, val = conjonction_filter
                        submask &= eval(f"data['{col}'] {op} {val}")
                    mask |= submask
        else:
            raise TypeError(
                f"Invalid type for 'filters' : {filters}. Shoud be in [List[Tuple], List[List[Tuple]]]"
            )
        return mask

    # Fonction de chargement des données SAS
    def read_sas(
        self,
        path: os.PathLike,
        columns: Optional[Union[None, List[str]]] = None,
        filters: Optional[Union[None, List[str], List[List[str]]]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Loads data from a SAS file.

        Args:
            path (os.PathLike):
                The path to the SAS file.
            columns (Optional[Union[None, List[str]]], optional):
                The columns to load. If None, all columns are loaded.
            filters (Optional[Union[None, List[str], List[List[str]]]], optional):
                The filters to apply. Each filter is a tuple of (column, operator, value).
                If filters is a list of tuples, all filters are applied in conjunction.
                If filters is a list of lists of tuples, each list is applied in disjunction.
            **kwargs
                Additional arguments to pass to the pyreadstat.read_sas7bdat function.

        Returns:
            (pd.DataFrame): The loaded data.
        """
        # Chargement des données
        data, _ = pyreadstat.read_sas7bdat(path, usecols=columns, **kwargs)
        # Filtrage des données
        if filters is not None:
            data = data.loc[self._build_mask_from_filters(data=data, filters=filters)]
        return data

    # Fonction de chargement des données CSV
    def read_csv(
        self,
        path: os.PathLike,
        columns: Optional[Union[None, List[str]]] = None,
        filters: Optional[Union[None, List[str], List[List[str]]]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Loads data from a CSV file.

        Args:
            path (os.PathLike):
                The path to the CSV file.
            columns (Optional[Union[None, List[str]]], optional):
                The columns to load. If None, all columns are loaded.
            filters (Optional[Union[None, List[str], List[List[str]]]], optional):
                The filters to apply. Each filter is a tuple of (column, operator, value).
                If filters is a list of tuples, all filters are applied in conjunction.
                If filters is a list of lists of tuples, each list is applied in disjunction.
            **kwargs
                Additional arguments to pass to the pd.read_csv function.

        Returns:
            (pd.DataFrame): The loaded data.
        """
        # Chargement des données
        data = pd.read_csv(path, usecols=columns, **kwargs)
        # Filtrage des données
        if filters is not None:
            data = data.loc[self._build_mask_from_filters(data=data, filters=filters)]
        return data

    # Fonction générale de chargement des données
    def load(
        self,
        path: os.PathLike,
        columns: Optional[Union[None, List[str]]] = None,
        filters: Optional[Union[None, List[str], List[List[str]]]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Loads data from a file based on its extension.

        Args:
            path (os.PathLike):
                The path to the file.
            columns (Optional[Union[None, List[str]]], optional):
                The columns to load. If None, all columns are loaded.
            filters (Optional[Union[None, List[str], List[List[str]]]], optional):
                The filters to apply. Each filter is a tuple of (column, operator, value).
                If filters is a list of tuples, all filters are applied in conjunction.
                If filters is a list of lists of tuples, each list is applied in disjunction.
            **kwargs
                Additional arguments to pass to the appropriate function.

        Returns:
            (pd.DataFrame): The loaded data.

        Raises:
            ValueError: If the file extension is not 'parquet', 'sas7bdat', or 'csv'.
        """
        # Extraction de l'extension du fichier à charger
        extension = path.split(".")[-1]
        # Distinction de la méthode de chargement suivant l'extension
        if (extension == "parquet") | (os.path.isdir(path)):
            data = pd.read_parquet(path, columns=columns, filters=filters, **kwargs)
        elif extension == "sas7bdat":
            data = self.read_sas(path=path, columns=columns, filters=filters, **kwargs)
        elif extension == "csv":
            data = self.read_csv(path=path, columns=columns, filters=filters, **kwargs)
        else:
            raise ValueError(
                f"Unknown extention : {extension}. Should be in ['parquet', 'sas7bdat', 'csv']"
            )

        return data