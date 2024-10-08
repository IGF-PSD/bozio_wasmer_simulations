# Importation des modules
# Modules de base
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Fonction construisant les pondération à partir des données ACOSS
def add_weights_eqtp_accos(
    data_dads: pd.DataFrame,
    year: int,
    var_eqtp: str,
    var_sal_brut: str,
    var_smic_proratise: str = "smic_proratise",
) -> pd.DataFrame:
    """
    Adds weights to the data based on ACOSS data.

    This function calculates the salary as a proportion of the SMIC and creates salary brackets.
    It then calculates the current weights and the target weights based on ACOSS data.
    The weights are calculated as the ratio of the target weights to the current weights.
    The function returns the data with the weights added.

    Args:
        data_dads (pd.DataFrame):
            The input data.
        year (int):
            The year for which the weights are calculated.
        var_eqtp (str):
            The name of the variable containing the employment data.
        var_sal_brut (str):
            The name of the variable containing the gross salary data.
        var_smic_proratise (str, optional):
            The name of the variable containing the prorated SMIC data. Default is 'smic_proratise'.

    Returns:
        (pd.DataFrame): The input data with the weights added.
    """
    # Salaire en proportion du SMIC
    if "salaire_brut_smic" not in data_dads.columns:
        data_dads["salaire_brut_smic"] = (
            data_dads[var_sal_brut] / data_dads[var_smic_proratise]
        )

    # Création de tranche de salaire brut en proportion du SMIC
    data_dads["tranche_salaire_brut_smic_100"] = (
        np.floor(data_dads["salaire_brut_smic"] * 100) / 100
    )
    data_dads["tranche_salaire_brut_smic_10"] = (
        np.floor(data_dads["salaire_brut_smic"] * 10) / 10
    )
    data_dads["tranche_salaire_brut_smic_1"] = np.floor(data_dads["salaire_brut_smic"])
    # Combination des tranches de 10 % et de 1 % en dessous de 2 SMIC et de 1 au dessus de 10 SMIC
    data_dads["tranche_salaire_brut_smic_100"] = np.where(
        data_dads["salaire_brut_smic"] < 2,
        data_dads["tranche_salaire_brut_smic_100"],
        data_dads["tranche_salaire_brut_smic_10"],
    )
    data_dads["tranche_salaire_brut_smic_100"] = np.where(
        data_dads["salaire_brut_smic"] > 10,
        data_dads["tranche_salaire_brut_smic_1"],
        data_dads["tranche_salaire_brut_smic_100"],
    )
    # Suppression des tranches de 10 % et de 1
    data_dads.drop(
        ["tranche_salaire_brut_smic_10", "tranche_salaire_brut_smic_1"],
        axis=1,
        inplace=True,
    )
    # Création des tranches entre 5 et 7.5 et entre 7,5 et 10
    data_dads.loc[
        (data_dads["salaire_brut_smic"] >= 5) & (data_dads["salaire_brut_smic"] < 7.5),
        "tranche_salaire_brut_smic_100",
    ] = 5
    data_dads.loc[
        (data_dads["salaire_brut_smic"] >= 7.5) & (data_dads["salaire_brut_smic"] < 10),
        "tranche_salaire_brut_smic_10",
    ] = 7.5

    # Ajout des poids
    # Calcul des poids actuels
    # data_weights = data_dads.groupby('tranche_salaire_brut_smic_10')[['salaire_de_base']].sum().rename({'salaire_de_base' : 'current_weight'}, axis=1).reset_index()
    data_weights = (
        data_dads.groupby("tranche_salaire_brut_smic_100")[[var_eqtp]]
        .sum()
        .rename({var_eqtp: "current_weight"}, axis=1)
        .reset_index()
    )

    # Poids de l'année
    data_weights_acoss = pd.read_excel(
        os.path.join(FILE_PATH.parents[3], f"data/distributions_{year}_missionBW.xlsx"),
        sheet_name="distributions",
        skiprows=3,
    )
    # Sélection et renomination des colonnes d'intérêt
    data_weights_acoss = data_weights_acoss[
        ["Étiquettes de lignes", "Somme de etp_f2"]
    ].rename(
        {
            "Étiquettes de lignes": "tranche_salaire_brut_smic_100",
            "Somme de etp_f2": "target_weight",
        },
        axis=1,
    )
    # Conversion en numérique
    data_weights_acoss["tranche_salaire_brut_smic_100"] = pd.to_numeric(
        data_weights_acoss["tranche_salaire_brut_smic_100"], errors="coerce"
    )
    # Suppression des lignes de total imputées par dans Nan
    data_weights_acoss.dropna(
        subset=["tranche_salaire_brut_smic_100"], axis=0, how="any", inplace=True
    )
    data_weights_acoss = data_weights_acoss.groupby(
        "tranche_salaire_brut_smic_100", as_index=False
    )[["target_weight"]].sum()
    # Appariement
    data_weights = pd.merge(
        left=data_weights,
        right=data_weights_acoss,
        on="tranche_salaire_brut_smic_100",
        how="outer",
        validate="one_to_one",
    ).fillna(0)
    # Création des poids
    data_weights["weights"] = (
        data_weights["target_weight"] / data_weights["current_weight"]
    )
    # Retraitement des + infini
    data_weights.loc[np.isinf(data_weights["weights"]), "weights"] = 0
    # Appariement avec les DADS
    data_dads = pd.merge(
        left=data_dads,
        right=data_weights[["tranche_salaire_brut_smic_100", "weights"]],
        how="left",
        on="tranche_salaire_brut_smic_100",
        validate="many_to_one",
    ).drop("tranche_salaire_brut_smic_100", axis=1)

    # Non pondération des salariés sous le SMIC
    data_dads.loc[data_dads["salaire_brut_smic"] < 1, "weights"] = 0
    # Ajout des poids des entreprises
    data_dads = pd.merge(
        left=data_dads,
        right=data_dads.groupby("siren", as_index=False)["weights"]
        .sum()
        .rename({"weights": "firm_total_weights"}, axis=1),
        on="siren",
        validate="many_to_one",
    )

    return data_dads
