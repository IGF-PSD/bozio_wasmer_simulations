# Importation des modules
# Modules de base
import numpy as np
import pandas as pd


# Fonction de création des variables d'intérêts concernant les effets emploi des réformes
def build_data_evol_emploi(
    data_source: pd.DataFrame,
    col_new_ct: str,
    col_evol: str,
    y0_elasticite: float,
    seuil_pallier_elasticite_smic: float,
    pallier_elasticite: float,
    name_elasticite: str,
    keep_elast: bool,
    to_concat: bool,
    col_ct="salaire_super_brut",
):
    """
    Creates variables of interest concerning the employment effects of reforms.

    Args:
        data_source (pd.DataFrame): The source data.
        col_new_ct (str): The column of the new cost of labor.
        col_evol (str): The column of the evolution.
        y0_elasticite (float): The initial value of the elasticity.
        seuil_pallier_elasticite_smic (float): The threshold of the elasticity in SMIC.
        pallier_elasticite (float): The step of the elasticity.
        name_elasticite (str): The name of the elasticity.
        keep_elast (bool): Whether to keep the elasticity.
        to_concat (bool): Whether to concatenate the result to the source data.
        col_ct (str, optional): The column of the cost of labor. Defaults to 'salaire_super_brut'.

    Returns:
        (pd.DataFrame): The data with the employment effects variables.
    """
    # Copie indépendante du jeu de données
    data_emploi = data_source[
        ["salaire_brut_smic", "quotite_de_travail", col_evol]
    ].copy()

    # Création du suffixe
    suffix = col_new_ct[(col_new_ct.find(col_ct) + len(col_ct) + 1) :]
    # Calcul de l'élasticité emploi (en nombre d'EQTP) selon la méthode suivante
    # y0_elasticite au niveau du smic, pallier_elasticite à partir de seuil_pallier_elasticite_smic smic et on tire une droite entre les deux
    data_emploi[f"elast_{name_elasticite}"] = np.maximum(
        y0_elasticite,
        np.where(
            data_emploi["salaire_brut_smic"] < seuil_pallier_elasticite_smic,
            (
                y0_elasticite
                + (pallier_elasticite - y0_elasticite)
                / (seuil_pallier_elasticite_smic - 1)
                * (data_emploi["salaire_brut_smic"] - 1)
            ),
            pallier_elasticite,
        ),
    )

    # Ajout de l'effet emploi
    data_emploi[f"effet_emploi_{name_elasticite}_{suffix}"] = (
        data_emploi["quotite_de_travail"]
        * data_emploi[f"elast_{name_elasticite}"]
        * data_emploi[col_evol]
    )

    # Déduction de la quotité de travail associée
    data_emploi[f"quotite_de_travail_{name_elasticite}_{suffix}"] = (
        data_emploi["quotite_de_travail"]
        + data_emploi[f"effet_emploi_{name_elasticite}_{suffix}"]
    )

    if to_concat:
        # Concaténation
        if not keep_elast:
            data_source = pd.concat(
                [
                    data_source,
                    data_emploi.drop(
                        [
                            "salaire_brut_smic",
                            "quotite_de_travail",
                            f"elast_{name_elasticite}",
                            col_evol,
                        ],
                        axis=1,
                    ),
                ],
                axis=1,
                join="outer",
            )
        else:
            data_source = pd.concat(
                [
                    data_source,
                    data_emploi.drop(
                        ["salaire_brut_smic", "quotite_de_travail", col_evol], axis=1
                    ),
                ],
                axis=1,
                join="outer",
            )
        # Suppression du jeu de données des emplois
        del data_emploi
        return data_source
    else:
        return data_emploi


# Fonction de création des variables d'intérêt en matière d'évolution du coût du travail
def build_data_evol_ct(
    data_source: pd.DataFrame,
    col_new_ct: str,
    to_concat: bool,
    col_ct: str = "salaire_super_brut",
):
    """
    Creates variables of interest concerning the evolution of the cost of labor.

    Args:
        data_source (pd.DataFrame): The source data.
        col_new_ct (str): The column of the new cost of labor.
        to_concat (bool): Whether to concatenate the result to the source data.
        col_ct (str, optional): The column of the cost of labor. Defaults to 'salaire_super_brut'.

    Returns:
        (pd.DataFrame): The data with the cost of labor evolution variables.
    """
    # Copie indépendante des grandeurs d'intérêt du jeu de données
    data_evol_ct = data_source[["siren", "weights", col_new_ct, col_ct]].copy()

    # Création du suffixe
    suffix = col_new_ct[(col_new_ct.find(col_ct) + len(col_ct) + 1) :]

    # Calcul de la différence du coût du travail
    data_evol_ct[f"diff_ct_{suffix}"] = data_evol_ct[col_new_ct] - data_evol_ct[col_ct]

    # Calcul de l'évolution des salaires
    data_evol_ct[f"evol_ct_{suffix}"] = (
        data_evol_ct[col_new_ct] - data_evol_ct[col_ct]
    ) / data_evol_ct[col_ct]

    # Ajout de l'évolution de la masse salariale au niveau de l'entreprise
    # Calcul de grandeurs d'intérêt
    data_evol_ct[f"diff_pond_ct_{suffix}"] = (
        data_evol_ct[col_new_ct] - data_evol_ct[col_ct]
    ).multiply(other=data_evol_ct["weights"])
    data_evol_ct["pond_ct"] = data_evol_ct[col_ct].multiply(
        other=data_evol_ct["weights"]
    )
    data_effet_siren = (
        data_evol_ct.groupby("siren")[f"diff_pond_ct_{suffix}"]
        .sum()
        .divide(other=data_evol_ct.groupby("siren")["pond_ct"].sum())
        .reset_index()
        .rename({0: f"evol_ms_{suffix}"}, axis=1)
    )
    # Appariement de la variation de la masse salariale par siren
    data_evol_ct = pd.merge(
        left=data_evol_ct,
        right=data_effet_siren,
        on="siren",
        how="left",
        validate="many_to_one",
    )
    # Suppression de la base des effets au niveau du Siren
    del data_effet_siren

    if to_concat:
        # Concaténarion au jeu de données d'origine
        data_source = pd.concat(
            [
                data_source,
                data_evol_ct.drop(
                    [
                        "siren",
                        "weights",
                        col_new_ct,
                        col_ct,
                        f"diff_pond_ct_{suffix}",
                        "pond_ct",
                    ],
                    axis=1,
                ),
            ],
            axis=1,
            join="outer",
        )
        # Suppression des jeux de données des évolutions
        del data_evol_ct
        return data_source
    else:
        return data_evol_ct
