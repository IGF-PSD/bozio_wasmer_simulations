# Importation des modules
# Modules de base
from typing import List

import numpy as np
import pandas as pd
from openfisca_france.model.prelevements_obligatoires.prelevements_sociaux.cotisations_sociales.exonerations import \
    TypesSecteurActivite
# Importation d'Openfisca
from openfisca_france.model.revenus.activite.salarie import (
    TypesCategorieSalarie, TypesContrat, TypesContratDeTravail)
# Modules du package
from bozio_wasmer_simulations.utils.passage import dict_nat_type_contrat


# Fonction de preprocessing des données pour les rendres compatibles avec Openfisca pour simuler les allègements généraux
def preprocess_dads_openfisca_ag(
    data_dads: pd.DataFrame,
    year: int,
    list_zonage_zrr: List[str],
    list_zonage_zrd: List[str],
):
    """
    Preprocesses the data to make it compatible with Openfisca for simulating general exemptions.

    Args:
    -----
        data_dads : pd.DataFrame
            The input data.
        year : int
            The year for which the data is being processed.
        list_zonage_zrr : List[str]
            A list of codes for rural revitalization zones.
        list_zonage_zrd : List[str]
            A list of codes for defense restructuring zones.

    Returns:
    --------
        pd.DataFrame
            The preprocessed data.
    """
    # Construction de colonnes d'intérêt

    # Création d'une pcs à deux chiffres
    data_dads["pcs_2"] = data_dads["pcs"].str[:2]
    # Conversion de la de la date de début de contrat de travail en datetime
    data_dads["date_fin_contrat"] = (
        pd.to_datetime(f"{year}-01-01", format="%Y-%m-%d")
        + pd.to_timedelta(arg=data_dads["datfin"], unit="D")
    ).dt.strftime("%Y-%m-%d")
    # Conversion en string
    data_dads["date_debut_contrat"] = (
        pd.to_datetime(data_dads["date_debut_contrat"], format="%Y-%m-%d")
        .dt.strftime("%Y-%m-%d")
        .fillna("1970-01-01")
    )

    # Secteur d'activite employeur (prend les valeurs 'agricole', 'non agricole' et 'non_renseigné')
    data_dads["secteur_activite_employeur"] = np.where(
        data_dads["a88"] == "01",
        TypesSecteurActivite.agricole,
        TypesSecteurActivite.non_agricole,
    )
    # Travailleur occasionnel agricole
    tode_condition = (data_dads["a88"] == "01") & data_dads["motifcdd"].isin(
        ["03", "04"]
    )
    data_dads["exoneration_cotisations_employeur_tode_eligibilite"] = np.where(
        tode_condition, True, False
    )
    data_dads["choix_exoneration_cotisations_employeur_agricole"] = np.where(
        tode_condition, True, False
    )
    data_dads["travailleur_occasionnel_agricole"] = np.where(
        tode_condition, True, False
    )
    # Indicatrices de zonage
    data_dads["zone_restructuration_defense"] = data_dads["comt"].isin(list_zonage_zrd)
    data_dads["zone_revitalisation_rurale"] = data_dads["comt"].isin(list_zonage_zrr)

    # Catégorie du salarié (prend des valeurs parmi 'prive_non_cadre', 'prive_cadre', 'public_titulaire_etat', 'public_titulaire_militaire', 'public_titulaire_territoriale', 'public_titulaire_hospitaliere', 'public_non_titulaire', 'non_pertinent')
    # On ne s'intéresse qu'au statut cadre/non cadre
    # 'domempl_empl' in ['1', '2', '3'],
    # 'nat_contrat'=='50' 104929
    data_dads["categorie_salarie"] = np.where(
        data_dads["pcs_2"] == "37",
        TypesCategorieSalarie.prive_cadre,
        TypesCategorieSalarie.prive_non_cadre,
    )
    # Contrat de travail (prend des valeurs parmi 'temps_plein', 'temps_partiel', 'forfait_heures_semaines', 'forfait_heures_mois', 'forfait_heures_annee', 'forfait_jours_annee', 'sans_objet')
    # On se restreint à l'implémentattion de 'temps_plein', 'temps_partiel'
    data_dads["contrat_de_travail"] = np.where(
        data_dads["eqtp"] < 1,
        TypesContratDeTravail.temps_partiel,
        TypesContratDeTravail.temps_plein,
    )
    data_dads["contrat_de_travail_fin"] = np.where(
        data_dads["datfin"] < 360, data_dads["date_fin_contrat"], "2099-12-31"
    )
    # Ajout du type de contrat de travail
    data_dads["contrat_de_travail_type"] = (
        data_dads["nat_contrat"].map(dict_nat_type_contrat).fillna(TypesContrat.cdi)
    )
    # Régime Alsace-Moselle
    data_dads["salarie_regime_alsace_moselle"] = np.where(
        data_dads["dept"].isin(["57", "67", "68"]), True, False
    )

    # Variables construites suivant le statut d'apprenti
    apprenti_condition = data_dads["contrat_travail"].isin(["04", "05"])
    # Construction de la rémunération
    data_dads["salaire_de_base"] = np.where(apprenti_condition, 0, data_dads["brut_s"])
    data_dads["remuneration_apprenti"] = np.where(
        apprenti_condition, data_dads["brut_s"], 0
    )
    # Date de début du contrat d'apprentissage
    data_dads["apprentissage_contrat_debut"] = np.where(
        apprenti_condition, data_dads["date_debut_contrat"], "1970-01-01"
    )
    # Statut d'apprenti
    data_dads["apprenti"] = np.where(apprenti_condition, True, False)

    # Variables construites suivant le statut de stagiaire
    stagiaire_condition = data_dads["nat_contrat"] == "29"
    # Nombre d'heures effectuées en stage
    data_dads["stage_duree_heures"] = np.where(
        stagiaire_condition, data_dads["nbheur"], 0
    )
    # Gratification
    data_dads["stage_gratification"] = np.where(
        stagiaire_condition, data_dads["salaire_de_base"], 0
    )
    data_dads["salaire_de_base"] = np.where(
        stagiaire_condition, 0, data_dads["salaire_de_base"]
    )

    # Ajout des taux de versement transport
    # data_dads['taux_versement_transport'] = 0.032
    # Ajout des taux d'accident du travail
    data_dads["taux_accident_travail"] = 0.0212

    # Ajout de l'effectif de l'entreprise
    data_dads = pd.merge(
        left=data_dads,
        right=data_dads.groupby(["siren"], as_index=False)["eqtp"]
        .sum()
        .rename({"eqtp": "effectif_entreprise"}, axis=1),
        on="siren",
        how="left",
        validate="many_to_one",
    )

    # Variables identiques à celles des DADS
    data_dads.rename(
        {
            "comr": "depcom",
            "comt": "depcom_entreprise",
            "date_debut_contrat": "contrat_de_travail_debut",
            "nbheur": "heures_remunerees_volume",
            "pepa": "prime_exceptionnelle_pouvoir_achat",
        },
        axis=1,
        inplace=True,
    )  # 'eqtp' : 'quotite_de_travail',
    # Ajout de la PPV si l'année de référence est bien 2022
    if year == 2022:
        data_dads.rename(
            {
                "ppv_defisc": "prime_partage_valeur_exoneree",
                "ppv_ndefisc": "prime_partage_valeur_non_exoneree",
            },
            axis=1,
            inplace=True,
        )
        data_dads["prime_partage_valeur_exoneree"] = data_dads[
            "prime_partage_valeur_exoneree"
        ].fillna(0)
        data_dads["prime_partage_valeur_non_exoneree"] = data_dads[
            "prime_partage_valeur_non_exoneree"
        ].fillna(0)

    # Complétion des Nan
    data_dads[["depcom", "depcom_entreprise"]] = data_dads[
        ["depcom", "depcom_entreprise"]
    ].fillna("")
    # data_dads['quotite_de_travail'] = data_dads['quotite_de_travail'].fillna(1)
    data_dads["salaire_de_base"] = data_dads["salaire_de_base"].fillna(0)

    return data_dads


# Fonction de post-processing des variables simulées
def preprocess_simulated_variables(data: pd.DataFrame):
    """
    Post-processes the simulated variables.

    Calculates the total exemptions, total reductions, and total exemptions and reductions.
    Corrects the unemployment benefits and calculates the gross salary.

    Args:
    -----
        data : pd.DataFrame
            The input data with simulated variables.

    Returns:
    --------
        pd.DataFrame
            The post-processed data.
    """
    # Somme des exonérations
    data["exonerations"] = (
        data[
            [
                "exoneration_cotisations_employeur_apprenti",
                "exoneration_cotisations_employeur_jei",
                "exoneration_cotisations_employeur_zrd",
                "exoneration_cotisations_employeur_zrr",
                "exoneration_cotisations_employeur_stagiaire",
            ]
        ].sum(axis=1)
        - data["exoneration_cotisations_employeur_tode"]
    )
    # Somme des allègements
    data["allegements"] = data[
        [
            "allegement_general",
            "allegement_cotisation_maladie",
            "allegement_cotisation_allocations_familiales",
        ]
    ].sum(axis=1)
    # Somme des exonérations et allègements
    data["exonerations_et_allegements"] = data[["exonerations", "allegements"]].sum(
        axis=1
    )
    # Correction des allocations chomages
    data["cotisations_employeur"] = (
        data["cotisations_employeur"]
        - data["chomage_employeur"]
        - data["assiette_cotisations_sociales"] * 0.0405
        - data[
            [
                "allegement_cotisation_maladie",
                "allegement_cotisation_allocations_familiales",
            ]
        ].sum(axis=1)
    )
    # Extraction des bandeaux de la variable hors allegements
    data["salaire_super_brut_hors_allegements"] = (
        data["salaire_super_brut_hors_allegements"]
        - data["chomage_employeur"]
        - data["assiette_cotisations_sociales"] * 0.0405
        + data[
            [
                "allegement_cotisation_maladie",
                "allegement_cotisation_allocations_familiales",
            ]
        ].sum(axis=1)
    )
    # Suppression de la colonne "chomage_employeur"
    # data.drop('chomage_employeur', axis=1, inplace=True)
    # Calcul du salaire super brut
    data["salaire_super_brut"] = (
        data["salaire_super_brut_hors_allegements"]
        - data["exonerations_et_allegements"]
        + data["prime_partage_valeur_exoneree"]
    )

    return data
