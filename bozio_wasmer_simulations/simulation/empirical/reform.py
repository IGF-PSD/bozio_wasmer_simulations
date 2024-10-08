# Importation des modules
# Openfisca
from openfisca_core.reforms import Reform
# Importation des utilitaires d'Openfisca
from openfisca_france.model.base import (ADD, MONTH, Individu,  # , Reform
                                         ParameterNode, Variable,
                                         calculate_output_add, max_, min_,
                                         not_, round_,
                                         set_input_divide_by_period)
from openfisca_france.model.prelevements_obligatoires.prelevements_sociaux.cotisations_sociales.allegements import (
    compute_allegement_annuel, compute_allegement_anticipe,
    compute_allegement_progressif)


# Fonction permettant de simuler une réforme structure avec l'ajout d'une nouvelle tranche
def create_and_apply_structural_reform_ag(
    tax_benefit_system, dict_params, period="2000-01-01"
):
    """
    Creates and applies a structural reform to add a new exemption.

    Depending on the type of general exemption to add, this function modifies the parameters of the tax-benefit system
    and adds a new variable to calculate. The type of exemption can be 'fillon', 'convexe_fillon', 'allocations_familiales',
    or 'maladie'.

    Args:
        tax_benefit_system (TaxBenefitSystem):
            The initial tax-benefit system.
        dict_params (dict):
            A dictionary containing the parameters of the reform. The 'TYPE' key indicates the type of exemption to add.
        period (str, optional):
            The period for which the parameters are defined. Default is '2000-01-01'.

    Returns:
        (TaxBenefitSystem): The reformed tax-benefit system.

    Raises:
        ValueError: If the type of exemption is not 'fillon', 'convexe_fillon', 'allocations_familiales', or 'maladie'.
    """

    # Distinction suivant le type d'allègement général à ajouter
    if dict_params["TYPE"] == "fillon":
        # Définition de la fonction de modification des paramètres
        def modify_new_fillon_parameters(parameters):
            # Définition des paramètres à ajouter
            reform_parameters_subtree = ParameterNode(
                name="new_fillon",
                data={
                    "ensemble_des_entreprises": {
                        "plafond": {
                            "values": {
                                period: {"value": dict_params["PARAMS"]["PLAFOND"]}
                            }
                        },
                        "entreprises_de_50_salaries_et_plus": {
                            "values": {
                                period: {
                                    "value": dict_params["PARAMS"][
                                        "TAUX_50_SALARIES_ET_PLUS"
                                    ]
                                }
                            }
                        },
                        "entreprises_de_moins_de_50_salaries": {
                            "values": {
                                period: {
                                    "value": dict_params["PARAMS"][
                                        "TAUX_MOINS_DE_50_SALARIES"
                                    ]
                                }
                            }
                        },
                    }
                },
            )
            # Ajout des paramètres au noeud d'intéret
            parameters.prelevements_sociaux.reductions_cotisations_sociales.add_child(
                "new_fillon", reform_parameters_subtree
            )

            return parameters

        # Définition de la classe de réforme
        class StructuralReformFillon(Reform):
            name = "Ajout d'un nouvel allègement général de type 'Allègement Fillon'"

            # Application de la réforme
            def apply(self):
                # Ajout des paramètres des allègements de cotisation
                self.modify_parameters(modifier_function=modify_new_fillon_parameters)
                # Ajout de la variable à calculer
                self.add_variable(new_allegement_fillon)

        # Application de la réforme au système socio-fiscal
        reformed_tax_benefit_system = StructuralReformFillon(tax_benefit_system)

        return reformed_tax_benefit_system

    elif dict_params["TYPE"] == "convexe_fillon":
        # Définition de la fonction de modification des paramètres
        def modify_convexe_fillon_parameters(parameters):
            # Définition des paramètres à ajouter
            reform_parameters_subtree = ParameterNode(
                name="convexe_fillon",
                data={
                    "ensemble_des_entreprises": {
                        "plafond": {
                            "values": {
                                period: {"value": dict_params["PARAMS"]["PLAFOND"]}
                            }
                        },
                        "entreprises_de_50_salaries_et_plus": {
                            "values": {
                                period: {
                                    "value": dict_params["PARAMS"][
                                        "TAUX_50_SALARIES_ET_PLUS"
                                    ]
                                }
                            }
                        },
                        "entreprises_de_moins_de_50_salaries": {
                            "values": {
                                period: {
                                    "value": dict_params["PARAMS"][
                                        "TAUX_MOINS_DE_50_SALARIES"
                                    ]
                                }
                            }
                        },
                        "exposant": {
                            "values": {
                                period: {"value": dict_params["PARAMS"]["EXPOSANT"]}
                            }
                        },
                    }
                },
            )
            # Ajout des paramètres au noeud d'intéret
            parameters.prelevements_sociaux.reductions_cotisations_sociales.add_child(
                "convexe_fillon", reform_parameters_subtree
            )

            return parameters

        # Définition de la classe de réforme
        class StructuralReformConvexeFillon(Reform):
            name = "Ajout d'un nouvel allègement général de type 'Allègement Fillon' dont la convexité est modifiée"

            # Application de la réforme
            def apply(self):
                # Ajout des paramètres des allègements de cotisation
                self.modify_parameters(
                    modifier_function=modify_convexe_fillon_parameters
                )
                # Ajout de la variable à calculer
                self.add_variable(new_allegement_convexe_fillon)

        # Application de la réforme au système socio-fiscal
        reformed_tax_benefit_system = StructuralReformConvexeFillon(tax_benefit_system)

        return reformed_tax_benefit_system

    elif dict_params["TYPE"] == "allocations_familiales":

        # Définition de la fonction de modification des paramètres
        def modify_new_allocations_familiales_parameters(parameters):
            # Définition des paramètres à ajouter
            reform_parameters_subtree = ParameterNode(
                name="new_allegement_cotisation_allocations_familiales",
                data={
                    "plafond_smic": {
                        "values": {period: {"value": dict_params["PARAMS"]["PLAFOND"]}}
                    },
                    "reduction": {
                        "values": {period: {"value": dict_params["PARAMS"]["TAUX"]}}
                    },
                },
            )
            # Ajout des paramètres au noeud d'intéret
            parameters.prelevements_sociaux.reductions_cotisations_sociales.add_child(
                "new_allegement_cotisation_allocations_familiales",
                reform_parameters_subtree,
            )

            return parameters

        # Définition de la classe de réforme
        class StructuralReformAllocationsFamiliales(Reform):
            name = "Ajout d'un nouvel allègement général de type 'Allègement cotisations allocations familiales'"

            # Application de la réforme
            def apply(self):
                # Ajout des paramètres des allègements de cotisation
                self.modify_parameters(
                    modifier_function=modify_new_allocations_familiales_parameters
                )
                # Ajout de la variable à calculer
                self.add_variable(new_allegement_allocations_familiales)

        # Application de la réforme du système socio-fiscal
        reformed_tax_benefit_system = StructuralReformAllocationsFamiliales(
            tax_benefit_system
        )

        return reformed_tax_benefit_system

    elif dict_params["TYPE"] == "maladie":
        # Définition de la fonction de modification des paramètres
        def modify_new_maladie_parameters(parameters):
            # Définition des paramètres à ajouter
            reform_parameters_subtree = ParameterNode(
                name="new_mmid",
                data={
                    "plafond": {
                        "values": {period: {"value": dict_params["PARAMS"]["PLAFOND"]}}
                    },
                    "taux": {
                        "values": {period: {"value": dict_params["PARAMS"]["TAUX"]}}
                    },
                },
            )
            # Ajout des paramètres au noeud d'intéret
            parameters.prelevements_sociaux.reductions_cotisations_sociales.add_child(
                "new_mmid", reform_parameters_subtree
            )

            return parameters

        # Définition de la classe de réforme
        class StructuralReformMaladie(Reform):
            name = "Ajout d'un nouvel allègement général de type 'Allègement cotisations maladie'"

            # Application de la réforme
            def apply(self):
                # Ajout des paramètres des allègements de cotisation
                self.modify_parameters(modifier_function=modify_new_maladie_parameters)
                # Ajout de la variable à calculer
                self.add_variable(new_allegement_cotisation_maladie)

        # Application de la réforme du système socio-fiscal
        reformed_tax_benefit_system = StructuralReformMaladie(tax_benefit_system)

        return reformed_tax_benefit_system
    else:
        raise ValueError(
            "Unknown allegement type : should be in ['fillon', 'allocations_familiales', 'maladie']"
        )


# Fonction de computation du nouvel allègement
def compute_new_allegement_fillon(individu, period, parameters):
    # Attention la période couvre plusieurs mois
    first_month = period.first_month

    # Extraction des informations d'intérêt de l'individu
    assiette = individu("assiette_allegement", period, options=[ADD])
    smic_proratise = individu("smic_proratise", period, options=[ADD])
    effectif_entreprise = individu("effectif_entreprise", first_month)

    # Calcul du taux
    # Extraction des paramètres
    new_fillon = parameters(
        period
    ).prelevements_sociaux.reductions_cotisations_sociales.new_fillon
    # Définition du seuil
    seuil = new_fillon.ensemble_des_entreprises.plafond
    # Définition du statut de petite entreprise
    petite_entreprise = effectif_entreprise < 50
    # Définition du taux maximal
    tx_max = (
        new_fillon.ensemble_des_entreprises.entreprises_de_50_salaries_et_plus
        * not_(petite_entreprise)
        + new_fillon.ensemble_des_entreprises.entreprises_de_moins_de_50_salaries
        * petite_entreprise
    )

    if seuil <= 1:
        return 0

    # Calcul du ratio smic-salaire
    ratio_smic_salaire = smic_proratise / (assiette + 1e-16)

    # Calcul du taux fillon (règle d'arrondi : 4 décimales au dix millième le plus proche)
    taux_new_fillon = round_(
        tx_max * min_(1, max_(seuil * ratio_smic_salaire - 1, 0) / (seuil - 1)), 4
    )

    return taux_new_fillon * assiette


# Définition de la nouvelle variable à calculer
class new_allegement_fillon(Variable):
    value_type = float
    entity = Individu
    label = "Nouvel allègement général utilisant la même assiette que les allègements de cotisations Fillon"
    definition_period = MONTH
    calculate_output = (calculate_output_add,)
    set_input = set_input_divide_by_period

    def formula(individu, period, parameters):
        # Extraction des caractéristiques d'intérêt de l'individu
        stagiaire = individu("stagiaire", period)
        apprenti = individu("apprenti", period)
        allegement_mode_recouvrement = individu(
            "allegement_general_mode_recouvrement", period
        )
        exoneration_cotisations_employeur_jei = individu(
            "exoneration_cotisations_employeur_jei", period
        )
        exoneration_cotisations_employeur_tode = individu(
            "exoneration_cotisations_employeur_tode", period
        )
        non_cumulee = not_(
            exoneration_cotisations_employeur_jei
            + exoneration_cotisations_employeur_tode
        )

        # Imputation de la valeur suivant les trois modes de recouvrement
        allegement = switch_on_allegement_mode(
            individu,
            period,
            parameters,
            allegement_mode_recouvrement,
            "new_allegement_fillon",
        )
        # allegement = compute_new_allegement_fillon(individu, period, parameters)

        return allegement * not_(stagiaire) * not_(apprenti) * non_cumulee


# Fonction de computation du nouvel allègement
def compute_new_allegement_allocations_familiales(individu, period, parameters):
    # Extraction des caractéristiques d'intérêt de l'individu
    ssiette = individu("assiette_allegement", period, options=[ADD])
    smic_proratise = individu("smic_proratise", period, options=[ADD])

    # Extraction des paramètres d'intérêt
    new_allegement_cotisation_allocations_familiales = parameters(
        period
    ).prelevements_sociaux.reductions_cotisations_sociales.new_allegement_cotisation_allocations_familiales

    # Montant de l'allègement
    return (
        (
            assiette
            < new_allegement_cotisation_allocations_familiales.plafond_smic
            * smic_proratise
        )
        * new_allegement_cotisation_allocations_familiales.reduction
        * assiette
    )


# Définition de la nouvelle variable à calculer
class new_allegement_allocations_familiales(Variable):
    value_type = float
    entity = Individu
    label = "Nouvel allègement général utilisant la même assiette que les allègements de cotisations allocations familiales"
    definition_period = MONTH
    set_input = set_input_divide_by_period

    def formula(individu, period, parameters):
        # Extraction des caractéristiques d'intérêt de l'individu
        stagiaire = individu("stagiaire", period)
        apprenti = individu("apprenti", period)
        allegement_mode_recouvrement = individu(
            "allegement_cotisation_allocations_familiales_mode_recouvrement", period
        )
        exoneration_cotisations_employeur_jei = individu(
            "exoneration_cotisations_employeur_jei", period
        )
        non_cumulee = not_(exoneration_cotisations_employeur_jei)
        choix_exoneration_cotisations_employeur_agricole = individu(
            "choix_exoneration_cotisations_employeur_agricole", period
        )

        # Calcul de l'allègement suivant les trois modes de recouvrement possibles
        # allegement = switch_on_allegement_mode(individu, period, parameters, allegement_mode_recouvrement, 'new_allegement_allocations_familiales',)
        allegement = compute_new_allegement_allocations_familiales(
            individu, period.this_year, parameters
        )

        return (
            allegement
            * not_(stagiaire)
            * not_(apprenti)
            * non_cumulee
            * not_(choix_exoneration_cotisations_employeur_agricole)
        )


# Fonction de computation du nouvel allègement
def compute_new_allegement_cotisation_maladie(individu, period, parameters):
    # Extraction des paramètres d'intérêt
    new_allegement_mmid = parameters(
        period
    ).prelevements_sociaux.reductions_cotisations_sociales.alleg_gen.new_mmid

    # Extraction des caractéristiques d'intérêt de l'individu
    assiette_allegement = individu("assiette_allegement", period, options=[ADD])
    smic_proratise = individu("smic_proratise", period, options=[ADD])
    # Extraction du plafond
    plafond_allegement_mmid = new_allegement_mmid.plafond  # en nombre de smic
    # Calcul du sous-plafond
    sous_plafond = assiette_allegement <= (smic_proratise * plafond_allegement_mmid)

    return sous_plafond * new_allegement_mmid.taux * assiette_allegement


# Définition de la nouvelle variable à calculer
class new_allegement_cotisation_maladie(Variable):
    value_type = float
    entity = Individu
    label = "Nouvel allègement général utilisant la même assiette que les allègements de cotisations maladie (ex-CICE)"
    definition_period = MONTH
    set_input = set_input_divide_by_period

    def formula(individu, period, parameters):
        # Extraction des caractéristiques d'intérêt de l'individu
        # Si l'employeur fait le choix de la TO-DE alors celle-ci remplace l'allègement de cotisation maladie.
        choix_exoneration_cotisations_employeur_agricole = individu(
            "choix_exoneration_cotisations_employeur_agricole", period
        )
        # Mode de recouvrement propose 3 modes de paiement possibles
        allegement_mode_recouvrement = individu(
            "allegement_cotisation_maladie_mode_recouvrement", period
        )

        # Calcul de l'allègement
        # allegement = switch_on_allegement_mode(individu, period, parameters, allegement_mode_recouvrement, 'new_allegement_cotisation_maladie')
        allegement = compute_new_allegement_cotisation_maladie(
            individu, period.this_year, parameters
        )

        return allegement * not_(choix_exoneration_cotisations_employeur_agricole)


# Fonction de computation du nouvel allègement
def compute_new_allegement_convexe_fillon(individu, period, parameters):
    # Attention la période couvre plusieurs mois
    first_month = period.first_month

    # Extraction des informations d'intérêt de l'individu
    assiette = individu("assiette_allegement", period, options=[ADD])
    smic_proratise = individu("smic_proratise", period, options=[ADD])
    effectif_entreprise = individu("effectif_entreprise", first_month)

    # Calcul du taux
    # Extraction des paramètres
    convexe_fillon = parameters(
        period
    ).prelevements_sociaux.reductions_cotisations_sociales.convexe_fillon
    # Définition du seuil
    seuil = convexe_fillon.ensemble_des_entreprises.plafond
    # Définition de l'exposant
    exposant = convexe_fillon.ensemble_des_entreprises.exposant
    # Définition du statut de petite entreprise
    petite_entreprise = effectif_entreprise < 50
    # Définition du taux maximal
    tx_max = (
        convexe_fillon.ensemble_des_entreprises.entreprises_de_50_salaries_et_plus
        * not_(petite_entreprise)
        + convexe_fillon.ensemble_des_entreprises.entreprises_de_moins_de_50_salaries
        * petite_entreprise
    )

    if seuil <= 1:
        return 0

    # Calcul du ratio smic-salaire
    ratio_smic_salaire = smic_proratise / (assiette + 1e-16)

    # Calcul du taux fillon (règle d'arrondi : 4 décimales au dix millième le plus proche)
    taux_convexe_fillon = round_(
        tx_max
        * (min_(1, max_(seuil * ratio_smic_salaire - 1, 0) / (seuil - 1))) ** exposant,
        4,
    )
    # taux_convexe_fillon = round_((tx_max* min_(1, max_(seuil * ratio_smic_salaire-1, 0)/(seuil-1)))**exposant, 4)
    # taux_convexe_fillon = round_((tx_max* max_(seuil * ratio_smic_salaire-1, 0)/(seuil-1))**exposant, 4)

    return taux_convexe_fillon * assiette


# Définition de la nouvelle variable à calculer
class new_allegement_convexe_fillon(Variable):
    value_type = float
    entity = Individu
    label = "Nouvel allègement général utilisant la même assiette que les allègements de cotisations Fillon mais en modifiant sa convexité"
    definition_period = MONTH
    calculate_output = (calculate_output_add,)
    set_input = set_input_divide_by_period

    def formula(individu, period, parameters):
        # Extraction des caractéristiques d'intérêt de l'individu
        stagiaire = individu("stagiaire", period)
        apprenti = individu("apprenti", period)
        allegement_mode_recouvrement = individu(
            "allegement_general_mode_recouvrement", period
        )
        exoneration_cotisations_employeur_jei = individu(
            "exoneration_cotisations_employeur_jei", period
        )
        exoneration_cotisations_employeur_tode = individu(
            "exoneration_cotisations_employeur_tode", period
        )
        non_cumulee = not_(
            exoneration_cotisations_employeur_jei
            + exoneration_cotisations_employeur_tode
        )

        # Imputation de la valeur suivant les trois modes de recouvrement
        allegement = switch_on_allegement_mode(
            individu,
            period,
            parameters,
            allegement_mode_recouvrement,
            "new_allegement_convexe_fillon",
        )
        # allegement = compute_convexe_allegement_fillon(individu, period, parameters)

        return allegement * not_(stagiaire) * not_(apprenti) * non_cumulee


def switch_on_allegement_mode(
    individu, period, parameters, mode_recouvrement, variable_name
):
    compute_function = globals()["compute_" + variable_name]

    TypesAllegementModeRecouvrement = mode_recouvrement.possible_values
    recouvrement_fin_annee = (
        mode_recouvrement == TypesAllegementModeRecouvrement.fin_d_annee
    )
    recouvrement_anticipe = (
        mode_recouvrement == TypesAllegementModeRecouvrement.anticipe
    )
    recouvrement_progressif = (
        mode_recouvrement == TypesAllegementModeRecouvrement.progressif
    )

    return (
        (
            recouvrement_fin_annee
            * compute_allegement_annuel(
                individu, period, parameters, variable_name, compute_function
            )
        )
        + (
            recouvrement_anticipe
            * compute_allegement_anticipe(
                individu, period, parameters, variable_name, compute_function
            )
        )
        + (
            recouvrement_progressif
            * compute_allegement_progressif(
                individu, period, parameters, variable_name, compute_function
            )
        )
    )
