from openfisca_france.model.revenus.activite.salarie import TypesContrat

# Dictionnaire de passage entre la nature du contrat des DADS et le type de contrat de travail d'Openfisca
dict_nat_type_contrat = {
    "01": TypesContrat.cdi,
    "02": TypesContrat.cdd,
    "03": TypesContrat.ctt,
    "07": TypesContrat.cdi,
    "08": TypesContrat.cdi,
    "09": TypesContrat.cdi,
    "10": TypesContrat.cdd,
    "29": TypesContrat.formation,
    # '32' : '', # Contrat d'appui au projet d'entreprise
    "50": TypesContrat.aucun,
    # '60' : '', # Contrat d'engagement éducatif
    # '70' : ', # Contrat de soutien et d'aide par le travail
    "80": TypesContrat.aucun,
    "81": TypesContrat.aucun,
    "82": TypesContrat.cdi,
    # '89' : '', # Volontariat de service civique
    # '90' : '', # Autre nature de contrat, convention, mandat
    "91": TypesContrat.cdi,
    "92": TypesContrat.cdd,
    # '93' : '', # Ligne de service
    "99": TypesContrat.aucun,
}
