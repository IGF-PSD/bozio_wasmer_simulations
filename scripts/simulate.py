# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
import yaml
import sys
sys.path.append('..')

# Chargement du fichier de configurations
with open("../config.yaml") as file:
    config = yaml.safe_load(file)

# Modules ad_hoc
from bozio_wasmer_simulations.simulation.empirical.base import CoreSimulation, ReformSimulation
from bozio_wasmer_simulations.description.datasets import build_data_evol_ct

# Itération des simulations des cotisations, des allègements et des réformes
if __name__ == '__main__' :
    # Simulation des cotisations et allègements
    # Initialisation
    core_simulation = CoreSimulation(project=config['CASD']['INGENFI'])
    # Itération sur les années de config
    core_simulation.build(year_data=config['SIMULATIONS']['CORE']['YEAR_DATA'], year_simul=config['SIMULATIONS']['CORE']['YEAR_SIMUL'])

    # Simulation de réformes
    reform_simulation = ReformSimulation()
    # Itération
    reform_simulation.build(scenarios=config['SCENARIOS'], taux_bascule_vm=config['BASCULE_VM']['TAUX'], data=core_simulation.data_dads)


    ## Variables spécifiques au scénario jeunes
    if "JEUNES" in config['SCENARIOS'].keys() :
        # Suppression des variables erronnées sur les jeunes
        reform_simulation.data_simulation.drop(['exonerations_et_allegements_jeunes', 'salaire_super_brut_jeunes', 'salaire_super_brut_vm_jeunes'], axis=1, inplace=True)

        # Calcul de nouveaux allègements
        reform_simulation.data_simulation['new_allegement_jeunes'] = np.where((reform_simulation.data_simulation['age'] < 26) & (reform_simulation.data_simulation['salaire_brut_smic']<=1.3), np.maximum(reform_simulation.data_simulation['new_allegement_jeunes'], reform_simulation.data_simulation['allegements']) , reform_simulation.data_simulation['new_allegement_jeunes'])

        # Déduction des variables d'intérêt
        # Somme des exonérations et allègements
        reform_simulation.data_simulation['exonerations_et_allegements_jeunes'] = reform_simulation.data_simulation[['exonerations', 'new_allegement_jeunes']].sum(axis=1)

        # Calcul du salaire brut avec la réforme
        reform_simulation.data_simulation['salaire_super_brut_jeunes'] = reform_simulation.data_simulation['salaire_super_brut_hors_allegements'] - reform_simulation.data_simulation['exonerations_et_allegements_jeunes'] + reform_simulation.data_simulation['prime_partage_valeur_exoneree']
        reform_simulation.data_simulation['salaire_super_brut_vm_jeunes'] = reform_simulation.data_simulation['salaire_super_brut_hors_allegements'] + reform_simulation.data_simulation['versement_transport'] * (1 - config['BASCULE_VM']['TAUX']) - reform_simulation.data_simulation['exonerations_et_allegements_jeunes'] + reform_simulation.data_simulation['prime_partage_valeur_exoneree']

        # Construction des variables de variation du cout du travail
        reform_simulation.data_simulation = build_data_evol_ct(data_source=reform_simulation.data_simulation, col_new_ct=f'salaire_super_brut_jeunes', col_ct='salaire_super_brut', to_concat=True)
        reform_simulation.data_simulation = build_data_evol_ct(data_source=reform_simulation.data_simulation, col_new_ct=f'salaire_super_brut_vm_jeunes', col_ct='salaire_super_brut', to_concat=True)


    # Exportation
    reform_simulation.data_simulation.to_csv(f"../data/data_simulation_{config['SIMULATIONS']['CORE']['YEAR_SIMUL']}.csv")