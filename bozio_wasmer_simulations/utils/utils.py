# Importation des modules
# Modules de base
import logging
import os
from collections.abc import Mapping
from copy import deepcopy


# Fonction auxiliaire d'appariement récursif des dictionnaires
def _recursive_dict_update(dict_source: dict, dict_update: dict) -> dict:
    """
    Recursively updates a dictionary with values from another dictionary.

    Args:
        dict_source (dict):
            The source dictionary.
        dict_update (dict):
            The dictionary containing the updates.

    Returns:
    --------
        dict: The updated dictionary.
    """
    # Initialisation du dictionnaire résultat
    dict_result = deepcopy(dict_source)

    # Mise à jour de l'ensemble des clés et valeurs
    for key, value in dict_update.items():
        # Distinction suivant que la valeur est ou non elle-même un dictionnaire
        if isinstance(value, Mapping):
            dict_result[key] = _recursive_dict_update(
                dict_source=dict_result.get(key, {}), dict_update=value
            )
        else:
            dict_result[key] = deepcopy(dict_update[key])

    return dict_result


# Fonction auxiliaire d'initialisation d'un logger
def _init_logger(filename: os.PathLike) -> logging.Logger:
    """
    Initializes a logger.

    Configures the logging format, level, and file handler.
    Creates the directory for the log file if it does not exist.

    Args:
        filename (os.PathLike):
            The path to the log file.

    Returns:
        logging.Logger: The initialized logger.
    """
    # Configuration du logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding="utf-8",
        level=logging.INFO,
    )

    # Vérification de l'existence du dossier pour le fichier de log
    log_directory = os.path.dirname(filename)
    # Création du chemin s'il n'existe pas déjà
    if (not os.path.exists(log_directory)) & (log_directory != ""):
        os.makedirs(log_directory)

    # Configuration du fichier de logs
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)

    # Initialisation du logger
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    return logger
