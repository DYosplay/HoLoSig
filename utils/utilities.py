    
import numpy as np
from numpy import typing as npt
from typing import Dict, Any
import json
import os

def traceback(acc_cost_matrix : npt.DTypeLike):
    """ Encontra a path (matching) dado uma matriz de custo acumulado obtida a partir do cálculo do DTW.
    Args:
        acc_cost_matrix (npt.DTypeLike): matriz de custo acumulado obtida a partir do cálculo do DTW.

    Returns:
        Tuple[npt.ArrayLike, npt.ArrayLike]: coordenadas ponto a ponto referente a primeira e segunda sequência utilizada no cálculo do DTW.
    """
    rows, columns = np.shape(acc_cost_matrix)
    rows-=2
    columns-=2

    r = [rows]
    c = [columns]

    while rows != 0 or columns != 0:
        aux = {}
        if rows-1 >= 0: aux[acc_cost_matrix[rows -1][columns]] = (rows - 1, columns)
        if columns-1 >= 0: aux[acc_cost_matrix[rows][columns-1]] = (rows, columns - 1)
        if rows-1 >= 0 and columns-1 >= 0: aux[acc_cost_matrix[rows -1][columns-1]] = (rows -1, columns - 1)
        keys = list(aux.keys())
        key = min(keys)

        rows, columns = aux[key]

        r.insert(0, rows)
        c.insert(0, columns)
    
    return np.array(r), np.array(c)

def dump_hyperparameters(hyperparameters : Dict[str, Any], res_folder : str):
    bsd = None
    bse = None
    if 'signs_dev' in hyperparameters:
        bsd = hyperparameters['signs_dev']
        hyperparameters['signs_dev'] = None
    if 'signs_eva' in hyperparameters:
        bse = hyperparameters['signs_eva']
        hyperparameters['signs_eva'] = None
    with open(res_folder + os.sep + 'hyperparameters.json', 'w') as fw:
        json.dump(hyperparameters, fw)
    hyperparameters['signs_dev'] = bsd
    hyperparameters['signs_eva'] = bse
