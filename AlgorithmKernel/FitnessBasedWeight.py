import random
import numpy as np
from typing import Union

"""
Dynamically generate random by fitness function value of head wolves
"""


def WeightfromMini(fitness: Union[list, np.ndarray]):
    """
    when fitness value is minimized
    :param fitness:
    :return:
    """
    fitsum = sum(fitness)
    wa = 0.5 * (1 - (fitness[0] / fitsum)) # alpha score
    wb = 0.5 * (1 - (fitness[1] / fitsum)) # beta score
    wd = 0.5 * (1 - (fitness[2] / fitsum)) # delta score

    return wa, wb, wd


def WeightfromMax(fitness: Union[list, np.ndarray]):
    """
    when fitness value is maxmized
    :param fitness:
    :return:
    """
    fitsum = sum(fitness)
    wa = fitness[0] / fitsum
    wb = fitness[1] / fitsum
    wd = fitness[2] / fitsum

    return wa, wb, wd
