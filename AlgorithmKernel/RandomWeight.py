import random
import numpy as np


def dis2one(target):
    if abs(target - 1) < 1e-4:
        return True
    else:
        return False


def RandomWeight():
    while 1:
        wa = random.random()
        wb = random.random()
        wd = random.random()
        print("Respectivelu: {:.4f}, {:.4f}, {:.4f}".format(wa, wb, wd))
        if wa > wb and wb > wd and wa > wd and dis2one((wa+wb+wd)):
            return np.array([wa, wb, wd])


# RandomWeight()
