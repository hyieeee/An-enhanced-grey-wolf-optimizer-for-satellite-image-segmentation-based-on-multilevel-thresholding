import random

import numpy as np
import math
from solution import solution
import time


class EGWO:  # Enhanced Grey Wolf Optimizer
    def __init__(self):
        # wa>wb>wd
        # sum(wa,wb,wd)=1
        self.wa = 0.6
        self.wb = 0.3
        self.wd = 0.1
        self.var = 0.01

    def EGWO(self, objf, lb, ub, dim, SearchAgents_no, Max_iter, add_random=True):
        """
        Enhanced Grey Wolf Optimizer
        :param objf:
        :param lb: list[lb]
        :param ub: list[ub]
        :param dim:
        :param SearchAgents_no: 15 to improve precisition
        :param Max_iters:
        :return:
        """

        Alpha_pos = np.zeros(dim)
        Alpha_score = float("inf")

        Beta_pos = np.zeros(dim)
        Beta_score = float("inf")

        Delta_pos = np.zeros(dim)
        Delta_score = float("inf")

        if not isinstance(lb, list):
            lb = [lb] * dim
        if not isinstance(ub, list):
            ub = [ub] * dim

        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            '''Positions[:, i] = (
                np.random.uniform(lb[i], ub[i], SearchAgents_no)
            )'''
            # discrete EGWO
            Positions[:, i] = (
                np.random.uniform(lb[i], ub[i], SearchAgents_no)
            )

        Convergence_curve = np.zeros(Max_iter)
        s = solution()  # record solution info

        # Loop counter
        print('Enhanced Grey Wolf Optimizer is optimizing  "' + objf.__name__ + '"')

        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        # Initialize position of lead wolves
        for l in range(0, Max_iter):
            Positions = np.sort(Positions, axis=1)  # ascending order
            for i in range(0, SearchAgents_no):

                # Return back the search agents that go beyond the boundaries of the search space
                # Different in Enhanced GWO
                '''for j in range(dim):
                    Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])'''

                # Calculate objective function for each search agent
                fitness = objf(Positions[i, :])

                # Update Alpha, Beta, and Delta
                if fitness < Alpha_score:
                    Delta_score = Beta_score  # Update delte
                    Delta_pos = Beta_pos.copy()
                    Beta_score = Alpha_score  # Update beta
                    Beta_pos = Alpha_pos.copy()
                    Alpha_score = fitness
                    # Update alpha
                    Alpha_pos = Positions[i, :].copy()

                if fitness > Alpha_score and fitness < Beta_score:
                    Delta_score = Beta_score  # Update delte
                    Delta_pos = Beta_pos.copy()
                    Beta_score = fitness  # Update beta
                    Beta_pos = Positions[i, :].copy()

                if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                    Delta_score = fitness  # Update delta
                    Delta_pos = Positions[i, :].copy()

            # Estimate the prey by lead wolves
            if add_random == True:
                var = self.var * (1 - ((1 * l) / Max_iter))
                ebsl = np.random.normal(0, var, dim)
                Prey_Position = self.wa * Alpha_pos + self.wb * Beta_pos + self.wd * Delta_pos + ebsl
            else:
                Prey_Position = self.wa * Alpha_pos + self.wb * Beta_pos + self.wd * Delta_pos

            # update wps positions
            for i in range(SearchAgents_no):
                for j in range(dim):
                    r = random.uniform(-2, 2)
                    pnew = Prey_Position[j] - r * abs(Prey_Position[j] - Positions[i, j])
                    # if exceed upper bound or lower bound, revise solution[j]
                    if pnew > ub[0]:  # lb:int
                        u = random.uniform(0, 1)
                        pnew = Positions[i, j] + u * (ub[0] - Positions[i, j])
                    elif pnew < lb[0]:
                        u = random.uniform(0, 1)
                        pnew = Positions[i, j] + u * (lb[0] - Positions[i, j])
                    Positions[i, j] = math.ceil(pnew)  # get int:discrete solutions

            Convergence_curve[l] = Alpha_score

            if l % 1 == 0:
                print(
                    ["At iteration " + str(l) + " the best fitness is " + str(Alpha_pos)]
                )

        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergence = Convergence_curve
        s.bestIndividual = Alpha_pos
        s.best = Alpha_score
        s.optimizer = "Enhanced GWO"
        s.objfname = objf.__name__

        return s


def objf(solution):
    """
    for test EGWO
    :param solution:
    :return:
    """
    s = 0
    for i in range(len(solution)):
        s += (solution[i] - 2) ** 2
    return s


def main():
    egwo = EGWO()
    egwo.EGWO(objf, -10, 10, 3, 20, 150, add_random=True)  # increase wps_no to improve presicition


'''if __name__ == "__main__":
    main()
'''