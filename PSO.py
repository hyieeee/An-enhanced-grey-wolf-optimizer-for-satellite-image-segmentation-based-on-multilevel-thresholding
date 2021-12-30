import cv2
import numpy as np
import time
from solution import solution
import random
import math


class PSO:
    def __init__(self):
        pass

    def PSO(self, objf, lb, ub, dim, PopSize, iters):

        # PSO parameters

        Vmax = 5
        wMax = 0.8
        wMin = 0.2
        c1 = 1.2
        c2 = 0.8

        s = solution()
        if not isinstance(lb, list):
            lb = [lb] * dim
        if not isinstance(ub, list):
            ub = [ub] * dim

        ######################## Initializations

        vel = np.zeros((PopSize, dim))

        pBestScore = np.zeros(PopSize)
        pBestScore.fill(float("inf"))

        pBest = np.zeros((PopSize, dim))
        gBest = np.zeros(dim)

        gBestScore = float("inf")

        # initialize solution
        pos = np.zeros((PopSize, dim))
        for i in range(dim):
            '''
            pos[:, i] = np.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]'''
            pos[:, i] = np.random.uniform(lb[i], ub[i], PopSize)

        convergence_curve = np.zeros(iters)

        ############################################
        print('PSO is optimizing  "' + objf.__name__ + '"')

        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        for l in range(0, iters):
            for i in range(0, PopSize):
                # pos[i,:]=checkBounds(pos[i,:],lb,ub)
                for j in range(dim):
                    pos[i, j] = np.clip(pos[i, j], lb[j], ub[j])
                # Calculate objective function for each particle
                pos = np.sort(pos, axis=1)
                fitness = objf(pos[i, :])

                if pBestScore[i] > fitness:
                    pBestScore[i] = fitness
                    pBest[i, :] = pos[i, :].copy()

                if gBestScore > fitness:
                    gBestScore = fitness
                    gBest = pos[i, :].copy()

            # Update the W of PSO
            w = wMax - l * ((wMax - wMin) / iters)

            for i in range(0, PopSize):
                for j in range(0, dim):
                    r1 = random.random()
                    r2 = random.random()
                    vel[i, j] = (
                            w * vel[i, j]
                            + c1 * r1 * (pBest[i, j] - pos[i, j])
                            + c2 * r2 * (gBest[j] - pos[i, j])
                    )

                    if vel[i, j] > Vmax:
                        vel[i, j] = Vmax

                    if vel[i, j] < -Vmax:
                        vel[i, j] = -Vmax

                    pos[i, j] = math.ceil(pos[i, j] + vel[i, j])

            convergence_curve[l] = gBestScore

            if l % 1 == 0:
                print(
                    [
                        "At iteration "
                        + str(l + 1)
                        + " the best fitness is "
                        + str(gBestScore)
                    ]
                )
        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergence = convergence_curve
        s.optimizer = "PSO"
        s.objfname = objf.__name__

        # for i in range(len(gBest)):
        #    gBest[i]=math.ceil(gBest[i])
        s.bestIndividual = gBest
        s.best = gBestScore

        return s


r'''
def main():
    ostu = Otsu(
        r"C:\\Users\\caoze\\Downloads\\A Grey Wolf Optimizer Based Automatic Clustering Algorithm for satellite image segmentation\\unused material\\Shanghai,_China.jpg"
    ,'r')
    pso = PSO()
    s = pso.PSO(ostu.objf, 0, 255, 7, 180, 150)
    print("The best combination of threshold is:" + str(s.bestIndividual))
    print("The between class variance is:" + str(s.best))


if __name__ == "__main__":
    main()'''
