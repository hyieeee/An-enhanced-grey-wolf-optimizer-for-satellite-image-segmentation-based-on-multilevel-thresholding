import cv2
import numpy as np
import math


def killzero(a):
    if a == 0:
        b = 1
        return b
    else:
        return a


class KapurEntropy:
    def __init__(self, path, channel):
        self.image = cv2.imread(path)
        self.b, self.g, self.r = cv2.split(self.image)
        self.height, self.width = np.shape(self.image)[0], np.shape(self.image)[1]
        self.pixnum = self.height * self.width  # 总像素数
        self.hist = np.zeros(256)
        if channel == 'b':
            for i in range(self.height):
                for j in range(self.width):
                    self.hist[self.b[i, j]] += 1  # 以处理blue层为例
            # print(self.hist)
        elif channel == 'g':
            for i in range(self.height):
                for j in range(self.width):
                    self.hist[self.g[i, j]] += 1
        elif channel == 'r':
            for i in range(self.height):
                for j in range(self.width):
                    self.hist[self.r[i, j]] += 1
        else:
            pass

    def objf(self, t: list):
        """
        objf:Kapur Entropy
        :param t: thresholds
        :return:
        """
        w = np.zeros(len(t) + 1)
        h = np.zeros(len(t) + 1)
        for i in range(256):
            if i >= 0 and i < t[0]:
                w[0] += self.hist[i] / self.pixnum
            elif i >= t[len(t) - 1] and i < 255:
                w[len(t)] += self.hist[i] / self.pixnum
            else:
                for j in range(len(t) - 1):
                    if i >= t[j] and i < t[j + 1]:
                        w[j + 1] += self.hist[i] / self.pixnum

        for i in range(256):
            if i >= 0 and i <= t[0]:  # grp1
                h[0] += -1 * ((self.hist[i] / self.pixnum) / killzero(w[0])) * math.log(
                    killzero((self.hist[i] / self.pixnum) / killzero(w[0])), 2)
            elif i >= t[len(t) - 1] and i <= 255:
                h[len(t)] += -1 * ((self.hist[i] / self.pixnum) / killzero(w[len(t)])) * math.log(killzero(
                    (self.hist[i] / self.pixnum) / killzero(w[len(t)])), 2)
            else:
                for j in range(len(t) - 1):
                    if i >= t[j] and i <= t[j + 1]:
                        h[j + 1] += -1 * ((self.hist[i] / self.pixnum) / killzero(w[j + 1])) * math.log(killzero(
                            ((self.hist[i] / self.pixnum) / killzero(w[j + 1]))), 2)
        H = np.sum(h)
        return (1 / (H + 1))
