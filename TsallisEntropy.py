import numpy as np
import math
import cv2


class TsallisEntropy:
    def __init__(self, path, channel, q_para=0.25):
        self.image = cv2.imread(path)
        self.b, self.g, self.r = cv2.split(self.image)
        self.height, self.width = np.shape(self.image)[0], np.shape(self.image)[1]
        self.pixnum = self.height * self.width  # 总像素数
        self.hist = np.zeros(256)  # from 0 t0 255

        self.q_para = q_para  # tsallis parameter

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
        # calculate Pi
        group_len=len(t)+1
        P = np.zeros(group_len)  # m(=len(t)+1) is the length of groups
        S = np.zeros(group_len)

        # group_seg: [0 thresholds 255]
        # length of group segmentation is that of S +1
        group_seg = [0]
        group_seg.extend(t)
        group_seg.extend([255])

        # calculate cap_P
        for i in range(256):
            if i >= group_seg[len(group_seg) - 2] and i <= group_seg[len(group_seg) - 1]:  # 255
                P[len(group_seg) - 2] += self.hist[i]
                break
            for j in range(len(group_seg)-2):
                if i >= group_seg[j] and i < group_seg[j + 1]:  # in other words, i <= group_seg[j+1]-1
                    P[j] += self.hist[i]


        # calculate S respectively
        for i in range(256):
            if i >= group_seg[len(group_seg) - 2] and i <= group_seg[len(group_seg) - 1]:
                if P[len(group_seg) - 2]:
                    S[len(group_seg) - 2] += math.pow((self.hist[i] / P[len(group_seg) - 2]), self.q_para)
            for j in range(len(group_seg) - 2):
                if i >= group_seg[j] and i < group_seg[j + 1]:
                    if P[j]:
                        temp = self.hist[i] / P[j]
                        S[j] += math.pow(temp, self.q_para)


        # calculate S
        S = -1 * S
        S = np.ones(len(S)) + S
        S = S / (np.ones(len(S)) * (self.q_para - 1))

        H = 0  # Sum of S respectively
        for s in S:
            H += s
        multi = 1
        for s in S:
            multi = s * multi

        H += (1 - self.q_para) * multi

        return H
