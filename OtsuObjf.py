import math
import numpy as np
import cv2

class Otsu:
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
        返回目标函数的计算值
        :param t: 可行解
        :return:
        """
        # N_obj = np.zeros(len(t) + 1)
        u = np.zeros(len(t) + 1)
        w = np.zeros(len(t) + 1)
        u_glb = 0
        objf = 0

        # group_seg: [0 thresholds 255]
        group_seg=[0]
        group_seg.extend(t)
        group_seg.extend([255])
        for i in range(256):
            for j in range(len(group_seg) - 1):
                # from 0 to tm (actually to N-1, N denotes number of grey level)
                if i >= group_seg[j] and i < group_seg[j + 1]:
                    # consistent with index of groups
                    w[j] += self.hist[i] / self.pixnum
                    u[j] += i * (self.hist[i] / self.pixnum)

            u_glb += i * (self.hist[i] / self.pixnum)

        for i in range(len(u)):
            if w[i] == 0:
                # then prob of each i in this group must be zero
                pass
            else:
                u[i] = u[i] / w[i]

        for i in range(len(w)):
            objf += w[i] * ((u[i] - u_glb) ** 2)

        return 1 / (objf + 1)
