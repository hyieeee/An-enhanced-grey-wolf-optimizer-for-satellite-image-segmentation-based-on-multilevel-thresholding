import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PSOFindMax import PSO
from OtsuObjf import Otsu
from GWOFindMax import GWO
from EGWOFindMax import EGWO
from KapurEntropy import KapurEntropy
from TsallisEntropy import TsallisEntropy
import math
from ExperimentRecord.Record import ParaSettingRecord, ResultRecord

# 根目录下记录image segmentation结果以及metrics结果（目前包含PNSR、MSE）
root = r"/Users/moka/Desktop/ExpRunningResult/Tsallis/EGWO"
# curve_root = r"C:\Users\caoze\Downloads\ExpRunningResult\Kapur\GWO\Convergence_Curves"
curve_record = "convergence_record.txt"
metrics_record = "metrics.txt"
target_dir = r"/Users/moka/Desktop/ExpRunningResult/Tsallis_Entropy"


class Channel:
    def __init__(self, path, RGBmode=True):
        """
        定义处理的图片的模式:具体包括所要处理的图片、object function
        :param RGBmode:
        """
        if RGBmode == True:
            self.mode = 1  # RGB 3 channel mode

            self.tsallisg = TsallisEntropy(path, 'g')
            self.tsallisb = TsallisEntropy(path, 'b')
            self.tsallisr = TsallisEntropy(path, 'r')

            '''self.otsug = Otsu(path, 'g')
            self.otsub = Otsu(path, 'b')
            self.otsur = Otsu(path, 'r')'''

            '''self.kapurg = KapurEntropy(path, 'g')
            self.kapurb = KapurEntropy(path, 'b')
            self.kapurr = KapurEntropy(path, 'r')'''
        else:
            self.mode = 0
            self.tsallisr = TsallisEntropy(path, 'r')
            # self.otsur = Otsu(path, 'r')
            '''self.kapur = KapurEntropy(path, 'r')'''

    def thresholds(self, dim, lb, ub, SearchAgents_no, iters, method: EGWO):
        """
        根据imagemode返回各通道的阈值
        :param method:
        :param dim: 阈值个数n
        :return:RGB图像返回各通道的阈值
        """
        self.MAXiter = iters
        if self.mode == 0:
            # s = method.EGWO(self.otsur.objf, lb, ub, dim, Popsize, iters)
            s, para_setting = method.EGWO(self.tsallisr.objf, lb, ub, dim, SearchAgents_no, iters)
            print("The best combination of threshold is:" + str(s.bestIndividual))
            print("The between class variance is:" + str(s.best))
            return s.bestIndividual
        else:
            # RGB mode
            sg, para_setting = method.EGWO(self.tsallisg.objf, lb, ub, dim, SearchAgents_no, iters)
            sb, para_setting = method.EGWO(self.tsallisb.objf, lb, ub, dim, SearchAgents_no, iters)
            sr, para_setting = method.EGWO(self.tsallisr.objf, lb, ub, dim, SearchAgents_no, iters)
            threshold = np.array([sg.bestIndividual, sb.bestIndividual, sr.bestIndividual])
            channel_solution = np.array([sg, sb, sr])
            var = 0
            for s in channel_solution:
                var += (1 / s.best) - 1
                print("Between class variance of this single channel is" + str((1 / s.best) - 1))
            var = var / 3
            print("each channel's thresholds are as follows:\n")
            print(threshold)
            print("The between class variance is:" + str(var))
            return threshold, channel_solution, para_setting

    def CCurve(self, solution, epoch):
        """
        画收敛曲线，分通道独立
        :param solution:
        :return:
        """
        axis_x = np.array(range(1, self.MAXiter + 1))
        if not os.path.exists(root):
            os.makedirs(root)

        for s in solution:
            for i in range(len(s.convergence)):
                s.convergence[i] = (1 / s.convergence[i]) - 1

        with open(os.path.join(root, curve_record), 'a', encoding="utf-8") as f:
            f.write("\r====================Epoch {} of level n {}===========================".format(epoch,
                                                                                                     len(solution[
                                                                                                             0].bestIndividual)))
            f.write("\rGreen channel(Grey) convergence:\n {}".format(solution[0].convergence))
            f.write("\rBlue channel(Grey) convergence:\n {}".format(solution[1].convergence))
            f.write("\rRed channel(Grey) convergence:\n {}".format(solution[2].convergence))
        '''
        plt.figure()
        plt.plot(axis_x, solution[0].convergence, 'g', axis_x, solution[1].convergence, 'b', axis_x,
                 solution[2].convergence, 'r')
        curve_name = str(len(solution[0].bestIndividual)) + "_Threshold_Epoch_" + str(epoch) + "_ccurve.jpg"
        plt.savefig(os.path.join(curve_root, curve_name))'''

    '''
    for s in solution:
        ccurve = s.convergence
        plt.figure()
        plt.plot(axis_x, ccurve)'''
    # plt.show()


class ImageOut:
    def __init__(self, path, threshold, epoch):
        """
        :param path:
        :param threshold:
        :param epoch: 1~10(0~9)
        """
        self.epoch = epoch
        # only RGB mode
        self.refimage = cv2.imread(path)
        self.height, self.width = np.shape(self.refimage)[0], np.shape(self.refimage)[1]
        self.b, self.g, self.r = cv2.split(self.refimage)
        self.b_ori, self.g_ori, self.r_ori = cv2.split(self.refimage)
        self.gt = np.append(np.append(0, threshold[0]), 255)
        self.bt = np.append(np.append(0, threshold[1]), 255)
        self.rt = np.append(np.append(0, threshold[2]), 255)

    def imageout(self):
        """
        将图像按照阈值进行结构化处理
        :return:
        """
        # observe change
        # green channel
        for i in range(len(self.gt) - 1):
            at1 = self.g > self.gt[i]
            at2 = self.g < self.gt[i + 1]
            at = (at1 == at2)
            self.g[at] = self.gt[i]

        # blue channel
        for i in range(len(self.bt) - 1):
            at1 = self.b > self.bt[i]
            at2 = self.b < self.bt[i + 1]
            at = (at1 == at2)
            self.b[at] = self.bt[i]

        # red channel
        for i in range(len(self.rt) - 1):
            at1 = self.r > self.rt[i]
            at2 = self.r < self.rt[i + 1]
            at = (at1 == at2)
            self.r[at] = self.rt[i]

        # imageout=np.array([self.bt,self.gt,self.rt])
        '''
        self.refimage[:, :, 0] = self.b
        self.refimage[:, :, 1] = self.g
        self.refimage[:, :, 2] = self.r
        cv2.imshow("constructly changed image", self.refimage)
        cv2.waitKey(0)'''

        imgout = cv2.merge([self.b, self.g, self.r])
        t_level = len(self.gt) - 2  # denote n: threshold level
        if not os.path.exists(root):
            os.makedirs(root)
        fn = str(t_level) + "_threshold_" + str(self.epoch) + ".jpg"
        cv2.imwrite(os.path.join(root, fn), imgout)

        # display segmentation result
        # plt.figure()
        # plt.imshow(imgout)
        # plt.show()
        # plt.pause(1)

    def cal_PSNR(self, result_to_write):
        """
        计算PSNR和MSE，评价图片在结构化改变过后信息保留情况
        :return:MSE,PNSR
        """
        M = self.height
        N = self.width
        # RGB mode
        MSE_UP = 0
        MSE_DOWN = M * N

        # Green Channel
        for i in range(self.height):
            for j in range(self.width):
                MSE_UP += ((abs(self.g_ori[i, j] - self.g[i, j])) ** 2) / MSE_DOWN
        GMSE = MSE_UP
        print("Green Channel's MSE is:{:.3f}".format(float(GMSE)))

        # Blue Channel
        MSE_UP = 0
        for i in range(self.height):
            for j in range(self.width):
                MSE_UP += ((abs(self.b_ori[i, j] - self.b[i, j])) ** 2) / MSE_DOWN
        BMSE = MSE_UP
        print("Blue Channel's MSE is:{:.3f}".format(float(BMSE)))

        # Red Channel
        MSE_UP = 0
        for i in range(self.height):
            for j in range(self.width):
                MSE_UP += ((abs(self.r_ori[i, j] - self.r[i, j])) ** 2) / MSE_DOWN
        RMSE = MSE_UP
        print("Blue Channel's MSE is:{:.3f}".format(float(RMSE)))

        MSE = (GMSE + BMSE + RMSE) / 3
        print("MSE of original image (sat1 by EGWO,object function by Tsallis's Method) is {:.3f}".format(float(MSE)))

        # calculate PSNR
        PSNR = math.log(((255 * 255) / MSE), 10) * 10  # log以10为底
        print("PSNR of original image (sat1 by EGWO,object function by Tsallis's Method) is {:.3f}".format(float(PSNR)))

        result_to_write["PSNR"].append(str(PSNR))
        result_to_write["MSE"].append(str(MSE))
        return result_to_write

        r'''
        # record running results
        if not os.path.exists(root):
            os.makedirs(root)
        with open(os.path.join(root, metrics_record), 'a', encoding='utf-8') as f:
            f.write(
                "\r===========================Epoch {} Metrics as Follows=======================================".format(
                    self.epoch))
            f.write("\rPSNR:{:.2f}".format(float(PSNR)))
            f.write("\rMSE:{:.2f}".format(float(MSE)))'''


def main():
    c = Channel(
        r"/Users/moka/Desktop/A Grey Wolf Optimizer Based Automatic Clustering Algorithm for satellite image segmentation/unused material/Shanghai,_China.jpg")
    if not os.path.exists(root):
        os.makedirs(root)

    para_record = False

    result_to_write = {"实验编号": ["3"] * 4 * 10, "阈值个数": ["5"] * 10 + ["7"] * 10 + ["9"] * 10 + ["11"] * 10,
                       "轮数": [str(x) for x in range(1, 11)] * 4, "红色通道阈值": [],
                       "绿色通道阈值": [], "蓝色通道阈值": [], "PSNR": [], "MSE": [], "FSIM": [], "CPU Time": []}

    for level in range(4, 11, 2):  # set threshold level from 2~11-->multi-level thresholds of segmentation
        for epoch in range(10):  # choose best from 10 epoches
            print("\r==========Epoch {} of level n={} starts==========".format(epoch + 1, level + 1))
            t, s, para_setting \
                = c.thresholds((level + 1), 0, 255, 15, 150, EGWO())

            # Insert：record algorithm paras setting
            if para_record == False:
                ParaSettingRecord(target_dir, "EGWO", para_setting)
                para_record = True

            result_to_write["红色通道阈值"].append(t[2])
            result_to_write["绿色通道阈值"].append(t[0])
            result_to_write["蓝色通道阈值"].append(t[1])
            '''
            with open(os.path.join(root, metrics_record), 'a', encoding="utf-8") as f:
                f.write(
                    "\r==========================Epoch {} of level n={}==============================".format(epoch + 1,
                                                                                                              level + 1))
                f.write("\rThresholds of green layer {}".format(t[0]))
                f.write("\rThresholds of blue layer {}".format(t[1]))
                f.write("\rThresholds of red layer {}".format(t[2]))
                '''
            imageout = ImageOut(
                r"/Users/moka/Desktop/A Grey Wolf Optimizer Based Automatic Clustering Algorithm for satellite image segmentation/unused material/Shanghai,_China.jpg",
                t, (epoch + 1)
            )
            c.CCurve(s, epoch + 1)
            imageout.imageout()
            result_to_write = imageout.cal_PSNR(result_to_write)
            print("==========Epoch {} of level n={} has finished==========".format(epoch + 1, level + 1))
    ResultRecord(target_dir, "EGWO", result_to_write)


if __name__ == "__main__":
    main()
