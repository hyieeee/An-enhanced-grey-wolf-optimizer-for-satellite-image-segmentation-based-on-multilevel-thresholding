import jsonlines
import numpy
import matplotlib.pyplot as plt
import sys

'''
====================================|PNSR Metrics|=====================================
======================|    EGWO     |     GWO    |     PSO    |========================
Thresholds Level : {} |   {20.07}   |   {20.07}  |   {20.07}  |

'''


def LocalizeIdx(ls: list, algorithm: str):
    for l in ls:
        if l[0] == algorithm:
            return ls.index(l)


def MetricsAnalyze(fnpath, frpath):
    psnr_contrast = [["EGWO"], ["GWO"], ["PSO"]]  # for temporary record EGWO,GWO,PSO
    mse_contrast = [["EGWO"], ["GWO"], ["PSO"]]
    with jsonlines.open(fnpath, 'r') as jsons:
        for json in jsons:
            if list(json.keys())[1] == "psnr_max":
                psnr = psnr_contrast[LocalizeIdx(psnr_contrast, json["Algorithm"])]
                for i in range(5, 12):  # from level n=5 to 11
                    psnr.append(json["psnr_max"][str(i)])

            if list(json.keys())[1] == "mse_min":
                mse = mse_contrast[LocalizeIdx(mse_contrast, json["Algorithm"])]
                for i in range(5, 12):
                    mse.append(json["mse_min"][str(i)])
    return psnr_contrast, mse_contrast


fnpath = r"/Users/moka/Desktop/OstuMethod/ExpRunningResult/Kapur/algo_metrics_records.jsonl"
frpath = r"/Users/moka/Desktop/OstuMethod/ExpRunningResult/Kapur/metrics_comparative.txt"
psnr_contrast, mse_contrast = MetricsAnalyze(fnpath, frpath)
fw = open(frpath, 'a', encoding="utf-8")
sys.stdout = fw
sys.stdout.write("\r====================================|MSE  Metrics|=====================================")
sys.stdout.write("\r======================|    EGWO     |     GWO    |     PSO    |========================")
for i in range(5, 12):
    sys.stdout.write("\rThresholds Level : {} |   {:.2f}   |   {:.2f}  |   {:.2f}  |========================".
                     format(i, mse_contrast[0][i - 4], mse_contrast[1][i - 4], mse_contrast[2][i - 4]))
fw.close()
