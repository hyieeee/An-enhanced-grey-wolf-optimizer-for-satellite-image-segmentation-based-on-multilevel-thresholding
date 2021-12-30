import jsonlines
import os
import sys
import re


def FindMax(fnpath, fwpath):
    fw = jsonlines.open(fwpath, 'a')
    with jsonlines.open(fnpath, 'r') as jsons:
        for json in jsons:
            metrics = {"Algorithm": "", "psnr_max": dict((str(x), 0) for x in range(2, 15)),
                       "mse_min": dict((str(x), 0) for x in range(2, 15))}
            metrics["Algorithm"] = json["Algorithm"]
            if list(json.keys())[1] == "psnrs":
                del metrics["mse_min"]
                for key in json["psnrs"].keys():  # from 2 to 14, consistent with thresholds level

                    if json["psnrs"][key] == []:
                        pass
                    else:
                        metrics["psnr_max"][key] = max(json["psnrs"][key])
                        # find consistent convergence
                        # idx = json["psnrs"][key].index(max(json["psnrs"][key]))

            elif list(json.keys())[1] == "mses":
                del metrics["psnr_max"]
                for key in json["mses"].keys():
                    if json["mses"][key] == []:
                        pass
                    else:
                        metrics["mse_min"][key] = min(json["mses"][key])

            fw.write(metrics)

    fw.close()


fnpath = r"C:\Users\caoze\Downloads\ExpRunningResult\Kapur\metrics.jsonl"
fwpath = r"C:\Users\caoze\Downloads\ExpRunningResult\Kapur\algo_metrics_records.jsonl"
FindMax(fnpath, fwpath)
