import os
import sys
import re
import jsonlines


def ReadMetrics(fnpath, fwpath):
    psnrs = dict((x, []) for x in range(2, 15))
    mses = dict((x, []) for x in range(2, 15))
    relabel = re.compile(".*Epoch (\d+) of level n=(\d+)=.*")  # get thresholds level
    relb_pnsr = re.compile("PSNR:(.*)")
    relb_mse = re.compile("MSE:(.*)")
    fw = jsonlines.open(fwpath, 'a')
    with open(fnpath, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            match = re.match(relabel, lines[i])
            if match:
                match = eval(match.group(2))
                psnr = re.match(relb_pnsr, lines[i + 5].strip())
                psnr = eval(psnr.group(1))
                psnrs[match].append(psnr)
                mse = eval(re.match(relb_mse, lines[i + 6].strip()).group(1))
                mses[match].append(mse)
                i += 7
            else:
                i += 1
        psnrs_record={"Algorithm":"PSO","psnrs":psnrs}
        mses_record={"Algorithm":"PSO","mses":mses}
        fw.write(psnrs_record)
        fw.write(mses_record)
    fw.close()


fnpath = r"C:\Users\caoze\Downloads\ExpRunningResult\Kapur\PSO\metrics.txt"
fwpath = r"C:\Users\caoze\Downloads\ExpRunningResult\Kapur\metrics.jsonl"
ReadMetrics(fnpath, fwpath)
