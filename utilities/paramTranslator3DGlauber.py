#!/usr/bin/env python3

from os import path
from glob import glob
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Usage: {} paramFolder".format(str(sys.argv[0])))
    exit(0)

folder = path.abspath(str(sys.argv[1]))
fileList = glob(path.join(folder, "parameter*"))

for ifile, filePath in enumerate(fileList):
    parDict = {}
    parFileName = filePath.split("/")[-1]
    with open(filePath, "r") as parfile:
        for line in parfile:
            line = line.split()
            parDict[str(line[0])] = float(line[1])
    transParFileName = "iEBE_{}".format(parFileName)
    transParDict = {}
    for key in parDict.keys():
        if "Slope" not in key and "shear" not in key:
            transParDict[key] = parDict[key]
    transParDict["ylossParam4At2"] = parDict["ylossParam4Slope1"]*2.
    transParDict["ylossParam4At4"] = (transParDict["ylossParam4At2"]
                                      + parDict["ylossParam4Slope2"]*2.)
    transParDict["ylossParam4At6"] = (transParDict["ylossParam4At4"]
                                      + parDict["ylossParam4Slope3"]*2.)
    transParDict["ylossParam4At10"] = (transParDict["ylossParam4At6"]
                                       + parDict["ylossParam4Slope3"]*4.)
    transParDict["Shear_to_S_ratio"] = parDict["shear_muB0"]
    transParDict["shear_muBf0p2"] = (parDict["shear_muB0p2"]
                                     /max(1e-8, parDict["shear_muB0"]))
    transParDict["shear_muBf0p4"] = (parDict["shear_muB0p4"]
                                     /max(1e-8, parDict["shear_muB0"]))
    with open(path.join(folder, transParFileName), "w") as parfile:
        for key in transParDict:
            parfile.write("{}  {}\n".format(key, transParDict[key]))
