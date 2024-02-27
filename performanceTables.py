import subprocess
import json
import argparse
from tabulate import tabulate

def main():
    subprocess.run(["rm -rf ./results_sim_*"], shell=True)

    wuShuComputeData = {}
    wuShuConstructTable = []
    wuShuConstructTable.append(["Cores","Dirac","ObjectID","Total Time","Total CPU Time","Time hM","Time hA Capacities","Time hA"])
    for dirac in range(1,5):
        print("Running simulation for Dirac",dirac)
        runResult = subprocess.run(["./build/sim_NSE/sim_5","0",str(dirac),"100","0","2","5"], check=True, capture_output=True,encoding="UTF-8")
        print("returncode",runResult.returncode)
        lines = runResult.stdout.splitlines()
        #print(lines)
        count = 1
        wuShuComputeCount = 1
        for line in lines:
            if line.find('--outputJSON') >= 0:
                splitString = line.split(';')
                tupleVals = splitString[1]
                #print(tupleVals)
                parsedJson = json.loads(str(tupleVals))
                wuShuConstructTable.append(["Default",str(dirac),str(count),parsedJson["time_total"],parsedJson["cpu_time_total"],parsedJson["time_loop_Hm"],parsedJson["time_loop_Ha_capacities"],parsedJson["time_loop_Ha"]])
                #data.append(["Default",str(dirac),str(count),tupleVals[0],tupleVals[1],tupleVals[2],tupleVals[3]])
                count+=1
            if line.find('--outputCalculationJSON') >=0:
                splitString = line.split(';')
                tupleVals = splitString[1]
                #print(tupleVals)
                parsedJson = json.loads(str(tupleVals))

                if parsedJson["object_id"] not in wuShuComputeData:
                    wuShuComputeData[parsedJson["object_id"]] = []
                wuShuComputeData[parsedJson["object_id"]].append({"dirac":str(dirac),"time_total":parsedJson["time_total"],"cpu_time_total":parsedJson["cpu_time_total"]})



    print(tabulate(wuShuConstructTable))
    for key,val in wuShuComputeData.items():
        wuShuComputeTable = []
        wuShuComputeTable.append(["Cores","Dirac","Iteration","Total Time","Total CPU Time"])
        print("table for obj",key)
        iterNumsPerDirac = {}
        for r in val:
            if r["dirac"] not in iterNumsPerDirac:
                iterNumsPerDirac[r["dirac"]]=0
            iterNumsPerDirac[r["dirac"]]+=1
            wuShuComputeTable.append(["Default",r["dirac"],iterNumsPerDirac[r["dirac"]],r["time_total"],r["cpu_time_total"]])
        print(tabulate(wuShuComputeTable))



if __name__ == "__main__":
    main()
