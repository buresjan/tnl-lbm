import subprocess
import json
import argparse
import os
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser(
                    prog='Parallel LBM Performance Table Maker',
                    description='This program runs the simulations and displays relevant info about the simulation run',
                    epilog='Bottom Text')
    parser.add_argument('-o', '--output', default="pyoutput.txt")
    parser.add_argument('-b', '--build', action='store_true')
    parser.add_argument('-t','--threads',type=int, default=0)

    args = parser.parse_args()

    if args.threads > 0:
        os.environ['OMP_NUM_THREADS']=str(args.threads)
        #subprocess.run(["export OMP_NUM_THREADS="+str(args.threads)], shell=True)
    #else:
        #os.environ['OMP_NUM_THREADS']=''
        #subprocess.run(["export OMP_NUM_THREADS"], shell=True)
    print("Running script for " + str(os.environ.get('OMP_NUM_THREADS', 'Default')) + " thread/s.")
    subprocess.run(["rm -rf ./results_sim_*"], shell=True)
    if args.build :
        subprocess.run(["cmake --build build"], shell=True, check=True)
    wuShuComputeData = {}
    wuShuConstructTable = []
    wuShuConstructTable.append(["Threads","Dirac","ObjectID","Total Time","Total CPU Time","Time hM","Time hM Capacities","time hM setElement","time hM Transpose","Time hA Capacities","Time hA","time_write","time matrix copy"])
    for dirac in range(1,5):
        print("Running simulation for Dirac",dirac)
        runResult = subprocess.run(["./build/sim_NSE/sim_5","0",str(dirac),"100","0","5","5"], check=True, capture_output=True,encoding="UTF-8")
        #runResult = subprocess.run(["./build/sim_NSE/sim_5","0",str(dirac),"100","0","2","5"], check=True, capture_output=True,encoding="UTF-8")
        print(runResult.stdout)
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
                wuShuConstructTable.append([parsedJson["threads"],str(dirac),str(count),parsedJson["time_total"],parsedJson["cpu_time_total"],parsedJson["time_loop_Hm"],parsedJson["time_Hm_capacities"],parsedJson["time_Hm_setElement"],parsedJson["time_Hm_transpose"],parsedJson["time_loop_Ha_capacities"],parsedJson["time_loop_Ha"],parsedJson["time_write1"],parsedJson["time_matrixCopy"]])
                #data.append(["Default",str(dirac),str(count),tupleVals[0],tupleVals[1],tupleVals[2],tupleVals[3]])
                count+=1
            if line.find('--outputCalculationJSON') >=0:
                splitString = line.split(';')
                tupleVals = splitString[1]
                #print(tupleVals)
                parsedJson = json.loads(str(tupleVals))

                if parsedJson["object_id"] not in wuShuComputeData:
                    wuShuComputeData[parsedJson["object_id"]] = []
                wuShuComputeData[parsedJson["object_id"]].append({"threads":parsedJson["threads"],"dirac":str(dirac),"time_total":parsedJson["time_total"],"cpu_time_total":parsedJson["cpu_time_total"]})


    file = open(args.output,'w')
    #print(tabulate(wuShuConstructTable))
    file.write(tabulate(wuShuConstructTable))
    for key,val in wuShuComputeData.items():
        wuShuComputeTable = []
        wuShuComputeTable.append(["Threads","Dirac","Iteration","Total Time","Total CPU Time"])
        print("table for obj",key)
        iterNumsPerDirac = {}
        for r in val:
            if r["dirac"] not in iterNumsPerDirac:
                iterNumsPerDirac[r["dirac"]]=0
            iterNumsPerDirac[r["dirac"]]+=1
            wuShuComputeTable.append([r["threads"],r["dirac"],iterNumsPerDirac[r["dirac"]],r["time_total"],r["cpu_time_total"]])
        file.write(tabulate(wuShuComputeTable))
        #print(tabulate(wuShuComputeTable))



if __name__ == "__main__":
    main()
