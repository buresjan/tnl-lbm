import subprocess
import json
import argparse
import os
from tabulate import tabulate

def run_sim(compute,diracmin=1,diracmax=4):
    wuShuComputeData = {}
    wuShuConstructTable = []
    variantString=""
    wuShuConstructTable.append(["Threads","Dirac","ObjectID","Total Time","Total CPU Time","Time hM Total","Time hM Capacities","time hM Construct","time hM Transpose","Time hA Capacities","Time hA","time_write","time matrix copy","time LL division"])
    for dirac in range(diracmin,diracmax+1):
        print("Running simulation for Dirac",dirac)
        runResult = subprocess.run(["./build/sim_NSE/sim_5","0",str(dirac),"100","0","5",str(compute)], check=True, capture_output=True,encoding="UTF-8")
        #runResult = subprocess.run(["./build/sim_NSE/sim_5","0",str(dirac),"100","0","2",str(compute)], check=True, capture_output=True,encoding="UTF-8")
        print(runResult.stdout)
        print("return code",runResult.returncode)
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
                wuShuConstructTable.append([parsedJson["threads"],str(dirac),str(count),parsedJson["time_total"],parsedJson["cpu_time_total"],parsedJson["time_loop_Hm"],parsedJson["time_loop_Hm_capacities"],parsedJson["time_loop_Hm_construct"],parsedJson["time_Hm_transpose"],parsedJson["time_loop_Ha_capacities"],parsedJson["time_loop_Ha"],parsedJson["time_write1"],parsedJson["time_matrixCopy"],parsedJson["time_LL_division"]])
                #data.append(["Default",str(dirac),str(count),tupleVals[0],tupleVals[1],tupleVals[2],tupleVals[3]])
                count+=1
                variantString = "hACapacities-"+str(parsedJson["variant_Ha_capacities"])+"_hA-"+str(parsedJson["variant_Ha"])
            if line.find('--outputCalculationJSON') >=0:
                splitString = line.split(';')
                tupleVals = splitString[1]
                #print(tupleVals)
                parsedJson = json.loads(str(tupleVals))

                if parsedJson["object_id"] not in wuShuComputeData:
                    wuShuComputeData[parsedJson["object_id"]] = []
                wuShuComputeData[parsedJson["object_id"]].append({"threads":parsedJson["threads"],"dirac":str(dirac),"time_total":parsedJson["time_total"],"cpu_time_total":parsedJson["cpu_time_total"]})
    return (wuShuConstructTable,wuShuComputeData,variantString)

def build(variantHaCapacities,variantHa):
    subprocess.run(["cmake -B build -DHA_CAPACITY_VARIANT="+str(variantHaCapacities)+" -DHA_VARIANT="+str(variantHa)], shell=True, check=True)
    subprocess.run(["cmake --build build"], shell=True, check=True)
def cleanFiles():
    subprocess.run(["rm -rf ./results_sim_*"], shell=True)

def main():
    #parse arguments
    variantString=""
    parser = argparse.ArgumentParser(
                    prog='Parallel LBM Performance Table Maker',
                    description='This program runs the simulations and displays relevant info about the simulation run',
                    epilog='Bottom Text')
    parser.add_argument('-o', '--output', default="prefix")
    parser.add_argument('-b', '--build', action='store_true')
    parser.add_argument('-t','--threads',type=int, default=0)
    parser.add_argument('--variantHaCapacities',type=int,default=1)
    parser.add_argument('--variantHa',type=int,default=1)
    parser.add_argument('--compute',choices=['gpu', 'cpu'],default='gpu')
    args = parser.parse_args()

    if args.threads > 0:
        os.environ['OMP_NUM_THREADS']=str(args.threads)

    filename = args.output +"_UNDEFINED_threads-"+str(args.threads)+"_"+variantString+".txt"
    compute = 5
    if args.compute == 'cpu':
        print("CPU selected")
        print("Running script for " + str(os.environ.get('OMP_NUM_THREADS', 'Default')) + " thread/s.")
        filename = args.output +"_CPU_threads-"+str(args.threads)+"_"+variantString+".txt"
        compute = 4
    elif args.compute == 'gpu':
        print("GPU selected")
        print("Running script for GPU")
        filename = args.output +"_GPU"+"_"+variantString+".txt"
        compute = 5
    else:
        print("Invalid device")
        return 1

    #run cmake
    if args.build :
       build(args.variantHaCapacities,args.variantHa)
    #clean files
    cleanFiles()
    #run simulation
    wuShuConstructTable, wuShuComputeData, variantString = run_sim(compute)

    file = open(filename,'w')
    #print(tabulate(wuShuConstructTable))
    file.write("Variants: "+variantString+'\n')
    file.write(tabulate(wuShuConstructTable)+'\n')
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
        file.write(tabulate(wuShuComputeTable)+'\n')
        #print(tabulate(wuShuComputeTable))

if __name__ == "__main__":
    main()
