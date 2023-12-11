import subprocess
from tabulate import tabulate

def main():
    subprocess.run(["rm", "-rf", "./results_sim_*"])
    data = []
    data.append(["Cores","Dirac","ObjectID","Total Time","Time hM","Time hA Capacities","Time hA"])
    for dirac in range(1,5):
        print("Running simulation for Dirac",dirac)
        runResult = subprocess.run(["./build/sim_NSE/sim_5","0",str(dirac),"100","0","2","5"], check=True, capture_output=True,encoding="UTF-8")
        print("returncode",runResult.returncode)
        lines = runResult.stdout.splitlines()
        #print(lines)
        count = 1
        for line in lines:
            if line.find('--timeTuple') >= 0:
                splitString = line.split(';')
                tupleVals = eval(splitString[1])
                print(tupleVals)
                data.append(["Default",str(dirac),str(count),tupleVals[0],tupleVals[1],tupleVals[2],tupleVals[3]])
                count+=1
    print(tabulate(data))


if __name__ == "__main__":
    main()
