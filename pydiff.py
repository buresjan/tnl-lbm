import argparse
import os

def printDiff(str1,str2):
    print("<<<<")
    print(str1)
    print(">>>>")
    print(str2)
    print("-----")

def main():
    parser = argparse.ArgumentParser(
                    prog='Parallel IBM Pydiff',
                    description='''This script compares two .mtx files and returns whether the content of the matrices is within a certain margin of error.
                    The margin of error can be set using the -m or --maring parameter. By default this value is set to 1e-5. ''',
                    epilog='Bottom Text')
    parser.add_argument("file1",type=argparse.FileType("r"))
    parser.add_argument("file2",type=argparse.FileType("r"))
    parser.add_argument('-m','--margin',type=float, default=1e-5) #this parameter used to be -d digits but needed to be renamed because it was a float number
    args = parser.parse_args()
    n = args.margin
    filestr1 = args.file1.readlines()
    filestr2 = args.file2.readlines()
    filestr1.sort()
    filestr2.sort()
    if len(filestr1) == len(filestr2):
        #continue diff
        for i in range(len(filestr1)):
            line1 = filestr1[i]
            line2 = filestr2[i]
            if "%" in line1 and "%" in line2:
                continue
            elif "%" not in line1 and "%" not in line2:
                spl1 = line1.split()
                spl2 = line2.split()
                float1 = float(spl1[2])
                float2 = float(spl1[2])
                floatDiff = abs(float1-float2)
                #trunc1 = round(float(spl1[2]),n)
                #trunc2 = round(float(spl2[2]),n)
                #this is printing empty string
                #print(trunc1)
                #print(trunc2)
                if(spl1[0] != spl2[0]):
                    printDiff(line1,line2)
                    exit(1)
                if(spl1[1] != spl2[1]):
                    printDiff(line1,line2)
                    exit(1)
                elif(floatDiff > n):
                    printDiff(line1,line2)
                    print("difference: ",floatDiff)
                    #printDiff(trunc1,trunc2)
                    exit(1)
            else:
                # files not mathcing:
                print("files not matching")
                exit(1)
    else:
        print("files not matching!")
        exit(1)

if __name__ == "__main__":
    main()
