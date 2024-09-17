#! /usr/bin/env python3

import argparse
import sys


def stripComments(line):
    comment_token = "%"
    index = line.find(comment_token)
    return line[:index]


def printDiff(str1, str2, *, file=sys.stdout):
    print("<<<<", file=file)
    print(str1, file=file)
    print(">>>>", file=file)
    print(str2, file=file)
    print("-----", file=file)


def main():
    parser = argparse.ArgumentParser(
        prog="MTX File Pydiff",
        description="""
            This script compares two .mtx files and returns whether the content
            of the matrices is within a certain margin of error. The margin of
            error can be set using the -m or --margin parameter. By default this
            value is set to 1e-5.
            """,
    )
    parser.add_argument("file1", type=argparse.FileType("r"))
    parser.add_argument("file2", type=argparse.FileType("r"))
    parser.add_argument("-m", "--margin", type=float, default=1e-5)

    args = parser.parse_args()

    margin = args.margin
    lines1 = args.file1.readlines()
    lines2 = args.file2.readlines()
    lines1.sort()
    lines2.sort()

    if len(lines1) != len(lines2):
        print(
            f"Files have different numbers of lines: {len(lines1)} vs {len(lines2)}",
            file=sys.stderr,
        )
        sys.exit(1)

    for i in range(len(lines1)):
        line1 = stripComments(lines1[i])
        line2 = stripComments(lines2[i])
        spl1 = line1.split()
        spl2 = line2.split()
        if len(spl1) == len(spl2) == 3:
            float1 = float(spl1[2])
            float2 = float(spl1[2])
            floatDiff = abs(float1 - float2)
            if spl1[0] != spl2[0] or spl1[1] != spl2[1]:
                printDiff(line1, line2, file=sys.stderr)
                sys.exit(1)
            if floatDiff > margin:
                printDiff(line1, line2, file=sys.stderr)
                print(f"difference: {floatDiff}", file=sys.stderr)
                sys.exit(1)
        elif line1 != line2:
            printDiff(line1, line2, file=sys.stderr)
            print("files not matching", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
