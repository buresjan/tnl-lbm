#!/bin/bash

set -eu

# The script uses paths relative to the project directory, change there before
# doing anything else.
projectDir="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
cd "$projectDir"

if (( $# == 1 )); then
    compute="$1"
else
    compute=CPU
fi
discretization_ratio=0.5
resolution=1

for method in modified original; do
    for dirac in {1..4}; do
        echo "$method method, $compute compute, dirac $dirac"
        ./build/sim_NSE/sim_IBM3 --compute "$compute" --method "$method" --dirac "$dirac" --discretization-ratio "$discretization_ratio" --resolution "$resolution"
        for matrix in A M; do
            echo "Diff matrix $matrix Dirac $dirac"
            ./pydiff.py "ibm_${compute}_matrix-${matrix}_method-${method}_dirac-$dirac.mtx" "./tests/baseline_ibm_matrices/matrix-${matrix}_method-${method}_dirac-$dirac.mtx"
        done
    done
done
