#!/bin/bash

set -eu

# The script uses paths relative to the project directory, change there before
# doing anything else.
projectDir="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
cd "$projectDir"

Re=100
resolution=1
hi=1

for method in 0 1; do
    if [[ "$method" == 0 ]]; then
        echo "Modified method"
    else
        echo "Original method"
    fi
    for dirac in {1..4}; do
        ./build/sim_NSE/sim_IBM3 $method $dirac $Re $hi $resolution 0
        for matrix in A M; do
            echo "Diff matrix $matrix Dirac $dirac"
            ./pydiff.py ibm_GPU_matrix-${matrix}_method-${method}_dirac-$dirac.mtx ./tests/baseline_ibm_matrices/matrix-${matrix}_method-${method}_dirac-$dirac.mtx
        done
    done
done
