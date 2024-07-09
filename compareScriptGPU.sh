#!/bin/bash
    set -e


    cmake -B build -S . -G Ninja
    ninja -C build

    rm -rf ./results_sim_*
    # matrix checking
    echo "Modified Matrices"
    ./build/sim_NSE/sim_5 0 1 100 0 2 5
    echo "Diff hA d1"
    python pydiff.py ws_tnl_hA_method-0_dirac-1.mtx original_matrices/ws_tnl_hA_original_method-0_dirac-1.mtx
    echo "Diff hM d1"
    python pydiff.py ws_tnl_hM_method-0_dirac-1.mtx original_matrices/ws_tnl_hM_original_method-0_dirac-1.mtx
    ./build/sim_NSE/sim_5 0 2 100 0 2 5
    echo "Diff hA d2"
    python pydiff.py ws_tnl_hM_method-0_dirac-2.mtx original_matrices/ws_tnl_hM_original_method-0_dirac-2.mtx
    echo "Diff hM d2"
    python pydiff.py ws_tnl_hA_method-0_dirac-2.mtx original_matrices/ws_tnl_hA_original_method-0_dirac-2.mtx
    ./build/sim_NSE/sim_5 0 3 100 0 2 5
    echo "Diff hA d3"
    python pydiff.py ws_tnl_hA_method-0_dirac-3.mtx original_matrices/ws_tnl_hA_original_method-0_dirac-3.mtx
    echo "Diff hM d3"
    python pydiff.py ws_tnl_hM_method-0_dirac-3.mtx original_matrices/ws_tnl_hM_original_method-0_dirac-3.mtx
    ./build/sim_NSE/sim_5 0 4 100 0 2 5
    echo "Diff hA d4"
    python pydiff.py ws_tnl_hA_method-0_dirac-4.mtx  original_matrices/ws_tnl_hA_original_method-0_dirac-4.mtx
    echo "Diff hM d4"
    python pydiff.py ws_tnl_hM_method-0_dirac-4.mtx original_matrices/ws_tnl_hM_original_method-0_dirac-4.mtx
    #old method
    # matrix checking
    echo "Original Matrices"
    ./build/sim_NSE/sim_5 1 1 100 0 2 5
    echo "Diff hA d1"
    python pydiff.py ws_tnl_hA_method-1_dirac-1.mtx original_matrices/ws_tnl_hA_original_method-1_dirac-1.mtx
    echo "Diff hM d1"
    python pydiff.py ws_tnl_hM_method-1_dirac-1.mtx original_matrices/ws_tnl_hM_original_method-1_dirac-1.mtx
    ./build/sim_NSE/sim_5 1 2 100 0 2 5
    echo "Diff hA d2"
    python pydiff.py ws_tnl_hA_method-1_dirac-2.mtx original_matrices/ws_tnl_hA_original_method-1_dirac-2.mtx
    echo "Diff hM d2"
    python pydiff.py ws_tnl_hM_method-1_dirac-2.mtx original_matrices/ws_tnl_hM_original_method-1_dirac-2.mtx
    ./build/sim_NSE/sim_5 1 3 100 0 2 5
    echo "Diff hA d3"
    python pydiff.py ws_tnl_hA_method-1_dirac-3.mtx original_matrices/ws_tnl_hA_original_method-1_dirac-3.mtx
    echo "Diff hM d3"
    python pydiff.py ws_tnl_hM_method-1_dirac-3.mtx original_matrices/ws_tnl_hM_original_method-1_dirac-3.mtx
    ./build/sim_NSE/sim_5 1 4 100 0 2 5
    echo "Diff hA d4"
    python pydiff.py ws_tnl_hA_method-1_dirac-4.mtx original_matrices/ws_tnl_hA_original_method-1_dirac-4.mtx
    echo "Diff hM d4"
    python pydiff.py ws_tnl_hM_method-1_dirac-4.mtx original_matrices/ws_tnl_hM_original_method-1_dirac-4.mtx
