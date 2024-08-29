#!/bin/bash

set -e


echo "Modified method"
./build/sim_NSE/sim_5 0 1 100 0 2 5
echo "Diff matrix A Dirac 1"
./pydiff.py ws_tnl_dA_method-0_dirac-1.mtx original_matrices/ws_tnl_hA_original_method-0_dirac-1.mtx
echo "Diff matrix M Dirac 1"
./pydiff.py ws_tnl_dM_method-0_dirac-1.mtx original_matrices/ws_tnl_hM_original_method-0_dirac-1.mtx
./build/sim_NSE/sim_5 0 2 100 0 2 5
echo "Diff matrix A Dirac 2"
./pydiff.py ws_tnl_dM_method-0_dirac-2.mtx original_matrices/ws_tnl_hM_original_method-0_dirac-2.mtx
echo "Diff matrix M Dirac 2"
./pydiff.py ws_tnl_dA_method-0_dirac-2.mtx original_matrices/ws_tnl_hA_original_method-0_dirac-2.mtx
./build/sim_NSE/sim_5 0 3 100 0 2 5
echo "Diff matrix A Dirac 3"
./pydiff.py ws_tnl_dA_method-0_dirac-3.mtx original_matrices/ws_tnl_hA_original_method-0_dirac-3.mtx
echo "Diff matrix M Dirac 3"
./pydiff.py ws_tnl_dM_method-0_dirac-3.mtx original_matrices/ws_tnl_hM_original_method-0_dirac-3.mtx
./build/sim_NSE/sim_5 0 4 100 0 2 5
echo "Diff matrix A Dirac 4"
./pydiff.py ws_tnl_dA_method-0_dirac-4.mtx original_matrices/ws_tnl_hA_original_method-0_dirac-4.mtx
echo "Diff matrix M Dirac 4"
./pydiff.py ws_tnl_dM_method-0_dirac-4.mtx original_matrices/ws_tnl_hM_original_method-0_dirac-4.mtx


echo "Original method"
./build/sim_NSE/sim_5 1 1 100 0 2 5
echo "Diff matrix A Dirac 1"
./pydiff.py ws_tnl_dA_method-1_dirac-1.mtx original_matrices/ws_tnl_hA_original_method-1_dirac-1.mtx
echo "Diff matrix M Dirac 1"
./pydiff.py ws_tnl_dM_method-1_dirac-1.mtx original_matrices/ws_tnl_hM_original_method-1_dirac-1.mtx
./build/sim_NSE/sim_5 1 2 100 0 2 5
echo "Diff matrix A Dirac 2"
./pydiff.py ws_tnl_dA_method-1_dirac-2.mtx original_matrices/ws_tnl_hA_original_method-1_dirac-2.mtx
echo "Diff matrix M Dirac 2"
./pydiff.py ws_tnl_dM_method-1_dirac-2.mtx original_matrices/ws_tnl_hM_original_method-1_dirac-2.mtx
./build/sim_NSE/sim_5 1 3 100 0 2 5
echo "Diff matrix A Dirac 3"
./pydiff.py ws_tnl_dA_method-1_dirac-3.mtx original_matrices/ws_tnl_hA_original_method-1_dirac-3.mtx
echo "Diff matrix M Dirac 3"
./pydiff.py ws_tnl_dM_method-1_dirac-3.mtx original_matrices/ws_tnl_hM_original_method-1_dirac-3.mtx
./build/sim_NSE/sim_5 1 4 100 0 2 5
echo "Diff matrix A Dirac 4"
./pydiff.py ws_tnl_dA_method-1_dirac-4.mtx original_matrices/ws_tnl_hA_original_method-1_dirac-4.mtx
echo "Diff matrix M Dirac 4"
./pydiff.py ws_tnl_dM_method-1_dirac-4.mtx original_matrices/ws_tnl_hM_original_method-1_dirac-4.mtx
