#pragma once

#include <TNL/Timer.h>
#include <TNL/Backend/KernelLaunch.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Host.h>
#include <TNL/DiscreteMath.h>
#include <TNL/Matrices/MatrixWriter.h>

#include <cstddef>
#include <string>
#include <omp.h>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "lbm_common/timeutils.h"
#include "lagrange_3D.h"
#include "ibm_kernels.h"
#include "dirac.h"

template< typename LBM >
auto Lagrange3D<LBM>::hmacroVector(int macro_idx) -> hVectorView
{
	if (macro_idx >= MACRO::N)
		throw std::logic_error("macro_idx must be less than MACRO::N");

	auto& block = lbm.blocks.front();

	// local size of the distributed array
	// FIXME: overlaps
	const idx n = block.local.x() * block.local.y() * block.local.z();

	// offset for the requested quantity
	// FIXME: overlaps
	const idx quantity_offset = block.hmacro.getStorageIndex(macro_idx, block.offset.x(), block.offset.y(), block.offset.z());

	// pointer to the data
	dreal* data = block.hmacro.getData() + quantity_offset;

	TNL_ASSERT_EQ(data, &block.hmacro(macro_idx, block.offset.x(), block.offset.y(), block.offset.z()), "indexing bug");

	return {data, n};
}

template< typename LBM >
auto Lagrange3D<LBM>::dmacroVector(int macro_idx) -> dVectorView
{
	if (macro_idx >= MACRO::N)
		throw std::logic_error("macro_idx must be less than MACRO::N");

	auto& block = lbm.blocks.front();

	// local size of the distributed array
	// FIXME: overlaps
	const idx n = block.local.x() * block.local.y() * block.local.z();

	// offset for the requested quantity
	// FIXME: overlaps
	const idx quantity_offset = block.dmacro.getStorageIndex(macro_idx, block.offset.x(), block.offset.y(), block.offset.z());

	// pointer to the data
	dreal* data = block.dmacro.getData() + quantity_offset;

	return {data, n};
}

template< typename LBM >
typename LBM::TRAITS::real Lagrange3D<LBM>::computeMinDist()
{
	minDist=1e10;
	maxDist=-1e10;
	for (std::size_t i=0;i<LL.size()-1;i++)
	{
	for (std::size_t j=i+1;j<LL.size();j++)
	{
		real d = dist(LL[j],LL[i]);
		if (d>maxDist) maxDist=d;
		if (d<minDist) minDist=d;
	}
	if (i%1000==0)
	fmt::print("computeMinDist {} of {}\n", i, LL.size());
	}
	return minDist;
}

template< typename LBM >
typename LBM::TRAITS::real Lagrange3D<LBM>::computeMaxDistFromMinDist(typename LBM::TRAITS::real mindist)
{
	maxDist=-1e10;
	for (std::size_t i=0;i<LL.size()-1;i++)
	{
		real search_dist = 2.0*mindist;
		for (std::size_t j=i+1;j<LL.size();j++)
		{
			real d = dist(LL[j],LL[i]);
			if (d < search_dist)
				if (d>maxDist) maxDist=d;
		}
	}
	return maxDist;
}


template< typename LBM >
void Lagrange3D<LBM>::computeMaxMinDist()
{
	maxDist=-1e10;
	minDist=1e10;
	if (lag_X<=0 || lag_Y<=0) return;
	for (int i=0;i<lag_X-1;i++)
	for (int j=0;j<lag_Y-1;j++)
	{
		int index = findIndex(i,j);
		for (int i1=0;i1<=1;i1++)
		for (int j1=0;j1<=1;j1++)
		if (j1!=0 || i1!=0)
		{
			int index1 = findIndex(i+i1,j+j1);
//			int index1 = findIndex(i+i1,j);
			real d = dist(LL[index1],LL[index]);
			if (d>maxDist) maxDist=d;
			if (d<minDist) minDist=d;
		}
	}
}


template< typename LBM >
int Lagrange3D<LBM>::createIndexArray()
{
	if (lag_X<=0 || lag_Y<=0) return 0;
	index_array = new int*[lag_X];
	for (int i=0;i<lag_X;i++) index_array[i] = new int[lag_Y];
	for (std::size_t k=0;k<LL.size();k++) index_array[LL[k].lag_x][LL[k].lag_y] = k;
	indexed=true;
	return 1;
}


template< typename LBM >
int Lagrange3D<LBM>::findIndex(int i, int j)
{
	if (!indexed) createIndexArray();
	if (!indexed) return 0;
	return index_array[i][j];
	// brute force
//	for (std::size_t k=0;k<LL.size();k++) if (LL[k].lag_x == i && LL[k].lag_y == j) return k;
//	fmt::print("findIndex({},{}): not found\n",i,j);
//	return 0;
}

template< typename LBM >
int Lagrange3D<LBM>::findIndexOfNearestX(typename LBM::TRAITS::real x)
{
	int imin=0;
	real xmindist=fabs(LL[imin].x-x);
	// brute force
	for (std::size_t k=1;k<LL.size();k++)
	if (fabs(LL[k].x - x) < xmindist)
	{
		imin=k;
		xmindist = fabs(LL[imin].x-x);
	}
	return imin;
}


template< typename LBM >
void Lagrange3D<LBM>::convertLagrangianPoints()
{
	using HLPVECTOR_REAL = TNL::Containers::Vector<LagrangePoint3D<real>,TNL::Devices::Host>;
	HLPVECTOR_REAL tempLL;
	tempLL = LL;

	hLL_lat.setSize(LL.size());

	TNL::Algorithms::parallelFor< TNL::Devices::Host >(
		(idx) 0,
		(idx) LL.size(),
		[] (
			idx i,
			typename HLPVECTOR_DREAL::ViewType hLL_lat,
			typename HLPVECTOR_REAL::ConstViewType tempLL,
			typename LBM::lat_t lat
		) mutable
		{
			hLL_lat[i] = lat.phys2lbmPoint({tempLL[i].x, tempLL[i].y, tempLL[i].z});
		},
		hLL_lat.getView(),
		tempLL.getConstView(),
		lbm.lat
	);
}

template< typename LBM >
void Lagrange3D<LBM>::allocateMatricesCPU()
{
	idx m = LL.size();	// number of lagrangian nodes
	idx n = lbm.lat.global.x()*lbm.lat.global.y()*lbm.lat.global.z();	// number of eulerian nodes

	spdlog::info("number of Lagrangian points: {}", m);

	// Convert Lagrangian points to lattice coordinates and to StaticVector with dreal
	convertLagrangianPoints();

	// Allocate matrices
	ws_tnl_hM.setDimensions(m, n);
	ws_tnl_hA = std::make_shared< hEllpack >();
	ws_tnl_hA->setDimensions(m, m);

	// Allocate row capacity vectors
	hM_row_capacities.setSize(m);
	hA_row_capacities.setSize(m);

	// Create vectors for the solution of the linear system
	for (int k=0;k<3;k++)
	{
		ws_tnl_hx[k].setSize(m);
		ws_tnl_hb[k].setSize(m);
		#ifdef USE_CUDA
		ws_tnl_dx[k].setSize(m);
		ws_tnl_db[k].setSize(m);
		ws_tnl_hxz[k].setSize(m);
		ws_tnl_hbz[k].setSize(m);
		#endif
	}

	// Zero-initialize x1, x2, x3
	for (int k=0;k<3;k++) ws_tnl_hx[k].setValue(0);
	#ifdef USE_CUDA
	for (int k=0;k<3;k++) ws_tnl_dx[k].setValue(0);
	for (int k=0;k<3;k++) ws_tnl_hxz[k].setValue(0);
	#endif
}

template< typename LBM >
void Lagrange3D<LBM>::constructMatricesCPU()
{
	// Start timer to measure WuShu construction time
	TNL::Timer timer;
	TNL::Timer loopTimer;

	double time_loop_Hm = 0;
	double time_loop_Hm_Capacities = 0;
	double time_loop_Hm_Construct = 0;
	double time_loop_Ha_Capacities = 0;
	double time_loop_Ha = 0;
	double time_write1 = 0;
	double time_matrixCopy = 0;
	double time_total = 0;
	double cpu_time_total = 0;
	double time_Hm_transpose = 0;

	fmt::print("started timer for wushu construction\n");
	timer.start();

	idx m = LL.size();	// number of lagrangian nodes

	loopTimer.start();
	// Calculate row capacities of hM
	const idx support = 5; // search in this support
	#pragma omp parallel for schedule(static)
	for (idx i = 0; i < m; i++)
	{
		idx rowCapacity = 0;

		idx fi_x = floor(hLL_lat[i].x() - (dreal)0.5);
		idx fi_y = floor(hLL_lat[i].y() - (dreal)0.5);
		idx fi_z = floor(hLL_lat[i].z() - (dreal)0.5);

		// FIXME: iterate over LBM blocks
		for (idx gz = MAX(0, fi_z - support); gz < MIN(lbm.blocks.front().local.z(), fi_z + support); gz++)
		for (idx gy = MAX(0, fi_y - support); gy < MIN(lbm.blocks.front().local.y(), fi_y + support); gy++)
		for (idx gx = MAX(0, fi_x - support); gx < MIN(lbm.blocks.front().local.x(), fi_x + support); gx++)
		{
			if (
				isDDNonZero(diracDeltaTypeEL, gx - hLL_lat[i].x()) &&
				isDDNonZero(diracDeltaTypeEL, gy - hLL_lat[i].y()) &&
				isDDNonZero(diracDeltaTypeEL, gz - hLL_lat[i].z())
			)
			{
				rowCapacity++;
			}
		}

		hM_row_capacities[i] = rowCapacity;
	}

	loopTimer.stop();
	time_loop_Hm_Capacities = loopTimer.getRealTime();
	fmt::print("------- matrix M row capacities time: {}\n", time_loop_Hm_Capacities);

	ws_tnl_hM.setRowCapacities(hM_row_capacities);

	loopTimer.reset();
	loopTimer.start();
	// Construct the matrix hM
	#pragma omp parallel for schedule(static)
	for (idx i = 0; i < m; i++)
	{
		idx fi_x = floor(hLL_lat[i].x() - (dreal)0.5);
		idx fi_y = floor(hLL_lat[i].y() - (dreal)0.5);
		idx fi_z = floor(hLL_lat[i].z() - (dreal)0.5);

		// FIXME: iterate over LBM blocks
		for (idx gz = MAX(0, fi_z - support); gz < MIN(lbm.blocks.front().local.z(), fi_z + support); gz++)
		for (idx gy = MAX(0, fi_y - support); gy < MIN(lbm.blocks.front().local.y(), fi_y + support); gy++)
		for (idx gx = MAX(0, fi_x - support); gx < MIN(lbm.blocks.front().local.x(), fi_x + support); gx++)
		{
			using real = typename Lagrange3D<LBM>::HLPVECTOR_DREAL::RealType::RealType;
			if (
				isDDNonZero(diracDeltaTypeEL, gx - hLL_lat[i].x()) &&
				isDDNonZero(diracDeltaTypeEL, gy - hLL_lat[i].y()) &&
				isDDNonZero(diracDeltaTypeEL, gz - hLL_lat[i].z())
			)
			{
				real dd =
					diracDelta(diracDeltaTypeEL, gx - hLL_lat[i].x()) *
					diracDelta(diracDeltaTypeEL, gy - hLL_lat[i].y()) *
					diracDelta(diracDeltaTypeEL, gz - hLL_lat[i].z());
				ws_tnl_hM.setElement(i,lbm.blocks.front().hmap.getStorageIndex(gx,gy,gz),dd);
			}
		}
	}
	loopTimer.stop();
	time_loop_Hm_Construct = loopTimer.getRealTime();
	time_loop_Hm = time_loop_Hm_Capacities + time_loop_Hm_Construct;
	fmt::print("------- matrix M construction time: {}\n", time_loop_Hm_Construct);

	// Transpose matrix M
	loopTimer.reset();
	loopTimer.start();
    ws_tnl_hMT.getTransposition(ws_tnl_hM);
	loopTimer.stop();
	time_Hm_transpose = loopTimer.getRealTime();

	int threads = omp_get_max_threads();
	// TODO: find the correct threshold for this condition
	//if( m < 1000 )
	//	threads = 1;
	(void) threads; // shut up clang warning

	loopTimer.reset();
	loopTimer.start();
	// Calculate row capacities of matrix A
	#pragma omp parallel for schedule(dynamic) num_threads(threads)
	for (idx index_row = 0; index_row < m; index_row++)
	{
		idx rowCapacity = 0;
		for (idx index_col = 0; index_col < m; index_col++)
		{
			if (methodVariant==DiracMethod::MODIFIED)
			{
				if(is3DiracNonZero(diracDeltaTypeLL, index_col, index_row, hLL_lat))
				{
					rowCapacity++;
				}
			}
			else
			{
				real val=0;
				auto row1 = ws_tnl_hM.getRow(index_row);
				auto row2 = ws_tnl_hM.getRow(index_col);
				for (idx in1=0; in1 < row1.getSize(); in1++)
				{
					for (idx in2=0; in2 < row2.getSize(); in2++)
					{
						if (row1.getColumnIndex(in1) == row2.getColumnIndex(in2))
						{
							val += row1.getValue(in1) * row2.getValue(in2);
							break;
						}
					}
				}
				if (val > 0)
					rowCapacity++;
			}
		}
		hA_row_capacities[index_row] = rowCapacity;
	}


	loopTimer.stop();
	time_loop_Ha_Capacities = loopTimer.getRealTime();
	fmt::print("------- matrix A row capacities time: {}\n", time_loop_Ha_Capacities);
	ws_tnl_hA->setRowCapacities(hA_row_capacities);


	loopTimer.reset();
	loopTimer.start();
	// Construct the matrix A
	#pragma omp parallel for schedule(dynamic) num_threads(threads)
	for (idx index_row=0;index_row<m;index_row++)
	{
		for (idx index_col=0;index_col<m;index_col++)
		{
			if (methodVariant==DiracMethod::MODIFIED)
			{
				if(is3DiracNonZero(diracDeltaTypeLL, index_col, index_row, hLL_lat))
				{
					real ddd = calculate3Dirac(diracDeltaTypeLL, index_col, index_row, hLL_lat);
					ws_tnl_hA->setElement(index_row, index_col, ddd);
				}
			} else
			{
				real val=0;
				auto row1 = ws_tnl_hM.getRow(index_row);
				auto row2 = ws_tnl_hM.getRow(index_col);
				for (idx in1=0; in1 < row1.getSize(); in1++)
				{
					for (idx in2=0; in2 < row2.getSize(); in2++)
					{
						if (row1.getColumnIndex(in1) == row2.getColumnIndex(in2))
						{
							val += row1.getValue(in1) * row2.getValue(in2);
							break;
						}
					}
				}
				if (val > 0) {
					ws_tnl_hA->setElement(index_row, index_col, val);
				}
			}
		}
	}
	loopTimer.stop();
	time_loop_Ha = loopTimer.getRealTime();
	fmt::print("------- matrix A construction time: {}\n", time_loop_Ha);


	// Update the preconditioner
	ws_tnl_hprecond->update(ws_tnl_hA);
	ws_tnl_hsolver.setMatrix(ws_tnl_hA);

	#ifdef USE_CUDA
	// Copy matrices from host to the GPU
	loopTimer.reset();
	loopTimer.start();
	ws_tnl_dA = std::make_shared< dEllpack >();
	*ws_tnl_dA = *ws_tnl_hA;
	ws_tnl_dM = ws_tnl_hM;
	ws_tnl_dMT = ws_tnl_hMT;
	loopTimer.stop();
	time_matrixCopy = loopTimer.getRealTime();

	// Update the preconditioner
	ws_tnl_dprecond->update(ws_tnl_dA);
	ws_tnl_dsolver.setMatrix(ws_tnl_dA);

	loopTimer.reset();
	loopTimer.start();
	TNL::Matrices::MatrixWriter< hEllpack >::writeMtx( "ws_tnl_hM_method-"+std::to_string((int)methodVariant)+"_dirac-"+std::to_string(diracDeltaTypeEL)+".mtx", ws_tnl_hM );
	TNL::Matrices::MatrixWriter< hEllpack >::writeMtx( "ws_tnl_hA_method-"+std::to_string((int)methodVariant)+"_dirac-"+std::to_string(diracDeltaTypeEL)+".mtx", *ws_tnl_hA );
	loopTimer.stop();
	time_write1 = loopTimer.getRealTime();
	#endif

	timer.stop();
	fmt::print("-------wuShuTnl final timer time: {}\n",timer.getRealTime());
	time_total = timer.getRealTime();
	cpu_time_total = timer.getCPUTime();
	nlohmann::json j;
	j["threads"] = omp_get_max_threads();
	j["time_total"] = time_total;
	j["cpu_time_total"] = cpu_time_total;
	j["time_loop_Hm"] = time_loop_Hm;
	j["time_loop_Ha"] = time_loop_Ha;
	j["time_loop_Ha_capacities"] = time_loop_Ha_Capacities;
	j["time_loop_Hm_capacities"] = time_loop_Hm_Capacities;
	j["time_loop_Hm_construct"] = time_loop_Hm_Construct;
	j["time_Hm_transpose"] = time_Hm_transpose;
	j["time_write1"] = time_write1;
	j["time_matrixCopy"] = time_matrixCopy;
	std::string jsonOutput = j.dump();
	fmt::print("--outputJSON;{}\n",jsonOutput);
}

template< typename LBM >
void Lagrange3D<LBM>::allocateMatricesGPU()
{
	idx m = LL.size();	// number of lagrangian nodes
	idx n = lbm.lat.global.x()*lbm.lat.global.y()*lbm.lat.global.z();	// number of eulerian nodes

	spdlog::info("number of Lagrangian points: {}", m);

	// Convert Lagrangian points to lattice coordinates and to StaticVector with dreal
	convertLagrangianPoints();
	dLL_lat = hLL_lat;

	// Allocate matrices
	ws_tnl_dM.setDimensions(m, n);
	ws_tnl_dA = std::make_shared< dEllpack >();
	ws_tnl_dA->setDimensions(m, m);

	// Allocate row capacity vectors
	dM_row_capacities.setSize(m);
	dA_row_capacities.setSize(m);

	// Create vectors for the solution of the linear system
	for (int k=0;k<3;k++)
	{
		#ifdef USE_CUDA
		ws_tnl_dx[k].setSize(m);
		ws_tnl_db[k].setSize(m);
		#endif
	}

	// Zero-initialize x1, x2, x3
	#ifdef USE_CUDA
	for (int k=0;k<3;k++) ws_tnl_dx[k].setValue(0);
	#endif
}

template< typename LBM >
void Lagrange3D<LBM>::constructMatricesGPU()
{
	// Start timer to measure WuShu construction time
	TNL::Timer timer;
	TNL::Timer loopTimer;

	double time_loop_Hm = 0;
	double time_loop_Hm_Capacities = 0;
	double time_loop_Hm_Construct = 0;
	double time_loop_Ha_Capacities = 0;
	double time_loop_Ha = 0;
	double time_write1 = 0;
	double time_matrixCopy = 0;
	double time_total = 0;
	double cpu_time_total = 0;
	double time_Hm_transpose = 0;

	fmt::print("started timer for wushu construction\n");
	timer.start();

	idx m = LL.size();	// number of lagrangian nodes


	loopTimer.reset();
	loopTimer.start();
	// Calculate row capacities of matrix M
	TNL::Backend::LaunchConfiguration dM_config;
	dM_config.blockSize.x = 256;
	dM_config.gridSize.x = TNL::roundUpDivision(m,dM_config.blockSize.x);
	TNL::Backend::launchKernelSync( dM_row_capacities_kernel<LBM>, dM_config,
		dLL_lat.getView(),
		dM_row_capacities.getView(),
		lbm.blocks.front().local,
		diracDeltaTypeEL);

	loopTimer.stop();
	time_loop_Hm_Capacities = loopTimer.getRealTime();
	fmt::print("------- matrix M row capacities time: {}\n", time_loop_Hm_Capacities);

	ws_tnl_dM.setRowCapacities(dM_row_capacities);

	loopTimer.reset();
	loopTimer.start();
	// Construct the matrix M
	TNL::Backend::LaunchConfiguration dM_config_build;
	dM_config_build.blockSize.x = 256;
	dM_config_build.gridSize.x = TNL::roundUpDivision(m,dM_config_build.blockSize.x);

	TNL::Backend::launchKernelSync( dM_construction_kernel<LBM>, dM_config_build,
		dLL_lat.getConstView(),
		ws_tnl_dM.getView(),
		lbm.blocks.front().local,
		#ifdef HAVE_MPI
		lbm.blocks.front().dmap.getConstLocalView(),
		#else
		lbm.blocks.front().dmap.getConstView(),
		#endif
		diracDeltaTypeEL);

	loopTimer.stop();
	time_loop_Hm_Construct = loopTimer.getRealTime();
	time_loop_Hm = time_loop_Hm_Capacities + time_loop_Hm_Construct;
	fmt::print("------- matrix M construction time: {}\n", time_loop_Hm_Construct);

	// Transpose matrix M
	loopTimer.reset();
	loopTimer.start();
    ws_tnl_dMT.getTransposition(ws_tnl_dM);
	loopTimer.stop();
	time_Hm_transpose = loopTimer.getRealTime();

	loopTimer.reset();
	loopTimer.start();
	// Calculate row capacities of matrix A
	TNL::Backend::LaunchConfiguration dA_config;
	dA_config.blockSize.x = 256;
	dA_config.gridSize.x = TNL::roundUpDivision(m, dA_config.blockSize.x);
	TNL::Backend::launchKernelSync(dA_row_capacities_kernel<LBM>, dA_config,
		dLL_lat.getConstView(),
		dA_row_capacities.getView(),
		ws_tnl_dM.getConstView(),
		diracDeltaTypeLL,
		methodVariant
		);

	loopTimer.stop();
	time_loop_Ha_Capacities = loopTimer.getRealTime();
	fmt::print("------- matrix A row capacities time: {}\n", time_loop_Ha_Capacities);
	ws_tnl_dA->setRowCapacities(dA_row_capacities);


	loopTimer.reset();
	loopTimer.start();
	// Construct the matrix A
	TNL::Backend::LaunchConfiguration dA_construct_config;
	dA_construct_config.blockSize.x =256;
	dA_construct_config.gridSize.x = TNL::roundUpDivision(m,dA_construct_config.blockSize.x);
	TNL::Backend::launchKernelSync(dA_construction_kernel<LBM>, dA_construct_config,
		dLL_lat.getConstView(),
		ws_tnl_dA->getView(),
		ws_tnl_dM.getConstView(),
		diracDeltaTypeLL,
		methodVariant
		);

	loopTimer.stop();
	time_loop_Ha = loopTimer.getRealTime();
	fmt::print("------- matrix A construction time: {}\n", time_loop_Ha);


	// Update the preconditioner
	ws_tnl_dprecond->update(ws_tnl_dA);
	ws_tnl_dsolver.setMatrix(ws_tnl_dA);

	loopTimer.reset();
	loopTimer.start();
	TNL::Matrices::MatrixWriter< dEllpack >::writeMtx( "ws_tnl_dM_method-"+std::to_string((int)methodVariant)+"_dirac-"+std::to_string(diracDeltaTypeEL)+".mtx", ws_tnl_dM );
	TNL::Matrices::MatrixWriter< dEllpack >::writeMtx( "ws_tnl_dA_method-"+std::to_string((int)methodVariant)+"_dirac-"+std::to_string(diracDeltaTypeEL)+".mtx", *ws_tnl_dA );
	loopTimer.stop();
	time_write1 = loopTimer.getRealTime();

	timer.stop();
	fmt::print("-------wuShuTnl final timer time: {}\n",timer.getRealTime());
	time_total = timer.getRealTime();
	cpu_time_total = timer.getCPUTime();
	nlohmann::json j;
	j["threads"] = omp_get_max_threads();
	j["time_total"] = time_total;
	j["cpu_time_total"] = cpu_time_total;
	j["time_loop_Hm"] = time_loop_Hm;
	j["time_loop_Ha"] = time_loop_Ha;
	j["time_loop_Ha_capacities"] = time_loop_Ha_Capacities;
	j["time_loop_Hm_capacities"] = time_loop_Hm_Capacities;
	j["time_loop_Hm_construct"] = time_loop_Hm_Construct;
	j["time_Hm_transpose"] = time_Hm_transpose;
	j["time_write1"] = time_write1;
	j["time_matrixCopy"] = time_matrixCopy;
	std::string jsonOutput = j.dump();
	fmt::print("--outputJSON;{}\n", jsonOutput);
}

template< typename Matrix, typename Vector >
__cuda_callable__
typename Matrix::RealType
rowVectorProduct( const Matrix& matrix, typename Matrix::IndexType i, const Vector& vector )
{
    typename Matrix::RealType result = 0;
    const auto row = matrix.getRow( i );

    for( typename Matrix::IndexType c = 0; c < row.getSize(); c++ ) {
        const typename Matrix::IndexType column = row.getColumnIndex( c );
        if( column != TNL::Matrices::paddingIndex< typename Matrix::IndexType > )
            result += row.getValue( c ) * vector[ column ];
    }

    return result;
}

//require: rho, vx, vy, vz
template< typename LBM >
void Lagrange3D<LBM>::computeForces(real time)
{
	const char* compute_desc = "undefined";
	switch (ws_compute)
	{
		case ws_computeCPU_TNL:                compute_desc = "ws_computeCPU_TNL"; break;
		case ws_computeGPU_TNL:                compute_desc = "ws_computeGPU_TNL"; break;
		case ws_computeHybrid_TNL:             compute_desc = "ws_computeHybrid_TNL"; break;
		case ws_computeHybrid_TNL_zerocopy:    compute_desc = "ws_computeHybrid_TNL_zerocopy"; break;
	}

	switch (ws_compute)
	{
		case ws_computeCPU_TNL:
		case ws_computeHybrid_TNL:
		case ws_computeHybrid_TNL_zerocopy:
			if (!allocated)
				allocateMatricesCPU();
			allocated = true;
			if (!constructed)
				constructMatricesCPU();
			constructed = true;
			break;
		case ws_computeGPU_TNL:
			if (!allocated)
				allocateMatricesGPU();
			allocated = true;
			if (!constructed)
				constructMatricesGPU();
			constructed = true;
			break;
	}
	spdlog::info("computing forces using ws_compute={}", compute_desc);

	TNL::Timer timer;
	timer.start();
	idx n = lbm.lat.global.x()*lbm.lat.global.y()*lbm.lat.global.z();

	const auto drho = dmacroVector(MACRO::e_rho);
	const auto dvx = dmacroVector(MACRO::e_vx);
	const auto dvy = dmacroVector(MACRO::e_vy);
	const auto dvz = dmacroVector(MACRO::e_vz);
	auto dfx = dmacroVector(MACRO::e_fx);
	auto dfy = dmacroVector(MACRO::e_fy);
	auto dfz = dmacroVector(MACRO::e_fz);

	const auto hrho = hmacroVector(MACRO::e_rho);
	const auto hvx = hmacroVector(MACRO::e_vx);
	const auto hvy = hmacroVector(MACRO::e_vy);
	const auto hvz = hmacroVector(MACRO::e_vz);
	auto hfx = hmacroVector(MACRO::e_fx);
	auto hfy = hmacroVector(MACRO::e_fy);
	auto hfz = hmacroVector(MACRO::e_fz);

	switch (ws_compute)
	{
		#ifdef USE_CUDA
		case ws_computeGPU_TNL:
		{
			// no Device--Host copy is required
			ws_tnl_dM.vectorProduct(dvx, ws_tnl_db[0], -1.0);
			ws_tnl_dM.vectorProduct(dvy, ws_tnl_db[1], -1.0);
			ws_tnl_dM.vectorProduct(dvz, ws_tnl_db[2], -1.0);
			// solver
			for (int k=0;k<3;k++) {
				auto start = std::chrono::steady_clock::now();
				ws_tnl_dsolver.solve(ws_tnl_db[k], ws_tnl_dx[k]);
				auto end = std::chrono::steady_clock::now();
				auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				real WT = int_ms * 1e-6;
				log("t={:e}s k={:d} TNL CG solver: WT={:e} iterations={:d} residual={:e}", time, k, WT, ws_tnl_dsolver.getIterations(), ws_tnl_dsolver.getResidue());
			}
			const auto x1 = ws_tnl_dx[0].getConstView();
			const auto x2 = ws_tnl_dx[1].getConstView();
			const auto x3 = ws_tnl_dx[2].getConstView();
			//TNL::Pointers::DevicePointer<dEllpack> MT(ws_tnl_dMT);
			TNL::Pointers::DevicePointer<dEllpack> MT_dptr(ws_tnl_dMT);
			const dEllpack* MT = &MT_dptr.template getData<TNL::Devices::Cuda>();
			auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
			{
				// skipping empty rows explicitly is much faster
				if( MT->getRowCapacity(i) > 0 ) {
					dfx[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x1);
					dfy[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x2);
					dfz[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x3);
				}
			};
			TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, n, kernel);
			break;
		}

		case ws_computeHybrid_TNL:
		{
			ws_tnl_dM.vectorProduct(dvx, ws_tnl_db[0], -1.0);
			ws_tnl_dM.vectorProduct(dvy, ws_tnl_db[1], -1.0);
			ws_tnl_dM.vectorProduct(dvz, ws_tnl_db[2], -1.0);
			// copy to Host
			for (int k=0;k<3;k++) ws_tnl_hb[k] = ws_tnl_db[k];
			// solve on CPU
			for (int k=0;k<3;k++) {
				auto start = std::chrono::steady_clock::now();
				ws_tnl_hsolver.solve(ws_tnl_hb[k], ws_tnl_hx[k]);
				auto end = std::chrono::steady_clock::now();
				auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				real WT = int_ms * 1e-6;
				log("t={:e}s k={:d} TNL CG solver: WT={:e} iterations={:d} residual={:e}", time, k, WT, ws_tnl_hsolver.getIterations(), ws_tnl_hsolver.getResidue());
			}
			// copy to GPU
			for (int k=0;k<3;k++) ws_tnl_dx[k] = ws_tnl_hx[k];
			// continue on GPU
			const auto x1 = ws_tnl_dx[0].getConstView();
			const auto x2 = ws_tnl_dx[1].getConstView();
			const auto x3 = ws_tnl_dx[2].getConstView();
//			TNL::Pointers::DevicePointer<dEllpack> MT(ws_tnl_dMT);
			TNL::Pointers::DevicePointer<dEllpack> MT_dptr(ws_tnl_dMT);
			const dEllpack* MT = &MT_dptr.template getData<TNL::Devices::Cuda>();
			auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
			{
				// skipping empty rows explicitly is much faster
				if( MT->getRowCapacity(i) > 0 ) {
					dfx[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x1);
					dfy[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x2);
					dfz[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x3);
				}
			};
			TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, n, kernel);
			break;
		}

		case ws_computeHybrid_TNL_zerocopy:
		{
			ws_tnl_dM.vectorProduct(dvx, ws_tnl_hbz[0], -1.0);
			ws_tnl_dM.vectorProduct(dvy, ws_tnl_hbz[1], -1.0);
			ws_tnl_dM.vectorProduct(dvz, ws_tnl_hbz[2], -1.0);
			// solve on CPU
			for (int k=0;k<3;k++) {
				auto start = std::chrono::steady_clock::now();
				ws_tnl_hsolver.solve(ws_tnl_hbz[k].getView(), ws_tnl_hxz[k]);
				auto end = std::chrono::steady_clock::now();
				auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				real WT = int_ms * 1e-6;
				log("t={:e}s k={:d} TNL CG solver: WT={:e} iterations={:d} residual={:e}", time, k, WT, ws_tnl_hsolver.getIterations(), ws_tnl_hsolver.getResidue());
			}
			// continue on GPU
			const auto x1 = ws_tnl_hxz[0].getConstView();
			const auto x2 = ws_tnl_hxz[1].getConstView();
			const auto x3 = ws_tnl_hxz[2].getConstView();
//			TNL::Pointers::DevicePointer<dEllpack> MT(ws_tnl_dMT);
			TNL::Pointers::DevicePointer<dEllpack> MT_dptr(ws_tnl_dMT);
			const dEllpack* MT = &MT_dptr.template getData<TNL::Devices::Cuda>();
			auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
			{
				// skipping empty rows explicitly is much faster
				if( MT->getRowCapacity(i) > 0 ) {
					dfx[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x1);
					dfy[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x2);
					dfz[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x3);
				}
			};
			TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, n, kernel);
			break;
		}
		#endif // USE_CUDA
		case ws_computeCPU_TNL:
		{
			// vx, vy, vz, rho must be copied from the device
			ws_tnl_hM.vectorProduct(hvx, ws_tnl_hb[0], -1.0);
			ws_tnl_hM.vectorProduct(hvy, ws_tnl_hb[1], -1.0);
			ws_tnl_hM.vectorProduct(hvz, ws_tnl_hb[2], -1.0);
			// solver
			for (int k=0;k<3;k++) {
				auto start = std::chrono::steady_clock::now();
				ws_tnl_hsolver.solve(ws_tnl_hb[k], ws_tnl_hx[k]);
				auto end = std::chrono::steady_clock::now();
				auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				real WT = int_ms * 1e-6;
				log("t={:e}s k={:d} TNL CG solver: WT={:e} iterations={:d} residual={:e}", time, k, WT, ws_tnl_hsolver.getIterations(), ws_tnl_hsolver.getResidue());
			}
			auto kernel = [&] (idx i) mutable
			{
				// skipping empty rows explicitly is much faster
				if( ws_tnl_hMT.getRowCapacity(i) > 0 ) {
					hfx[i] += 2 * hrho[i] * rowVectorProduct(ws_tnl_hMT, i, ws_tnl_hx[0]);
					hfy[i] += 2 * hrho[i] * rowVectorProduct(ws_tnl_hMT, i, ws_tnl_hx[1]);
					hfz[i] += 2 * hrho[i] * rowVectorProduct(ws_tnl_hMT, i, ws_tnl_hx[2]);
				}
			};
			TNL::Algorithms::parallelFor< TNL::Devices::Host >((idx) 0, n, kernel);
			// copy forces to the device
			// FIXME: this is copied multiple times when there are multiple Lagrange3D objects
			// (ideally there should be only one Lagrange3D object that comprises all immersed bodies)
			dfx = hfx;
			dfy = hfy;
			dfz = hfz;
			break;
		}
		default:
			fmt::print(stderr, "lagrange_3D: Wu Shu compute flag {} unrecognized.\n", ws_compute);
			break;
	}
	timer.stop();
	double time_total = timer.getRealTime();
	double cpu_time_total = timer.getCPUTime();

	nlohmann::json j;
	j["threads"]= omp_get_max_threads();
	j["time_total"] = time_total;
	j["cpu_time_total"] = cpu_time_total;
	j["object_id"]=obj_id+1;

	std::string jsonOutput = j.dump();
	fmt::print("--outputCalculationJSON;{}\n",jsonOutput);
}

template< typename LBM >
template< typename... ARGS >
void Lagrange3D<LBM>::log(const char* fmts, ARGS... args)
{
	FILE* f = fopen(logfile.c_str(), "at"); // append information
	if (f==0) {
		fmt::print(stderr, "unable to create/access file {}\n", logfile);
		return;
	}
	// insert time stamp
	fmt::print(f, "{} ", timestamp());
	fmt::print(f, fmts, args...);
	fmt::print(f, "\n");
	fclose(f);

	//fmt::print(fmts, args...);
	//fmt::print("\n");
}

template< typename LBM >
Lagrange3D<LBM>::Lagrange3D(LBM &inputLBM, const std::string& resultsDir,int obj_id) : lbm(inputLBM)
{
	logfile = fmt::format("{}/ibm_solver.log", resultsDir);
	this->obj_id = obj_id;
	ws_tnl_hsolver.setMaxIterations(10000);
	ws_tnl_hsolver.setConvergenceResidue(3e-4);
	ws_tnl_hprecond = std::make_shared< hPreconditioner >();
//	ws_tnl_hsolver.setPreconditioner(ws_tnl_hprecond);
	#ifdef USE_CUDA
	ws_tnl_dsolver.setMaxIterations(10000);
	ws_tnl_dsolver.setConvergenceResidue(3e-4);
	ws_tnl_dprecond = std::make_shared< dPreconditioner >();
//	ws_tnl_dsolver.setPreconditioner(ws_tnl_dprecond);
	#endif
}

template< typename LBM >
Lagrange3D<LBM>::~Lagrange3D()
{
	if (index_array)
	{
		for (int i=0;i<lag_X;i++) delete [] index_array[i];
		delete [] index_array;
	}
}
