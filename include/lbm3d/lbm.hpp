#pragma once

#include "lbm.h"

template< typename LBM_TYPE >
LBM<LBM_TYPE>::LBM(const TNL::MPI::Comm& communicator, lat_t ilat, real iphysViscosity, real iphysDt)
: communicator(communicator), lat(ilat)
{
	// initialize MPI info
	rank = communicator.rank();
	nproc = communicator.size();

	// uniform decomposition by default
	auto local_range = TNL::Containers::Partitioner<idx>::splitRange(lat.global.x(), communicator);
	idx3d local, offset;
	local.x() = local_range.getEnd() - local_range.getBegin();
	local.y() = lat.global.y();
	local.z() = lat.global.z();
	offset.x() = local_range.getBegin();
	offset.y() = offset.z() = 0;
	int neighbour_left = (rank - 1 + nproc) % nproc;
	int neighbour_right = (rank + 1 + nproc) % nproc;
	blocks.emplace_back(communicator, lat.global, local, offset, neighbour_left, neighbour_right, neighbour_left, rank, neighbour_right);
	total_blocks = nproc;

	physDt = iphysDt;
	physCharLength = lat.physDl * (real)lat.global.y();
	physViscosity = iphysViscosity;
}

template< typename LBM_TYPE >
LBM<LBM_TYPE>::LBM(const TNL::MPI::Comm& communicator, lat_t ilat, std::vector<BLOCK>&& blocks, real iphysViscosity, real iphysDt)
: communicator(communicator), lat(ilat), blocks(std::forward<std::vector<BLOCK>>(blocks))
{
	// initialize MPI info
	rank = communicator.rank();
	nproc = communicator.size();

	total_blocks = TNL::MPI::reduce(blocks.size(), MPI_SUM, communicator);

	physDt = iphysDt;
	physCharLength = lat.physDl * (real)lat.global.y();
	physViscosity = iphysViscosity;
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::resetForces(real ifx, real ify, real ifz)
{
	for( auto& block : blocks )
		block.resetForces(ifx, ify, ifz);
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyForcesToDevice()
{
	for( auto& block : blocks )
		block.copyForcesToDevice();
}


template< typename LBM_TYPE >
bool LBM<LBM_TYPE>::isAnyLocalIndex(idx x, idx y, idx z)
{
	for( auto& block : blocks )
		if( block.isLocalIndex(x, y, z) )
			return true;
	return false;
}

template< typename LBM_TYPE >
bool LBM<LBM_TYPE>::isAnyLocalX(idx x)
{
	for( auto& block : blocks )
		if( block.isLocalX(x) )
			return true;
	return false;
}

template< typename LBM_TYPE >
bool LBM<LBM_TYPE>::isAnyLocalY(idx y)
{
	for( auto& block : blocks )
		if( block.isLocalY(y) )
			return true;
	return false;
}

template< typename LBM_TYPE >
bool LBM<LBM_TYPE>::isAnyLocalZ(idx z)
{
	for( auto& block : blocks )
		if( block.isLocalZ(z) )
			return true;
	return false;
}


template< typename LBM_TYPE >
void LBM<LBM_TYPE>::setMap(idx x, idx y, idx z, map_t value)
{
	for( auto& block : blocks )
		block.setMap(x, y, z, value);
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::setBoundaryX(idx x, map_t value)
{
	for( auto& block : blocks )
		block.setBoundaryX(x, value);
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::setBoundaryY(idx y, map_t value)
{
	for( auto& block : blocks )
		block.setBoundaryY(y, value);
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::setBoundaryZ(idx z, map_t value)
{
	for( auto& block : blocks )
		block.setBoundaryZ(z, value);
}

template< typename LBM_TYPE >
bool LBM<LBM_TYPE>::isFluid(idx x, idx y, idx z)
{
	for( auto& block : blocks )
		if (block.isFluid(x, y, z))
			return true;
	return false;
}


template< typename LBM_TYPE >
void LBM<LBM_TYPE>::resetMap(map_t geo_type)
{
	for( auto& block : blocks )
		block.resetMap(geo_type);
}


template< typename LBM_TYPE >
void  LBM<LBM_TYPE>::copyMapToHost()
{
	for( auto& block : blocks )
		block.copyMapToHost();
}

template< typename LBM_TYPE >
void  LBM<LBM_TYPE>::copyMapToDevice()
{
	for( auto& block : blocks )
		block.copyMapToDevice();
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyMacroToHost()
{
	for( auto& block : blocks )
		block.copyMacroToHost();
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyMacroToDevice()
{
	for( auto& block : blocks )
		block.copyMacroToDevice();
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyDFsToHost(uint8_t dfty)
{
	for( auto& block : blocks )
		block.copyDFsToHost(dfty);
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyDFsToDevice(uint8_t dfty)
{
	for( auto& block : blocks )
		block.copyDFsToDevice(dfty);
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyDFsToHost()
{
	for( auto& block : blocks )
		block.copyDFsToHost();
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyDFsToDevice()
{
	for( auto& block : blocks )
		block.copyDFsToDevice();
}

#ifdef HAVE_MPI
template< typename LBM_TYPE >
void LBM<LBM_TYPE>::synchronizeDFsAndMacroDevice(uint8_t dftype)
{
	// stage 0: set inputs, allocate buffers
	// stage 1: fill send buffers
	for( auto& block : blocks ) {
		block.synchronizeDFsDevice_start(dftype);
		if (MACRO::use_syncMacro)
			block.synchronizeMacroDevice_start();
	}

	// stage 2: issue all send and receive async operations
	for( auto& block : blocks ) {
		for (int i = 0; i < LBM_TYPE::Q; i++)
			block.dreal_sync[i].stage_2();
		if (MACRO::use_syncMacro)
			for (int i = 0; i < MACRO::N; i++)
				block.dreal_sync[LBM_TYPE::Q + i].stage_2();
	}

	// stage 3: copy data from receive buffers
	for( auto& block : blocks ) {
		for (int i = 0; i < LBM_TYPE::Q; i++)
			block.dreal_sync[i].stage_3();
		if (MACRO::use_syncMacro)
			for (int i = 0; i < MACRO::N; i++)
				block.dreal_sync[LBM_TYPE::Q + i].stage_3();
	}

	// stage 4: ensure everything has finished
	for( auto& block : blocks ) {
		for (int i = 0; i < LBM_TYPE::Q; i++)
			block.dreal_sync[i].stage_4();
		if (MACRO::use_syncMacro)
			for (int i = 0; i < MACRO::N; i++)
				block.dreal_sync[LBM_TYPE::Q + i].stage_4();
	}
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::synchronizeMapDevice()
{
	for( auto& block : blocks )
		block.synchronizeMapDevice_start();
	for( auto& block : blocks )
		block.map_sync.wait();
}
#endif  // HAVE_MPI

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::computeCPUMacroFromLat()
{
	for( auto& block : blocks )
		block.computeCPUMacroFromLat();
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::allocateHostData()
{
	for( auto& block : blocks )
		block.allocateHostData();
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::allocateDeviceData()
{
	for( auto& block : blocks )
		block.allocateDeviceData();
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::updateKernelData()
{
	for( auto& block : blocks )
	{
		// needed for A-A pattern
		block.data.even_iter = (iterations % 2) == 0;

		// rotation (no-op for A-A pattern ... DFMAX=1)
		int i = iterations % DFMAX; 			// i = 0, 1, 2, ... DMAX-1

		for (int k=0;k<DFMAX;k++)
		{
			int knew = (k-i)<=0 ? (k-i+DFMAX) % DFMAX : k-i;
	//		block.data.dfs[k] = block.dfs[knew];
			block.data.dfs[k] = block.dfs[knew].getData();
	//		printf("updateKernelData:: assigning data.dfs[%d] = dfs[%d]\n",k, knew);
		}
	}
}

template< typename LBM_TYPE >
	template< typename F >
void LBM<LBM_TYPE>::forLocalLatticeSites(F f)
{
	for( auto& block : blocks )
		block.forLocalLatticeSites(f);
}

template< typename LBM_TYPE >
	template< typename F >
void LBM<LBM_TYPE>::forAllLatticeSites(F f)
{
	for( auto& block : blocks )
		block.forAllLatticeSites(f);
}
