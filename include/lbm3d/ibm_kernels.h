#pragma once

#include "defs.h"
#include "lbm3d/lagrange_3D.h"
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/SparseMatrix.h>
#include "dirac.h"



//TODO: Add ifdef __CUDACC__
#ifdef USE_CUDA
template < typename LBM >
__global__ void dM_row_capacities_kernel(
	typename Lagrange3D<LBM>::DLPVECTOR_DREAL::ConstViewType LL,
	typename Lagrange3D<LBM>::dEllpack::RowCapacitiesType::ViewType dM_row_capacities,
	typename LBM::TRAITS::idx3d lbmBlockLocal,
	int diracDeltaTypeEL
	)
{
	using idx = typename LBM::TRAITS::idx;
	using real = typename Lagrange3D<LBM>::DLPVECTOR_DREAL::RealType::Real;

	idx support=5; // search in this support
	//Blocksize = blockDim
	idx i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i>=dM_row_capacities.getSize())
	{
		return;
	}
	idx fi_x = floor(LL[i].x);
	idx fi_y = floor(LL[i].y);
	idx fi_z = floor(LL[i].z);

	// FIXME: iterate over LBM blocks
	for (int gz=MAX( 0, fi_z - support);gz<MIN(lbmBlockLocal.z(), fi_z + support);gz++)
	for (int gy=MAX( 0, fi_y - support);gy<MIN(lbmBlockLocal.y(), fi_y + support);gy++)
	for (int gx=MAX( 0, fi_x - support);gx<MIN(lbmBlockLocal.x(), fi_x + support);gx++)
	{
		//real dd = diracDelta((real)(gx + 0.5) - LL[i].x/lbm.lat.physDl) * diracDelta((real)(gy + 0.5) - LL[i].y/lbm.lat.physDl) * diracDelta((real)(gz + 0.5) - LL[i].z/lbm.lat.physDl);
		if (
			isDDNonZero(diracDeltaTypeEL,(real)(gx + 0.5) - LL[i].x)&&
			isDDNonZero(diracDeltaTypeEL,(real)(gy + 0.5) - LL[i].y)&&
			isDDNonZero(diracDeltaTypeEL,(real)(gz + 0.5) - LL[i].z)
		)
		{
			dM_row_capacities[i]++;
		}
	}
}


template < typename LBM>
__global__ void dM_construction_kernel(
	typename Lagrange3D<LBM>::DLPVECTOR_DREAL::ConstViewType LL,
	typename Lagrange3D<LBM>::dEllpack::ViewType ws_tnl_dM,
	typename LBM::TRAITS::idx3d lbmBlockLocal,
	#ifdef HAVE_MPI
	typename LBM::BLOCK::dmap_array_t::ConstLocalViewType dmap,
	#else
	typename LBM::BLOCK::dmap_array_t::ConstViewType dmap,
	#endif
	int diracDeltaTypeEL
)
{
	using idx = typename LBM::TRAITS::idx;
	using real = typename Lagrange3D<LBM>::DLPVECTOR_DREAL::RealType::Real;
	idx support=5; // search in this support
	//Blocksize = blockDim
	idx i = blockIdx.x * blockDim.x + threadIdx.x;
	//m = LL.size()
	if(i>=LL.getSize())
	{
		return;
	}
	idx fi_x = floor(LL[i].x);
	idx fi_y = floor(LL[i].y);
	idx fi_z = floor(LL[i].z);

	// FIXME: iterate over LBM blocks
	for (int gz=MAX( 0, fi_z - support);gz<MIN(lbmBlockLocal.z(), fi_z + support);gz++)
	for (int gy=MAX( 0, fi_y - support);gy<MIN(lbmBlockLocal.y(), fi_y + support);gy++)
	for (int gx=MAX( 0, fi_x - support);gx<MIN(lbmBlockLocal.x(), fi_x + support);gx++)
	{

		if
		(
			isDDNonZero(diracDeltaTypeEL,(real)(gx + 0.5) - LL[i].x)&&
			isDDNonZero(diracDeltaTypeEL,(real)(gy + 0.5) - LL[i].y)&&
			isDDNonZero(diracDeltaTypeEL,(real)(gz + 0.5) - LL[i].z)
		)
		{
			real dd =
			diracDelta(diracDeltaTypeEL,(real)(gx + 0.5) - LL[i].x) *
			diracDelta(diracDeltaTypeEL,(real)(gy + 0.5) - LL[i].y) *
			diracDelta(diracDeltaTypeEL,(real)(gz + 0.5) - LL[i].z);
			idx index = dmap.getStorageIndex(gx,gy,gz);
			ws_tnl_dM.setElement(i,index,dd);
		}
	}
}

template < typename LBM>
__global__ void dA_row_capacities_kernel(
	typename Lagrange3D<LBM>::DLPVECTOR_DREAL::ConstViewType LL,
	typename Lagrange3D<LBM>::dEllpack::RowCapacitiesType::ViewType dA_row_capacities,
	typename Lagrange3D<LBM>::dEllpack::ConstViewType ws_tnl_dM,
	int diracDeltaTypeLL,
	DiracMethod methodVariant
)
{
	using idx = typename LBM::TRAITS::idx;
	using real = typename Lagrange3D<LBM>::DLPVECTOR_DREAL::RealType::Real;

	//Blocksize = blockDim
	idx index_row = blockIdx.x * blockDim.x + threadIdx.x;
	int m = dA_row_capacities.getSize();
	if(index_row >= m)
	{
		return;
	}

	int rowCapacity = 0;  //Number of elements where DiracDelta > 0 //old
	for (int index_col=0;index_col<m;index_col++) //old
	{
		if (methodVariant==DiracMethod::MODIFIED)
		{
			if(is3DiracNonZero(diracDeltaTypeLL, index_col, index_row,LL))
			{
				rowCapacity++;
			}
		} else
		{
			real val=0;
			auto row1 = ws_tnl_dM.getRow(index_row);
			auto row2 = ws_tnl_dM.getRow(index_col);
			for (idx in1=0; in1 < row1.getSize(); in1++)
			for (idx in2=0; in2 < row2.getSize(); in2++)
			{
				if (row1.getColumnIndex(in1) == row2.getColumnIndex(in2))
				{
					val += row1.getValue(in1) * row2.getValue(in2);
					break;
				}
			}
			if (val > 0)
			rowCapacity++;
		}
	}
	dA_row_capacities[index_row] = rowCapacity;
}
template < typename LBM>
__global__ void dA_construction_kernel(
	typename Lagrange3D<LBM>::DLPVECTOR_DREAL::ConstViewType LL,
	typename Lagrange3D<LBM>::dEllpack::ViewType ws_tnl_dA,
	typename Lagrange3D<LBM>::dEllpack::ConstViewType ws_tnl_dM,
	int diracDeltaTypeLL,
	DiracMethod methodVariant
)
{
	using idx = typename LBM::TRAITS::idx;
	using real = typename Lagrange3D<LBM>::DLPVECTOR_DREAL::RealType::Real;

	//Blocksize = blockDim
	idx index_row = blockIdx.x * blockDim.x + threadIdx.x;
	int m = LL.getSize();
	if(index_row >= m)
	{
		return;
	}
	for (int index_col=0;index_col<m;index_col++)
	{
		if (methodVariant==DiracMethod::MODIFIED)
		{
			if(is3DiracNonZero(diracDeltaTypeLL, index_col, index_row,LL))
			{
				//calculate dirac with selected dirac type
				real ddd = calculate3Dirac(diracDeltaTypeLL, index_col, index_row,LL);
				ws_tnl_dA.setElement(index_row,index_col, ddd);
			}
		} else
		{
			real val=0;
			auto row1 = ws_tnl_dM.getRow(index_row);
			auto row2 = ws_tnl_dM.getRow(index_col);
			for (idx in1=0; in1 < row1.getSize(); in1++)
			for (idx in2=0; in2 < row2.getSize(); in2++)
			{
				if (row1.getColumnIndex(in1) == row2.getColumnIndex(in2))
				{
					val += row1.getValue(in1) * row2.getValue(in2);
					break;
				}
			}
			if (val > 0)
			{
				ws_tnl_dA.setElement(index_row,index_col, val);
			}
		}
	}
}
#endif //USE_CUDA
