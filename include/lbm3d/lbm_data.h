#pragma once

template < typename TRAITS >
struct LBM_Data
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using map_t = typename TRAITS::map_t;
	using indexer_t = typename TRAITS::xyz_indexer_t;

	// indexing
	indexer_t indexer;
	idx XYZ;	// precomputed indexer.getStorageSize(), i.e. product of (X+overlaps_x)*(Y+overlaps_y)*(Z+overlaps_z)

	// array pointers
	dreal* dfs[DFMAX];
	dreal* dmacro;
	map_t* dmap;

	// scalars
	dreal lbmViscosity;
	// homogeneous force field
	dreal fx = 0;
	dreal fy = 0;
	dreal fz = 0;

	// sizes NOT including overlaps
	CUDA_HOSTDEV idx X() { return indexer.template getSize<0>(); }
	CUDA_HOSTDEV idx Y() { return indexer.template getSize<1>(); }
	CUDA_HOSTDEV idx Z() { return indexer.template getSize<2>(); }

	CUDA_HOSTDEV map_t map(idx x, idx y, idx z)
	{
		return dmap[indexer.getStorageIndex(x, y, z)];
	}

	CUDA_HOSTDEV idx Fxyz(int q, idx x, idx y, idx z)
	{
		return q * XYZ + indexer.getStorageIndex(x, y, z);
	}

	CUDA_HOSTDEV dreal& df(uint8_t type, int q, idx x, idx y, idx z)
	{
		return dfs[type][Fxyz(q,x,y,z)];
	}

	CUDA_HOSTDEV dreal& macro(int id, idx x, idx y, idx z)
	{
		return dmacro[Fxyz(id, x, y, z)];
	}
};


template < typename TRAITS >
struct LBM_Data_ConstInflow : LBM_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	dreal inflow_rho=no1;
	dreal inflow_vx=0;
	dreal inflow_vy=0;
	dreal inflow_vz=0;

	template < typename LBM_KS >
	CUDA_HOSTDEV void inflow(LBM_KS &KS, idx x, idx y, idx z)
	{
		KS.rho = inflow_rho;
		KS.vx  = inflow_vx;
		KS.vy  = inflow_vy;
		KS.vz  = inflow_vz;
	}
};

template < typename TRAITS >
struct LBM_Data_NoInflow : LBM_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template < typename LBM_KS >
	CUDA_HOSTDEV void inflow(LBM_KS &KS, idx x, idx y, idx z)
	{
		KS.rho = no1;
		KS.vx  = 0;
		KS.vy  = 0;
		KS.vz  = 0;
	}
};
