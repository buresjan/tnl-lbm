#pragma once

#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <string.h>
#include <iostream>
#include <png.h>
#include "ciselnik.h"

#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/DistributedNDArray.h>
#include <TNL/Containers/DistributedNDArraySynchronizer.h>
#include <TNL/Containers/Partitioner.h>
#include <TNL/MPI.h>

using TNLMPI_INIT = TNL::MPI::ScopedInitializer;


#ifdef __CUDACC__
	#define CUDA_HOSTDEV __host__ __device__
	#define CUDA_HOSTDEV_NOINLINE CUDA_HOSTDEV __noinline__
#else
	#define CUDA_HOSTDEV
	#define CUDA_HOSTDEV_NOINLINE
#endif

#ifdef USE_CUDA
	#define checkCudaDevice TNL_CHECK_CUDA_DEVICE
	#include <cuda_profiler_api.h>
#endif // USE_CUDA

#ifdef HAVE_MPI
	// CUDA streams for overlapping computation and communication
	cudaStream_t cuda_streams[3];
#endif


// number of dist. functions, default=2
// quick fix, use templates to define DFMAX ... through TRAITS maybe ?
#ifdef USE_DFMAX3
enum : uint8_t { df_cur, df_out, df_prev, DFMAX }; // special 3 dfs
#else
enum : uint8_t { df_cur, df_out, DFMAX }; // default 2 dfs
#endif

template <
	typename _dreal = float,	// real number representation on GPU
	typename _real = double,	// real number representation on CPU
	typename _idx = long int,	// array index on CPU and GPU (can be very large)
	typename _map_t = short int
>
struct Traits
{
	using real = _real;
	using dreal = _dreal;
	using idx = _idx;
	using map_t = _map_t;

	using xyz_permutation = std::index_sequence< 0, 2, 1 >;		// x, z, y
	using d4_permutation = std::index_sequence< 0, 1, 3, 2 >;		// id, x, z, y

#ifdef HAVE_MPI
	using xyz_overlaps = std::index_sequence< 1, 0, 0 >;	// x, y, z
	using d4_overlaps = std::index_sequence< 0, 1, 0, 0 >;	// id, x, y, z
#else
	using xyz_overlaps = std::index_sequence< 0, 0, 0 >;	// x, y, z
	using d4_overlaps = std::index_sequence< 0, 0, 0, 0 >;	// id, x, y, z
#endif

	template< typename Value, typename Device >
	using array3d = TNL::Containers::NDArray<
		Value,
		TNL::Containers::SizesHolder< idx, 0, 0, 0 >,	// x, y, z
		xyz_permutation,
		Device,
		idx,
		xyz_overlaps >;
	template< std::size_t N, typename Value, typename Device >
	using array4d = TNL::Containers::NDArray<
		Value,
		TNL::Containers::SizesHolder< idx, N, 0, 0, 0 >,	// N, x, y, z
		d4_permutation,
		Device,
		idx,
		d4_overlaps >;

	using xyz_indexer_t = typename array3d<dreal, TNL::Devices::Cuda>::IndexerType;
};

using TraitsSP = Traits<float>; //_dreal is float only
using TraitsDP = Traits<double>;

template<
	typename _COLL,
	typename _DATA,
	template<typename> class _BC,
	typename _EQ,
	typename _STREAMING,
	typename _MACRO,
	typename _CPU_MACRO,
	typename _TRAITS
>
struct D3Q27
{
	using COLL=_COLL;
	using DATA=_DATA;
	using BC=_BC<D3Q27>;
	using EQ=_EQ;
	using STREAMING=_STREAMING;
	using MACRO=_MACRO;
	using CPU_MACRO=_CPU_MACRO;
	using TRAITS=_TRAITS;

	// KernelStruct
	template < typename REAL >
	struct KernelStruct
	{
		REAL f[27];
		REAL fz=0, fx=0, fy=0;
		REAL vz=0, vx=0, vy=0;
		REAL rho=1.0, lbmViscosity=1.0;

	#if defined(USE_CYMODEL) || defined(USE_CASSON)
		REAL S11=0.,S12=0.,S22=0.,S32=0.,S13=0.,S33=0.;

		//Non-Newtonian parameters
		#if defined(USE_CYMODEL)
		REAL lbm_nu0=0, lbm_lambda=0, lbm_a=0, lbm_n=0;
		#elif defined(USE_CASSON)
		REAL lbm_k0=0, lbm_k1=0;
		#endif

		REAL mu;
	#endif
	};

	using __hmap_array_t = typename TRAITS::template array3d<typename TRAITS::map_t, TNL::Devices::Host>;
	using __dmap_array_t = typename TRAITS::template array3d<typename TRAITS::map_t, TNL::Devices::Cuda>;
	using __bool_array_t = typename TRAITS::template array3d<bool, TNL::Devices::Host>;

	using __hlat_array_t = typename TRAITS::template array4d<27, typename TRAITS::dreal, TNL::Devices::Host>;
	using __dlat_array_t = typename TRAITS::template array4d<27, typename TRAITS::dreal, TNL::Devices::Cuda>;

	using __hmacro_array_t = typename TRAITS::template array4d<MACRO::N, typename TRAITS::dreal, TNL::Devices::Host>;
	using __dmacro_array_t = typename TRAITS::template array4d<MACRO::N, typename TRAITS::dreal, TNL::Devices::Cuda>;
	using __cpumacro_array_t = typename TRAITS::template array4d<CPU_MACRO::N, typename TRAITS::dreal, TNL::Devices::Host>;

#ifdef HAVE_MPI
	using sync_array_t = TNL::Containers::DistributedNDArray< typename TRAITS::template array3d<typename TRAITS::dreal, TNL::Devices::Cuda > >;

	using hmap_array_t = TNL::Containers::DistributedNDArray< __hmap_array_t >;
	using dmap_array_t = TNL::Containers::DistributedNDArray< __dmap_array_t >;
	using bool_array_t = TNL::Containers::DistributedNDArray< __bool_array_t >;

	using hlat_array_t = TNL::Containers::DistributedNDArray< __hlat_array_t >;
	using dlat_array_t = TNL::Containers::DistributedNDArray< __dlat_array_t >;

	using hmacro_array_t = TNL::Containers::DistributedNDArray< __hmacro_array_t >;
	using dmacro_array_t = TNL::Containers::DistributedNDArray< __dmacro_array_t >;
	using cpumacro_array_t = TNL::Containers::DistributedNDArray< __cpumacro_array_t >;
#else
	using sync_array_t = typename TRAITS::template array3d<typename TRAITS::dreal, TNL::Devices::Cuda>;

	using hmap_array_t = __hmap_array_t;
	using dmap_array_t = __dmap_array_t;
	using bool_array_t = __bool_array_t;

	using hlat_array_t = __hlat_array_t;
	using dlat_array_t = __dlat_array_t;

	using hmacro_array_t = __hmacro_array_t;
	using dmacro_array_t = __dmacro_array_t;
	using cpumacro_array_t = __cpumacro_array_t;
#endif

	using hmap_view_t = typename hmap_array_t::ViewType;
	using dmap_view_t = typename dmap_array_t::ViewType;
	using bool_view_t = typename bool_array_t::ViewType;

	using hlat_view_t = typename hlat_array_t::ViewType;
	using dlat_view_t = typename dlat_array_t::ViewType;
};


//#define USE_HIGH_PRECISION_RHO // use num value ordering to compute rho inlbm_common.h .. slow!!!
//#define USE_GALILEAN_CORRECTION // Geier 2015: use Gal correction in BKG and CUM?
//#define USE_GEIER_CUM_2017 // use Geier 2017 Cummulant improvement A,B terms
//#define USE_GEIER_CUM_ANTIALIAS // use antialiasing Dxu, Dyv, Dzw from Geier 2015/2017

#define MAX( a , b) (((a)>(b))?(a):(b))
#define MIN( a , b) (((a)<(b))?(a):(b))

#define FILENAME_CHARS 500

#define SQ(x) ((x) * (x)) // square function; replaces SQ(x) by ((x) * (x)) in the code
#define NORM(x, y, z) sqrt(SQ(x) + SQ(y) + SQ(z))

enum { SOLVER_UMFPACK, SOLVER_PETSC };

/*
// ordering suitable for esotwist - opposite directions have IDs different by 13
// (first half)
enum
{
	pzz=0,
	zpz=1,
	zzp=2,
	ppz=3,
	pzp=4,
	zpp=5,
	ppp=6,
	ppm=7,
	pmp=8,
	mpp=9,
	zpm=10,
	pzm=11,
	pmz=12,
	// (second half)
	mzz=13,
	zmz=14,
	zzm=15,
	mmz=16,
	mzm=17,
	zmm=18,
	mmm=19,
	mmp=20,
	mpm=21,
	pmm=22,
	zmp=23,
	mzp=24,
	mpz=25,
	// (central)
	zzz=26
};
*/
// ordering suitable for MPI and 1D distribution
// (this ordering is not necessary for MPI anymore - but df_sync_directions must be kept consistent)
enum
{
	// left third
	mmm=0,
	mmz=1,
	mmp=2,
	mzm=3,
	mzz=4,
	mzp=5,
	mpm=6,
	mpz=7,
	mpp=8,
	// central third
	zmm=9,
	zmz=10,
	zmp=11,
	zzm=12,
	zzz=13,
	zzp=14,
	zpm=15,
	zpz=16,
	zpp=17,
	// right third
	pmm=18,
	pmz=19,
	pmp=20,
	pzm=21,
	pzz=22,
	pzp=23,
	ppm=24,
	ppz=25,
	ppp=26
};

// static array of sync directions for the MPI synchronizer
// (indexing must correspond to the enum above)
static TNL::Containers::SyncDirection df_sync_directions[27] = {
	// left third
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Left,
	// central third
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	// right third
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Right,
};

// vopicarna
CUDA_HOSTDEV inline
int directionIndex(int i, int j, int k)
{
	if (i==-1 && j==-1 && k==-1) return mmm;
	if (i==-1 && j==-1 && k== 0) return mmz;
	if (i==-1 && j==-1 && k== 1) return mmp;
	if (i==-1 && j== 0 && k==-1) return mzm;
	if (i==-1 && j== 0 && k== 0) return mzz;
	if (i==-1 && j== 0 && k== 1) return mzp;
	if (i==-1 && j== 1 && k==-1) return mpm;
	if (i==-1 && j== 1 && k== 0) return mpz;
	if (i==-1 && j== 1 && k== 1) return mpp;

	if (i== 0 && j==-1 && k==-1) return zmm;
	if (i== 0 && j==-1 && k== 0) return zmz;
	if (i== 0 && j==-1 && k== 1) return zmp;
	if (i== 0 && j== 0 && k==-1) return zzm;
	if (i== 0 && j== 0 && k== 0) return zzz;
	if (i== 0 && j== 0 && k== 1) return zzp;
	if (i== 0 && j== 1 && k==-1) return zpm;
	if (i== 0 && j== 1 && k== 0) return zpz;
	if (i== 0 && j== 1 && k== 1) return zpp;

	if (i== 1 && j==-1 && k==-1) return pmm;
	if (i== 1 && j==-1 && k== 0) return pmz;
	if (i== 1 && j==-1 && k== 1) return pmp;
	if (i== 1 && j== 0 && k==-1) return pzm;
	if (i== 1 && j== 0 && k== 0) return pzz;
	if (i== 1 && j== 0 && k== 1) return pzp;
	if (i== 1 && j== 1 && k==-1) return ppm;
	if (i== 1 && j== 1 && k== 0) return ppz;
	if (i== 1 && j== 1 && k== 1) return ppp;
	return zzz;
}


#define Main main // TNL fix when LBM is included into TNL


// default
#include "lbm_data.h"  // LBM_Data is a general template (for any Q)
#include "d3q27_macro.h"
#include "d3q27_bc.h"

#include "d3q27_eq.h"
#include "d3q27_eq_inv_cum.h"
#include "d3q27_eq_well.h"
#include "d3q27_eq_entropic.h"

#include "d3q27_streaming.h"

#include "d3q27_col_cum.h"
#include "d3q27_col_bgk.h"
#include "d3q27_col_clbm.h"
#include "d3q27_col_fclbm.h"
#include "d3q27_col_mrt.h"
#include "d3q27_col_srt.h"
#include "d3q27_col_cum_sgs.h"
#include "d3q27_col_kbc_n.h"
#include "d3q27_col_kbc_c.h"
#include "d3q27_col_srt_modif_force.h"
#include "d3q27_col_clbm_fei.h"

#include "d3q27_col_srt_well.h"
#include "d3q27_col_clbm_well.h"
#include "d3q27_col_cum_well.h"
#include "d3q27_col_bgk_well.h"
