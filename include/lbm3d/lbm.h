#pragma once

#include "defs.h"

template<
	typename T_LBM_TYPE
	// TODO: const int Q
>
struct LBM
{
	using LBM_TYPE = T_LBM_TYPE;
	using MACRO = typename LBM_TYPE::MACRO;
	using CPU_MACRO = typename LBM_TYPE::CPU_MACRO;
	using TRAITS = typename LBM_TYPE::TRAITS;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using real = typename TRAITS::real;
	using map_t = typename TRAITS::map_t;
	using point_t = typename TRAITS::point_t;
	using idx3d = typename TRAITS::idx3d;

	using hmap_array_t = typename LBM_TYPE::hmap_array_t;
	using dmap_array_t = typename LBM_TYPE::dmap_array_t;
	using bool_array_t = typename LBM_TYPE::bool_array_t;
	using hlat_array_t = typename LBM_TYPE::hlat_array_t;
	using dlat_array_t = typename LBM_TYPE::dlat_array_t;
	using dlat_view_t = typename LBM_TYPE::dlat_view_t;
	using hmacro_array_t = typename LBM_TYPE::hmacro_array_t;
	using dmacro_array_t = typename LBM_TYPE::dmacro_array_t;
	using cpumacro_array_t = typename LBM_TYPE::cpumacro_array_t;
	using sync_array_t = typename LBM_TYPE::sync_array_t;

	map_t null=0;
	dreal drealnull=-1e88;

	// KernelData contains only the necessary data for the CUDA kernel. these are copied just before the kernel is called
	typename LBM_TYPE::DATA data;
	// block size for the LBM kernel (used in core.h)
	int block_size = 128;

	hmap_array_t hmap;
	dmap_array_t dmap;
	bool_array_t wall; // indicates whether there is a wall (custom define outside the class State)

	// TODO: use hmap directly instead of this method
	map_t& map(idx x, idx y, idx z) { return hmap(x,y,z); }

	// macroscopic quantities
	hmacro_array_t hmacro;
	dmacro_array_t dmacro;
	cpumacro_array_t cpumacro;

	// distribution functions
	hlat_array_t hfs[DFMAX];
	dlat_array_t dfs[DFMAX];

	// MPI
	int rank = 0;
	int nproc = 1;

	// lattice sizes
	idx global_X, global_Y, global_Z;
	idx local_X, local_Y, local_Z;
	idx offset_X, offset_Y, offset_Z;

	int df_overlap_X() { return data.indexer.template getOverlap< 0 >(); }
	int df_overlap_Y() { return data.indexer.template getOverlap< 1 >(); }
	int df_overlap_Z() { return data.indexer.template getOverlap< 2 >(); }

#ifdef HAVE_MPI
	// synchronizers for dfs, macro and map
	TNL::Containers::DistributedNDArraySynchronizer< typename sync_array_t::ViewType, false, false > dreal_sync[27 + MACRO::N];
	TNL::Containers::DistributedNDArraySynchronizer< dmap_array_t, false, false > map_sync;

	// synchronization methods
	template< typename Array >
	void startDrealArraySynchronization(Array& array, int sync_offset);
	void synchronizeDFsDevice_start(uint8_t dftype = df_out);
	void synchronizeDFsDevice(uint8_t dftype = df_out);
	void synchronizeMacroDevice_start();
	void synchronizeMacroDevice();
	void synchronizeMapDevice();
#endif

	// input parameters: constant in time
	point_t physOrigin;			// spatial coordinates of the point at the center between (0,0,0) and (1,1,1) lattice sites
	real physDl; 				// spatial step (fixed throughout the computation)
	real physDt;				// temporal step (fixed or variable throughout the computation)
	real physFilDl;				// spatial step for filaments(should be fixed throught the computation)
	real physViscosity;			// physical viscosity of the fluid
	real physFluidDensity;			// physical (characteristic) density of the fluid (constant)
	real physFinalTime;			// default 1e10
	real physStartTime;			// used for ETA calculation only (default is 0)
	real physCharLength;			// characteristic length, default is physDl * (real)Y but you can specify that manually
	int iterations;			// number of lbm iterations

	bool terminate;			// flag for terminal error detection

//	real Re(real physvel) { return fabs(physvel) * physDl * (real)Y / physViscosity; } // TODO: change Y to charLength --- specify this explicitely
	real Re(real physvel) { return fabs(physvel) * physCharLength / physViscosity; } // TODO: change Y to charLength --- specify this explicitely
	dreal lbmViscosity() { return (dreal) (physDt / physDl / physDl * physViscosity); }
	real physTime() { return physDt*(real)iterations; }
	real lbm2physVelocity(real lbm_velocity) { return lbm_velocity / physDt * physDl; }
	real lbm2physForce(real lbm_force) { return lbm_force * physDl / physDt / physDt; }
	real phys2lbmVelocity(real phys_velocity)  { return phys_velocity * physDt / physDl; }
	real phys2lbmForce(real phys_force) { return phys_force / physDl * physDt * physDt; }
	dreal lbmInputDensity;

//	real physNormVelocity(idx gi) { return NORM( lbm2physVelocity(hvx[gi]), lbm2physVelocity(hvy[gi]), lbm2physVelocity(hvz[gi]) ); }
//	real physDensity(idx gi) { return hrho[gi]*physFluidDensity; }
//	real physNormVelocity(idx GX, idx GY, idx GZ) { return physNormVelocity(pos(GX,GY,GZ)); }
//	real physDensity(idx GX, idx GY, idx GZ) { return physDensity(pos(GX,GY,GZ)); }

	// getters for physical coordinates (note that here x,y,z are *global* lattice indices)
	point_t lbm2physPoint(idx x, idx y, idx z) { return physOrigin + (point_t(x, y, z) - 0.5) * physDl; }
	real lbm2physX(idx x) { return physOrigin.x() + (x-0.5) * physDl; }
	real lbm2physY(idx y) { return physOrigin.y() + (y-0.5) * physDl; }
	real lbm2physZ(idx z) { return physOrigin.z() + (z-0.5) * physDl; }

	// physical to lattice coordinates (but still real rather than idx, rounding can be done later)
	point_t phys2lbmPoint(point_t p) { return (p - physOrigin) / physDl + 0.5; }
	real phys2lbmX(real x) { return (x - physOrigin.x()) / physDl + 0.5; }
	real phys2lbmY(real y) { return (y - physOrigin.y()) / physDl + 0.5; }
	real phys2lbmZ(real z) { return (z - physOrigin.z()) / physDl + 0.5; }

	void resetForces() { resetForces(0,0,0);}
	void resetForces(real ifx, real ify, real ifz);
	void copyForcesToDevice();

	// all this is needed for IBM only and forcing
	dreal* hrho() { return &hmacro(MACRO::e_rho, offset_X, offset_Y, offset_Z); }
	dreal* hvx() { return &hmacro(MACRO::e_vx, offset_X, offset_Y, offset_Z); }
	dreal* hvy() { return &hmacro(MACRO::e_vy, offset_X, offset_Y, offset_Z); }
	dreal* hvz() { return &hmacro(MACRO::e_vz, offset_X, offset_Y, offset_Z); }
	dreal* hfx() { return &hmacro(MACRO::e_fx, offset_X, offset_Y, offset_Z); }
	dreal* hfy() { return &hmacro(MACRO::e_fy, offset_X, offset_Y, offset_Z); }
	dreal* hfz() { return &hmacro(MACRO::e_fz, offset_X, offset_Y, offset_Z); }

	dreal* drho() { return &dmacro(MACRO::e_rho, offset_X, offset_Y, offset_Z); }
	dreal* dvx() { return &dmacro(MACRO::e_vx, offset_X, offset_Y, offset_Z); }
	dreal* dvy() { return &dmacro(MACRO::e_vy, offset_X, offset_Y, offset_Z); }
	dreal* dvz() { return &dmacro(MACRO::e_vz, offset_X, offset_Y, offset_Z); }
	dreal* dfx() { return &dmacro(MACRO::e_fx, offset_X, offset_Y, offset_Z); }
	dreal* dfy() { return &dmacro(MACRO::e_fy, offset_X, offset_Y, offset_Z); }
	dreal* dfz() { return &dmacro(MACRO::e_fz, offset_X, offset_Y, offset_Z); }

	void copyMapToDevice();
	void copyMapToHost();
	void copyMacroToHost();
	void copyMacroToDevice();

	void copyDFsToHost(uint8_t dfty);
	void copyDFsToDevice(uint8_t dfty);
	void copyDFsToHost();
	void copyDFsToDevice();

	void computeCPUMacroFromLat();

	// Helpers for indexing - methods check if the given GLOBAL (multi)index is in the local range
	bool isLocalIndex(idx x, idx y, idx z);
	bool isLocalX(idx x);
	bool isLocalY(idx y);
	bool isLocalZ(idx z);

	// Global methods - use GLOBAL indices !!!
	void defineWall(idx x, idx y, idx z, bool value);
	void setBoundaryX(idx x, map_t value);
	void setBoundaryY(idx y, map_t value);
	void setBoundaryZ(idx z, map_t value);
	bool getWall(idx x, idx y, idx z);
	bool isFluid(idx x, idx y, idx z);

	void projectWall();
	void resetMap(map_t geo_type);
	void setEqLat(uint8_t dftype, idx x, idx y, idx z, real irho, real ivx, real ivy, real ivz); // prescribe rho,vx,vy,vz at a given point into "hfs" array
	void setEqLat(idx x, idx y, idx z, real irho, real ivx, real ivy, real ivz); // prescribe rho,vx,vy,vz at a given point into "hfs" array

	bool quit() { return terminate; }

	void allocateHostData();
	void allocateDeviceData();
	void updateKernelData();		// copy physical parameters to data structure accessible by the CUDA kernel

	LBM(idx iX, idx iY, idx iZ, real iphysViscosity, real iphysDl, real iphysDt, point_t iphysOrigin);
	~LBM();
};

#include "lbm.hpp"
