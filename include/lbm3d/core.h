#pragma once

#include <omp.h>

#include "defs.h"
#include "state.h"


template < typename NSE >
#ifdef TODO
CUDA_HOSTDEV
void LBMKernel(
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z,
	typename NSE::DATA SD,
	short int rank,
	short int nproc
)
#else
#ifdef USE_CUDA
//__launch_bounds__(32, 16)
__global__ void cudaLBMKernel(
	typename NSE::DATA SD,
	short int rank,
	short int nproc,
	typename NSE::TRAITS::idx offset_x
)
#else
CUDA_HOSTDEV
void LBMKernel(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z,
	short int rank,
	short int nproc
)
#endif
#endif
{
	using dreal = typename NSE::TRAITS::dreal;
	using idx = typename NSE::TRAITS::idx;
	using map_t = typename NSE::TRAITS::map_t;

#ifndef TODO
	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset_x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
#endif
	map_t gi_map = SD.map(x, y, z);

	typename NSE::KernelStruct<dreal> KS;

	// copy quantities
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (NSE::BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
//		xp = (SD.overlap_right) ? x+1 : MIN(x+1, SD.X-1);
//		xm = (SD.overlap_left) ? x-1 : MAX(x-1,0);
		xp = (rank != nproc-1) ? x+1 : MIN(x+1, SD.X()-1);
		xm = (rank != 0) ? x-1 : MAX(x-1,0);
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}

	// optional computation of the forcing term (e.g. for the non-Newtonian model)
	NSE::MACRO::template computeForcing<NSE::BC>(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

	// Streaming
	if (NSE::BC::isStreaming(gi_map))
		NSE::STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

	// compute Density & Velocity
	if (NSE::BC::isComputeDensityAndVelocity(gi_map))
		NSE::COLL::computeDensityAndVelocity(KS);


	// boundary conditions
	if (NSE::BC::BC(SD,KS,gi_map,xm,x,xp,ym,y,yp,zm,z,zp)==false)
		NSE::COLL::collision(KS);

	NSE::COLL::copyKS2DFout(SD,KS,x,y,z);
	NSE::MACRO::outputMacro(SD, KS, x, y, z);
}


// wrapper: work on Macro before LBMKernel
template < typename NSE >
#ifdef USE_CUDA
__global__ void cudaMacroWorker(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx offset_x
)
#else
void MacroWorker(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z
)
#endif
{
	using idx = typename NSE::TRAITS::idx;
	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset_x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
	NSE::MACRO::template kernelWorker<NSE>(SD,x,y,z);
}

// initial condition --> hmacro on CPU
template < typename NSE >
void LBMKernelInit(
	typename NSE::DATA& SD,
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z
)
{
	using map_t = typename NSE::TRAITS::map_t;
	using idx = typename NSE::TRAITS::idx;
	using dreal = typename NSE::TRAITS::dreal;

	map_t gi_map = SD.map(x, y, z);

	typename NSE::KernelStruct<dreal> KS;
	for (int i=0;i<27;i++) KS.f[i] = SD.df(df_cur, i, x, y, z);

	// copy quantities
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);

	KS.fx=0.;
	KS.fy=0.;
	KS.fz=0.;

	// compute Density & Velocity
	if (NSE::BC::isComputeDensityAndVelocity(gi_map))
		NSE::COLL::computeDensityAndVelocity(KS);

	NSE::MACRO::outputMacro(SD, KS, x, y, z);
}


//template<typename L, typename M, typename LBM_DATA>
template < typename NSE >
#ifdef USE_CUDA
__global__ void cudaLBMComputeVelocitiesStar(
	typename NSE::DATA SD,
	short int rank,
	short int nproc
)
#else
void LBMComputeVelocitiesStar(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z,
	short int rank,
	short int nproc
)
#endif
{
	using dreal = typename NSE::TRAITS::dreal;
	using idx = typename NSE::TRAITS::idx;
	using map_t = typename NSE::TRAITS::map_t;

	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
	map_t gi_map = SD.map(x, y, z);

	typename NSE::KernelStruct<dreal> KS;

	// copy quantities
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (NSE::BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
//		xp = (SD.overlap_right) ? x+1 : MIN(x+1, SD.X-1);
//		xm = (SD.overlap_left) ? x-1 : MAX(x-1,0);
		xp = (rank != nproc-1) ? x+1 : MIN(x+1, SD.X()-1);
		xm = (rank != 0) ? x-1 : MAX(x-1,0);
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}

	// Streaming
	if (NSE::BC::isStreaming(gi_map))
		NSE::STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

	KS.fx=0;
	KS.fy=0;
	KS.fz=0;

	// compute Density & Velocity
	if (NSE::BC::isComputeDensityAndVelocity(gi_map))
		NSE::COLL::computeDensityAndVelocity(KS);

	NSE::MACRO::outputMacro(SD, KS, x, y, z);
}

//template<typename L, typename M, typename LBM_DATA>
template < typename NSE >
#ifdef USE_CUDA
__global__ void cudaLBMComputeVelocitiesStarAndZeroForce(
	typename NSE::DATA SD,
	short int rank,
	short int nproc
)
#else
void LBMComputeVelocitiesStarAndZeroForce(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z,
	short int rank,
	short int nproc
)
#endif
{
	using dreal = typename NSE::TRAITS::dreal;
	using idx = typename NSE::TRAITS::idx;
	using map_t = typename NSE::TRAITS::map_t;

	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
	map_t gi_map = SD.map(x, y, z);

	typename NSE::KernelStruct<dreal> KS;

	// copy quantities
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (NSE::BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
//		xp = (SD.overlap_right) ? x+1 : MIN(x+1, SD.X-1);
//		xm = (SD.overlap_left) ? x-1 : MAX(x-1,0);
		xp = (rank != nproc-1) ? x+1 : MIN(x+1, SD.X()-1);
		xm = (rank != 0) ? x-1 : MAX(x-1,0);
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}

	// Streaming
	if (NSE::BC::isStreaming(gi_map))
		NSE::STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

	KS.fx=0;
	KS.fy=0;
	KS.fz=0;

	// compute Density & Velocity
	if (NSE::BC::isComputeDensityAndVelocity(gi_map))
		NSE::COLL::computeDensityAndVelocity(KS);

	NSE::MACRO::outputMacro(SD, KS, x, y, z);
	// reset forces
	NSE::MACRO::zeroForces(SD, x, y, z);
}


template < typename STATE >
void SimUpdate(STATE& state, typename STATE::T_LBM& lbm)
{
	using LBM_TYPE = typename STATE::LBM_TYPE;
	using TRAITS = typename LBM_TYPE::TRAITS;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	// debug
	if (lbm.data.lbmViscosity == 0) {
		state.log("error: LBM viscosity is 0");
		state.lbm.terminate = true;
		return;
	}

	#ifdef USE_CUDA
		checkCudaDevice;
		dim3 blockSize(1, lbm.block_size, 1);
		dim3 gridSize(lbm.local_X, lbm.local_Y/lbm.block_size, lbm.local_Z);

		// check for PEBKAC problem existing between keyboard and chair
		if (gridSize.y * lbm.block_size != lbm.local_Y) {
			state.log("error: lbm.local_Y (which is %d) is not aligned to a multiple of the block size (which is %d)", lbm.local_Y, lbm.block_size);
			state.lbm.terminate = true;
			return;
		}
	#endif

	// flags
	bool doComputeVelocitiesStar=false;
	bool doCopyQuantitiesStarToHost=false;
	bool doZeroForceOnDevice=false;
	bool doZeroForceOnHost=false;
	bool doComputeLagrangePhysics=false;
	bool doCopyForceToDevice=false;

	// determine global flags
	// NOTE: all Lagrangian points are assumed to be on the first GPU
	// TODO
//	if (lbm.data.rank == 0 && state.FF.size() > 0)
	if (state.FF.size() > 0)
	{
		doComputeLagrangePhysics=true;
		for (int i=0;i<state.FF.size();i++)
		if (state.FF[i].implicitWuShuForcing)
		{
			doComputeVelocitiesStar=true;
			switch (state.FF[i].ws_compute)
			{
				case ws_computeCPU:
				case ws_computeCPU_TNL:
					doCopyQuantitiesStarToHost=true;
					doZeroForceOnHost=true;
					doCopyForceToDevice=true;
					break;
				case ws_computeGPU_TNL:
				case ws_computeHybrid_TNL:
				case ws_computeHybrid_TNL_zerocopy:
				case ws_computeGPU_CUSPARSE:
				case ws_computeHybrid_CUSPARSE:
					doZeroForceOnDevice=true;
					break;
			}
		}
	}


	if (doComputeVelocitiesStar)
	{
		#ifdef USE_CUDA
			if (doZeroForceOnDevice)
				cudaLBMComputeVelocitiesStarAndZeroForce< LBM_TYPE ><<<gridSize, blockSize>>>(lbm.data, lbm.rank, lbm.nproc);
			else
				cudaLBMComputeVelocitiesStar< LBM_TYPE ><<<gridSize, blockSize>>>(lbm.data, lbm.rank, lbm.nproc);
			checkCudaDevice;
		#else
			#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < lbm.local_X; x++)
			for (idx z = 0; z < lbm.local_Z; z++)
			for (idx y = 0; y < lbm.local_Y; y++)
			if (doZeroForceOnDevice)
				LBMComputeVelocitiesStarAndZeroForce< LBM_TYPE >(lbm.data, x, y, z);
			else
				LBMComputeVelocitiesStar< LBM_TYPE >(lbm.data, x, y, z);
		#endif
		if (doCopyQuantitiesStarToHost)
		{
			lbm.copyMacroToHost();
		}
	}


	// reset lattice force vectors dfx and dfy
	if (doZeroForceOnHost)
	{
		lbm.resetForces();
	}

//	state.log("core.h state.computeAllLagrangeForces() start");
	if (doComputeLagrangePhysics)
	{
		state.computeAllLagrangeForces();
	}
//	state.log("core.h state.computeAllLagrangeForces() done");

	if (doCopyForceToDevice)
	{
		lbm.copyForcesToDevice();
	}


	// call hook method (used e.g. for extra kernels in the non-Newtonian model)
	state.computeBeforeLBMKernel();


#ifdef TODO
	if (LBM_TYPE::MACRO::use_kernelWorker) cudaMacroWorker< LBM_TYPE ><<<gridSize, blockSize>>>(lbm.data);
	TNL::ParallelFor3D< TNL::Devices::Cuda >::exec(
			(idx) 0, (idx) 0, (idx) 0,
			lbm.local_X, lbm.local_Y, lbm.local_Z,
			LBMKernel< LBM_TYPE >,
			lbm.data, lbm.rank, lbm.nproc
		);
	// TODO: overlap computation with synchronization, just like below
	lbm.synchronizeDFsDevice(df_out);
#else
	#ifdef USE_CUDA
		#ifdef HAVE_MPI
		if (lbm.nproc == 1)
		{
		#endif
			if (LBM_TYPE::MACRO::use_kernelWorker) cudaMacroWorker< LBM_TYPE ><<<gridSize, blockSize>>>(lbm.data, (idx) 0);
			cudaLBMKernel< LBM_TYPE ><<<gridSize, blockSize>>>(lbm.data, lbm.rank, lbm.nproc, (idx) 0);
			cudaDeviceSynchronize();
			checkCudaDevice;
			// copying of overlaps is not necessary for nproc == 1 (nproc is checked in streaming as well)
		#ifdef HAVE_MPI
		}
		else
		{
			dim3 gridSizeForBoundary(lbm.df_overlap_X(), lbm.local_Y/lbm.block_size, lbm.local_Z);
			dim3 gridSizeForInternal(lbm.local_X - 2*lbm.df_overlap_X(), lbm.local_Y/lbm.block_size, lbm.local_Z);

			// run cudaMacroWorker on the boundaries (NOTE: 1D distribution is assumed)
			if (LBM_TYPE::MACRO::use_kernelWorker)
			{
				// NOTE: we assume that the overlaps for DFs and macro are equal
				cudaMacroWorker< LBM_TYPE ><<<gridSizeForBoundary, blockSize, 0, cuda_streams[0]>>>(lbm.data, (idx) 0);
				cudaMacroWorker< LBM_TYPE ><<<gridSizeForBoundary, blockSize, 0, cuda_streams[1]>>>(lbm.data, lbm.local_X - lbm.df_overlap_X());
			}

			// compute on boundaries (NOTE: 1D distribution is assumed)
			cudaLBMKernel< LBM_TYPE ><<<gridSizeForBoundary, blockSize, 0, cuda_streams[0]>>>(lbm.data, lbm.rank, lbm.nproc, (idx) 0);
			cudaLBMKernel< LBM_TYPE ><<<gridSizeForBoundary, blockSize, 0, cuda_streams[1]>>>(lbm.data, lbm.rank, lbm.nproc, lbm.local_X - lbm.df_overlap_X());

			// run cudaMacroWorker on internal lattice sites
			if (LBM_TYPE::MACRO::use_kernelWorker)
			{
				cudaMacroWorker< LBM_TYPE ><<<gridSizeForBoundary, blockSize, 0, cuda_streams[2]>>>(lbm.data, lbm.df_overlap_X());
			}

			// compute on internal lattice sites
			cudaLBMKernel< LBM_TYPE ><<<gridSizeForInternal, blockSize, 0, cuda_streams[2]>>>(lbm.data, lbm.rank, lbm.nproc, lbm.df_overlap_X());

			// wait for the computations on boundaries to finish
			cudaStreamSynchronize(cuda_streams[0]);
			cudaStreamSynchronize(cuda_streams[1]);

			// start communication of the latest DFs and dmacro on overlaps
			lbm.synchronizeDFsDevice_start(df_out);
			if (LBM_TYPE::MACRO::use_syncMacro)
				lbm.synchronizeMacroDevice_start();

			// wait for the communication to finish
			// (it is important to do this before waiting for the computation, otherwise MPI won't progress)
			for (int i = 0; i < 27; i++)
				lbm.dreal_sync[i].wait();
			if (LBM_TYPE::MACRO::use_syncMacro)
				for (int i = 0; i < LBM_TYPE::MACRO::N; i++)
					lbm.dreal_sync[27 + i].wait();

			// wait for the computation on the interior to finish
			cudaStreamSynchronize(cuda_streams[2]);

			// synchronize the whole GPU and check errors
			cudaDeviceSynchronize();
			checkCudaDevice;
		}
		#endif
	#else
		#pragma omp parallel for schedule(static) collapse(2)
		for (idx x=0; x<lbm.local_X; x++)
		for (idx z=0; z<lbm.local_Z; z++)
		for (idx y=0; y<lbm.local_Y; y++)
		{
			if (LBM_TYPE::MACRO::use_kernelWorker)
				MacroWorker< LBM_TYPE >(lbm.data, x, y, z);
			LBMKernel< LBM_TYPE >(lbm.data, x, y, z, lbm.rank, lbm.nproc);
		}
		#ifdef HAVE_MPI
		// TODO: overlap computation with synchronization, just like above
		lbm.synchronizeDFsDevice(df_out);
		#endif
	#endif
#endif

	lbm.iterations++;

	bool doit=false;
	for (int c=0;c<MAX_COUNTER;c++) if (c!=PRINT && c!=SAVESTATE) if (state.cnt[c].action(lbm.physTime())) doit = true;
	if (doit)
	{
		lbm.copyMacroToHost();
		if (LBM_TYPE::CPU_MACRO::N>0) lbm.copyDFsToHost(df_out); // to be able to compute rho, vx, vy, vz etc... based on DFs on CPU to save GPU memory FIXME may not work with ESOTWIST
		#ifdef USE_CUDA
		checkCudaDevice;
		#endif
	}
}

template < typename STATE >
void AfterSimUpdate(STATE& state, timespec& t1, timespec& t2, int& lbmPrevIterations)
{
	// call hook method (used e.g. for the coupled LBM-MHFEM solver)
	state.computeAfterLBMKernel();

	typename STATE::T_LBM& lbm = state.lbm;

	#ifdef USE_CUDA
	// synchronization is not necessary for correctness, only to get correct MLUPS
	bool doit=false;
	for (int c=0;c<MAX_COUNTER;c++) if (state.cnt[c].action(lbm.physTime())) doit = true;
	if (doit)
	{
		cudaDeviceSynchronize();
		checkCudaDevice;
		TNL::MPI::Barrier();
	}
	#endif

	bool write_info=false;

	if (state.cnt[VTK1D].action(lbm.physTime()) ||
	    state.cnt[VTK2D].action(lbm.physTime()) ||
	    state.cnt[VTK3D].action(lbm.physTime()) ||
	    state.cnt[VTK3DCUT].action(lbm.physTime()) ||
	    state.cnt[PROBE1].action(lbm.physTime()) ||
	    state.cnt[PROBE2].action(lbm.physTime()) ||
	    state.cnt[PROBE3].action(lbm.physTime())
	    )
	{
		// common copy
		state.lbm.computeCPUMacroFromLat();
		// probe1
		if (state.cnt[PROBE1].action(lbm.physTime()))
		{
			state.probe1();
			state.cnt[PROBE1].count++;
		}
		// probe2
		if (state.cnt[PROBE2].action(lbm.physTime()))
		{
			state.probe2();
			state.cnt[PROBE2].count++;
		}
		// probe3
		if (state.cnt[PROBE3].action(lbm.physTime()))
		{
			state.probe3();
			state.cnt[PROBE3].count++;
		}
		// 3D VTK
		if (state.cnt[VTK3D].action(lbm.physTime()))
		{
			state.writeVTKs_3D();
			state.cnt[VTK3D].count++;
		}
		// 3D VTK CUT
		if (state.cnt[VTK3DCUT].action(lbm.physTime()))
		{
			state.writeVTKs_3Dcut();
			state.cnt[VTK3DCUT].count++;
		}
		// 2D VTK
		if (state.cnt[VTK2D].action(lbm.physTime()))
		{
			state.writeVTKs_2D();
			state.cnt[VTK2D].count++;
		}
		// 1D VTK
		if (state.cnt[VTK1D].action(lbm.physTime()))
		{
			state.writeVTKs_1D();
			state.cnt[VTK1D].count++;
		}
		write_info = true;
	}

	if (state.cnt[PRINT].action(lbm.physTime()))
	{
		write_info = true;
		state.cnt[PRINT].count++;
	}

	// statReset is called after all probes and VTK output
	// copy macro from host to device after reset
	if (state.cnt[STAT_RESET].action(lbm.physTime()))
	{
		state.statReset();
		lbm.copyMacroToDevice();
		state.cnt[STAT_RESET].count++;
	}
	if (state.cnt[STAT2_RESET].action(lbm.physTime()))
	{
		state.stat2Reset();
		lbm.copyMacroToDevice();
		state.cnt[STAT2_RESET].count++;
	}

	if (lbm.rank == 0)	// only the first process writes MLUPS
	if (lbm.iterations > 1)
//	if (write_info || (state.printIter > 0 && lbm.iterations % state.printIter == 0) )
	if (write_info)
	{
		clock_gettime(CLOCK_REALTIME, &t2);
		long timediff = (t2.tv_sec - t1.tv_sec) * 1000000000 + (t2.tv_nsec - t1.tv_nsec);
		// to avoid numerical errors - split LUPS computation in two parts
		double LUPS = lbm.iterations - lbmPrevIterations;
		LUPS *= lbm.global_X * lbm.global_Y * lbm.global_Z * 1000000000.0 / timediff;
		write_info = true;
		clock_gettime(CLOCK_REALTIME, &t1);
		lbmPrevIterations=lbm.iterations;

		// simple estimate of time of accomplishment
		double ETA = state.getWallTime() * (lbm.physFinalTime - lbm.physTime()) / (lbm.physTime() - lbm.physStartTime);

		if (state.verbosity>0)
		{
			state.log("MLUPS=%.1f iter=%d t=%1.3fs dt=%1.2e lbmVisc=%1.2e WT=%.0fs ETA=%.0fs",
				LUPS * 1e-6,
				lbm.iterations,
				lbm.physTime(),
				lbm.physDt,
				lbm.lbmViscosity(),
				state.getWallTime(),
				ETA
			);
		}
	}
}


template < typename STATE >
void execute(STATE& state)
{
	using TRAITS = typename STATE::TRAITS;
	using LBM_TYPE = typename STATE::LBM_TYPE;

	using idx = typename TRAITS::idx;

	typename STATE::T_LBM& lbm = state.lbm; // only a reference to state.lbm is stored to lbm

	state.log("MPI info: rank=%d, nproc=%d, local_X=%d, offset_X=%d", lbm.rank, lbm.nproc, lbm.local_X, lbm.offset_X);

	// reset counters
	for (int c=0;c<MAX_COUNTER;c++) state.cnt[c].count=0;
	state.cnt[SAVESTATE].count = 1;  // skip initial save of state
	lbm.iterations=0;
	int lbmPrevIterations=0;

	struct timespec t1, t2;
	clock_gettime(CLOCK_REALTIME, &t1);

#ifdef HAVE_MPI
	// get the range of stream priorities for current GPU
	int priority_high, priority_low;
	cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
	// high-priority streams for boundaries
	cudaStreamCreateWithPriority(&cuda_streams[0], cudaStreamNonBlocking, priority_high);
	cudaStreamCreateWithPriority(&cuda_streams[1], cudaStreamNonBlocking, priority_high);
	// low-priority stream for the interior
	cudaStreamCreateWithPriority(&cuda_streams[2], cudaStreamNonBlocking, priority_low);
#endif

	// check for loadState
//	if(state.flagExists("current_state/df_0"))
	if(state.flagExists("loadstate"))
	{
		state.loadState(); // load saved state into CPU memory
		state.lbm.physStartTime = state.lbm.physTime();
	}
	else
	{
		// setup map and DFs in CPU memory
		state.reset();

		// create LBM_DATA with host pointers
		typename LBM_TYPE::DATA SD;
		for (uint8_t dfty=0;dfty<DFMAX;dfty++)
			SD.dfs[dfty] = lbm.hfs[dfty].getData();
		#ifdef HAVE_MPI
		SD.indexer = lbm.hmap.getLocalView().getIndexer();
		#else
		SD.indexer = lbm.hmap.getIndexer();
		#endif
		SD.XYZ = SD.indexer.getStorageSize();
		SD.dmap = lbm.hmap.getData();
		SD.dmacro = lbm.hmacro.getData();

		// initialize macroscopic quantities on CPU
		#pragma omp parallel for schedule(static) collapse(2)
		for (idx x = 0; x < lbm.local_X; x++)
		for (idx z = 0; z < lbm.local_Z; z++)
		for (idx y = 0; y < lbm.local_Y; y++)
			LBMKernelInit<LBM_TYPE>(SD, x, y, z);
	}

	state.log("\nSTART: simulation LBM:%s dimensions %d x %d x %d lbmVisc %e physDl %e physDt %e", LBM_TYPE::COLL::id, lbm.global_X, lbm.global_Y, lbm.global_Z, lbm.lbmViscosity(), lbm.physDl, lbm.physDt);

	lbm.allocateDeviceData();
	lbm.copyMapToDevice();
	lbm.copyDFsToDevice();
	lbm.copyMacroToDevice();  // important when a state has been loaded

#ifdef HAVE_MPI
	// synchronize overlaps with MPI (initial synchronization can be synchronous)
	lbm.synchronizeMapDevice();
	lbm.synchronizeDFsDevice(df_cur);
	if (STATE::MACRO::use_syncMacro)
		lbm.synchronizeMacroDevice();
#endif

	// make snapshot for the initial condition
	AfterSimUpdate(state, t1, t2, lbmPrevIterations);

	bool quit = false;
	while (!quit)
	{
		// update kernel data (viscosity, swap df1 and df2)
		lbm.updateKernelData();
		state.updateKernelVelocities();

		SimUpdate(state, lbm);

		// post-processing: snapshots etc.
		AfterSimUpdate(state, t1, t2, lbmPrevIterations);

		// check wall time
		// (Note that state.wallTimeReached() must be called exactly once per iteration!)
		if (state.wallTimeReached())
		{
			// copy all quantities to CPU from lbm
			lbm.copyMapToHost();
			lbm.copyDFsToHost();
			lbm.copyMacroToHost();

			state.log("maximum wall time reached");
			// copy data to CPU (if needed)
			state.saveState(true);
			quit = true;
		}
		// check savestate
//		else if (state.cnt[SAVESTATE].action(lbm.physTime()))
		else if (state.cnt[SAVESTATE].action(state.getWallTime(true)))
		{
			// copy all quantities to CPU from lbm
			lbm.copyMapToHost();
			lbm.copyDFsToHost();
			lbm.copyMacroToHost();

			state.saveState();
			state.cnt[SAVESTATE].count++;
		}

		// check final time
		if (lbm.physTime() > lbm.physFinalTime)
		{
			state.log("physFinalTime reached");
			quit = true;
		}

		// handle termination locally
		if (lbm.quit())
		{
			state.log("terminate flag triggered");
			quit = true;
		}

		// distribute quit among all MPI processes
		bool local_quit = quit;
		TNL::MPI::Allreduce(&local_quit, &quit, 1, MPI_LOR, TNL::MPI::AllGroup());
	}

#ifdef HAVE_MPI
	for (int i = 0; i < 3; i++)
		cudaStreamDestroy(cuda_streams[i]);
#endif
}
