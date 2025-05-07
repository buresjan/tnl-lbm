#pragma once

#include <vector>
#include <string>

#include <sys/stat.h>
#include <sys/wait.h>

#include <TNL/Timer.h>
#include <fmt/core.h>
#include <adios2.h>

#include "../lbm_common/logging.h"
#include "../lbm_common/fileutils.h"
#include "lbm.h"
#include "vtk_writer.h"
#include "checkpoint.h"

// ibm: lagrangian filament/surface
#include "lagrange_3D.h"

// sparse box origin+length where to plot - can be just a part of the domain
template <typename IDX>
struct probe3Dcut
{
	IDX ox, oy, oz;	 // lower left front point
	IDX lx, ly, lz;	 // length
	IDX step;		 // 1: every voxel 2: every 3 voxels etc.
	std::string name;
	int cycle;
};

template <typename IDX>
struct probe2Dcut
{
	int type;  // 0=X, 1=Y, 2=Z
	std::string name;
	IDX position;  // x/y/z ... LBM units ... int
	int cycle;
};

template <typename IDX>
struct probe1Dcut
{
	int type;  // 0=X, 1=Y, 2=Z
	std::string name;
	IDX pos1;  // x/y/z
	IDX pos2;  // y/z
	int cycle;
};

template <typename REAL>
struct probe1Dlinecut
{
	std::string name;
	using point_t = TNL::Containers::StaticVector<3, REAL>;
	point_t from;  // physical units
	point_t to;	   // physical units
	int cycle;
};

// for print/stat/write/reset counters
template <typename REAL>
struct counter
{
	int count = 0;
	REAL period = -1.0;
	bool action(REAL time)
	{
		return period > 0 && time >= count * period;
	}
};

enum
{
	STAT_RESET,
	STAT2_RESET,
	PRINT,
	VTK1D,
	VTK2D,
	VTK3D,
	PROBE1,
	PROBE2,
	PROBE3,
	SAVESTATE,
	VTK3DCUT,
	MAX_COUNTER
};

template <typename NSE>
struct State
{
	using TRAITS = typename NSE::TRAITS;
	using MACRO = typename NSE::MACRO;
	using BLOCK_NSE = LBM_BLOCK<NSE>;
	using Lagrange3D = ::Lagrange3D<LBM<NSE>>;

	using map_t = typename TRAITS::map_t;
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using real = typename TRAITS::real;
	using point_t = typename TRAITS::point_t;
	using idx3d = typename TRAITS::idx3d;
	using lat_t = typename LBM<NSE>::lat_t;

	using T_PROBE3DCUT = probe3Dcut<idx>;
	using T_PROBE2DCUT = probe2Dcut<idx>;
	using T_PROBE1DCUT = probe1Dcut<idx>;
	using T_PROBE1DLINECUT = probe1Dlinecut<real>;
	using T_COUNTER = counter<real>;

	std::string id;

	adios2::ADIOS adios;
	CheckpointManager checkpoint;

	LBM<NSE> nse;

	std::vector<T_PROBE3DCUT> probe3Dvec;
	std::vector<T_PROBE2DCUT> probe2Dvec;
	std::vector<T_PROBE1DCUT> probe1Dvec;
	std::vector<T_PROBE1DLINECUT> probe1Dlinevec;

	// Immersed boundary method
	Lagrange3D ibm;

	void writeVTK_Points(const char* name, real time, int cycle);
	void writeVTK_Points(const char* name, real time, int cycle, const typename Lagrange3D::HLPVECTOR& hLL_lat);

	// how often to probe/print/write/stat
	T_COUNTER cnt[MAX_COUNTER];
	virtual void probe1() {}
	virtual void probe2() {}
	virtual void probe3() {}
	virtual void statReset() {}
	virtual void stat2Reset() {}

	// vtk export
	template <typename real1, typename real2>
	bool vtk_helper(const char* iid, real1 ivalue, int idofs, char* id, real2& value, int& dofs)  /// simplifies data output routine
	{
		sprintf(id, "%s", iid);
		dofs = idofs;
		value = ivalue;
		return true;
	}
	virtual void writeVTKs_2D();
	template <typename... ARGS>
	void add2Dcut_X(idx x, const char* fmt, ARGS... args);
	template <typename... ARGS>
	void add2Dcut_Y(idx y, const char* fmt, ARGS... args);
	template <typename... ARGS>
	void add2Dcut_Z(idx z, const char* fmt, ARGS... args);

	virtual void writeVTKs_3D();

	// 3D cuts
	virtual void writeVTKs_3Dcut();
	template <typename... ARGS>
	void add3Dcut(idx ox, idx oy, idx oz, idx lx, idx ly, idx lz, idx step, const char* fmt, ARGS... args);

	virtual void writeVTKs_1D();

	template <typename... ARGS>
	void add1Dcut(point_t from, point_t to, const char* fmt, ARGS... args);
	template <typename... ARGS>
	void add1Dcut_X(real y, real z, const char* fmt, ARGS... args);
	template <typename... ARGS>
	void add1Dcut_Y(real x, real z, const char* fmt, ARGS... args);
	template <typename... ARGS>
	void add1Dcut_Z(real x, real y, const char* fmt, ARGS... args);
	void write1Dcut(point_t from, point_t to, const std::string& fname);
	void write1Dcut_X(idx y, idx z, const std::string& fname);
	void write1Dcut_Y(idx x, idx z, const std::string& fname);
	void write1Dcut_Z(idx x, idx y, const std::string& fname);

	virtual bool outputData(const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs)
	{
		return false;
	}

	bool projectPNG_X(
		const std::string& filename,
		idx x0,
		bool rotate = false,
		bool mirror = false,
		bool flip = false,
		real amin = 0,
		real amax = 1,
		real bmin = 0,
		real bmax = 1
	);	// amin, amax, bmin, bmax ... used for cropping, see the code
	bool projectPNG_Y(
		const std::string& filename,
		idx y0,
		bool rotate = false,
		bool mirror = false,
		bool flip = false,
		real amin = 0,
		real amax = 1,
		real bmin = 0,
		real bmax = 1
	);	// amin, amax, bmin, bmax ... used for cropping, see the code
	bool projectPNG_Z(
		const std::string& filename,
		idx z0,
		bool rotate = false,
		bool mirror = false,
		bool flip = false,
		real amin = 0,
		real amax = 1,
		real bmin = 0,
		real bmax = 1
	);	// amin, amax, bmin, bmax ... used for cropping, see the code

	// simulation control
	virtual bool estimateMemoryDemands();  // called from State constructor
	virtual void reset();
	virtual void resetDFs();				  // called from State::reset -- sets the initial DFs on GPU
	virtual void setupBoundaries() {}		  // called from State::reset
	virtual void SimInit();					  // called from core.h -- before time loop
	virtual void updateKernelData();		  // called from core.h -- calls updateKernelData on all LBM blocks
	virtual void updateKernelVelocities() {}  // called from core.h -- setup current velocity profile for the Kernel
	virtual void SimUpdate();				  // called from core.h -- from the time loop, once per time step
	virtual void AfterSimUpdate();			  // called from core.h -- once before the time loop and then after each SimUpdate() call
	virtual void AfterSimFinished();		  // called from core.h -- at the end of the simulation (after the time loop)
	virtual void computeBeforeLBMKernel() {}  // called from core.h just before the main LBMKernel -- extra kernels e.g. for the non-Newtonian model
	virtual void computeAfterLBMKernel() {}	  // called from core.h after the main LBMKernel -- extra kernels e.g. for the coupled LBM-MHFEM solver
	virtual void copyAllToDevice();			  // called from SimInit -- copy the initial state to the GPU
	virtual void copyAllToHost();			  // called from core.h -- inside the time loop before saving state

	/* Checks if the solver can compute the requested simulation. The following
	 * factors are considered:
	 *
	 * 1. If there is another instance of the solver running on the same
	 *    `results_{id}` directory, this function returns `false`. Otherwise,
	 * 	  this instance locks the `results_{id}` directory using `flock`.
	 * 2. If the "loadstate" flag exists in the `results_{id}`, this function
	 *	  returns `true`.
	 * 3. If either "finished" or "terminated" flag exists in the `results_{id}`,
	 *	  this function return `false`.
	 * 4. Otherwise, this function returns `true`.
	 *
	 * All conditions are checked by rank 0 and broadcast to all other ranks.
	 * Hence, the result is consistent across all MPI ranks.
	 *
	 * The "loadstate" flag is checked in `State::SimInit` and created/deleted
	 * in the `execute` function in `core.h`. The "finished" and "terminated"
	 * flags are created in the `execute` function in `core.h`.
	 */
	bool canCompute();
	int lock_fd = -1;

	void flagCreate(const char* flagname);
	void flagDelete(const char* flagname);
	bool flagExists(const char* flagname);

	// checkpoint data in the main State class
	void checkpointState(adios2::Mode mode);
	// checkpoint additional data in subclasses of State
	virtual void checkpointStateLocal(adios2::Mode mode) {}
	// called periodically through cnt[SAVESTATE]
	void saveState();
	void loadState();

	// timers for walltime, GLUPS and ETA calculations
	TNL::Timer timer_total;
	long wallTime = -1;	 // maximum allowed wallTime in seconds, use negative value to disable wall time check
	bool wallTimeReached();
	double getWallTime(
		bool collective = false
	);	// collective: must be true when called by all MPI ranks and false otherwise (e.g. when called only by rank 0)

	// helpers for incremental GLUPS calculation
	int glups_prev_iterations = 0;
	double glups_prev_time = 0;

	// timers for profiling
	TNL::Timer timer_SimInit, timer_SimUpdate, timer_AfterSimUpdate, timer_compute, timer_compute_overlaps, timer_wait_communication,
		timer_wait_computation;

	// constructors
	template <typename... ARGS>
	State(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, ARGS&&... args)
	: id(id),
#ifdef HAVE_MPI
	  adios(communicator),
#else
	  adios(),
#endif
	  checkpoint(adios),
	  nse(communicator, lat, std::forward<ARGS>(args)...),
	  ibm(nse, id)
	{
		// try to lock the results directory
		if (nse.rank == 0) {
			const std::string dir = fmt::format("results_{}", id);
			mkdir(dir.c_str(), 0777);
			const std::string lock_filename = fmt::format("results_{}/lock", id);
			lock_fd = tryLockFile(lock_filename.c_str());
		}

		// let all ranks know if we have a lock
		bool have_lock = lock_fd >= 0;
		TNL::MPI::Bcast(&have_lock, 1, 0, communicator);

		// initialize default spdlog logger (check the lock to avoid writing
		// to log files opened by another instance)
		if (have_lock)
			init_logging(id, communicator);

		bool local_estimate = estimateMemoryDemands();
		bool global_result = TNL::MPI::reduce(local_estimate, MPI_LAND, communicator);
		if (! local_estimate)
			spdlog::error("Not enough memory available (CPU or GPU). [disable this check in lbm3d/state.h -> State constructor]");
		if (! global_result)
			throw std::runtime_error("Not enough memory available (CPU or GPU).");

		// allocate host data -- after the estimate
		nse.allocateHostData();

		// initial time of current simulation
		timer_total.start();
	}

	virtual ~State()
	{
		deinit_logging();
		releaseLock(lock_fd);
	}
};

#include "state.hpp"
