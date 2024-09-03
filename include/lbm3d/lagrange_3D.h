#pragma once

#include <TNL/Devices/Host.h>
#include <vector>
#include <algorithm>
#include <string>

#include <math.h>

#include "defs.h"
#include "lbm.h"

#include <fmt/core.h>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Solvers/Linear/CG.h>
#include <TNL/Solvers/Linear/Preconditioners/Diagonal.h>
#include <TNL/Solvers/Linear/Preconditioners/ILU0.h>
#include <TNL/Allocators/CudaHost.h>
#include <TNL/Containers/Vector.h>

template< typename Device, typename Index, typename IndexAlocator >
using SlicedEllpackSegments = TNL::Algorithms::Segments::SlicedEllpack< Device, Index, IndexAlocator >;
template< typename Real, typename Device, typename Index >
using SlicedEllpack = TNL::Matrices::SparseMatrix< Real,
													Device,
													Index,
													TNL::Matrices::GeneralMatrix,
													SlicedEllpackSegments
													>;

enum {
	ws_computeCPU_TNL,
	ws_computeGPU_TNL,
	ws_computeHybrid_TNL,
	ws_computeHybrid_TNL_zerocopy
};

enum class DiracMethod //Enum for deciding which method is used for calculation
{
	MODIFIED = 0,
	ORIGINAL = 1,
};

template< typename LBM >
struct Lagrange3D
{
	using TRAITS = typename LBM::TRAITS;
	using MACRO = typename LBM::MACRO;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using real = typename TRAITS::real;
	// TRAITS::point_t is based on real, we want dreal
	using point_t = TNL::Containers::StaticVector< 3, dreal >;

	LBM &lbm;

//	using hVector = TNL::Containers::Vector< real, TNL::Devices::Host, idx, TNL::Allocators::CudaHost<real> >;
	using hVector = TNL::Containers::Vector< dreal, TNL::Devices::Host, idx, TNL::Allocators::CudaHost<dreal> >;
	using dVector = TNL::Containers::Vector< dreal, TNL::Devices::Cuda, idx >;
//	using hEllpack = SlicedEllpack< real, TNL::Devices::Host, idx >;
	using hEllpack = SlicedEllpack< dreal, TNL::Devices::Host, idx >;
	using dEllpack = SlicedEllpack< dreal, TNL::Devices::Cuda, idx >;
	using hEllpackPtr = std::shared_ptr< hEllpack >;
	using dEllpackPtr = std::shared_ptr< dEllpack >;

	using hVectorView = TNL::Containers::VectorView< dreal, TNL::Devices::Host, idx >;
	using dVectorView = TNL::Containers::VectorView< dreal, TNL::Devices::Cuda, idx >;

	// ws_ using sparse matrices
	hEllpackPtr ws_tnl_hA;
	hEllpack ws_tnl_hM; // matrix realizing projection of u* to lagrange desc.
	hEllpack ws_tnl_hMT; // matrix realizing projection of uB from lagrange desc. to Euler desc. .... basially transpose of M
	hVector ws_tnl_hx[3], ws_tnl_hb[3];

	typename hEllpack::RowCapacitiesType hM_row_capacities;
	typename hEllpack::RowCapacitiesType hA_row_capacities;

	using hSolver = TNL::Solvers::Linear::CG< hEllpack >;
	using hPreconditioner = TNL::Solvers::Linear::Preconditioners::Diagonal< hEllpack >;
//	using hPreconditioner = TNL::Solvers::Linear::Preconditioners::ILU0< hEllpack >;
	hSolver ws_tnl_hsolver;
	typename std::shared_ptr< hPreconditioner > ws_tnl_hprecond;

	#ifdef USE_CUDA
	dEllpackPtr ws_tnl_dA; // square matrix A
	dEllpack ws_tnl_dM; // matrix realizing projection of u* to lagrange desc.
	dEllpack ws_tnl_dMT; // matrix realizing projection of uB from lagrange desc. to Euler desc. .... basially transpose of M
	dVector ws_tnl_dx[3], ws_tnl_db[3];
	// for ws_computeHybrid_TNL_zerocopy
	using hVectorPinned = TNL::Containers::Vector< dreal, TNL::Devices::Host, idx, TNL::Allocators::CudaHost<dreal> >;
	hVectorPinned ws_tnl_hxz[3], ws_tnl_hbz[3];

	typename dEllpack::RowCapacitiesType dM_row_capacities;
	typename dEllpack::RowCapacitiesType dA_row_capacities;

	using dSolver = TNL::Solvers::Linear::CG< dEllpack >;
	using dPreconditioner = TNL::Solvers::Linear::Preconditioners::Diagonal< dEllpack >;
//	using dPreconditioner = TNL::Solvers::Linear::Preconditioners::ILU0< dEllpack >;
	dSolver ws_tnl_dsolver;
	typename std::shared_ptr< dPreconditioner > ws_tnl_dprecond;
	#endif

	DiracMethod methodVariant=DiracMethod::MODIFIED;		// use continuous ws_ trick with 2 dirac functions
	int ws_compute=ws_computeGPU_TNL;		// ws_computeCPU, ws_computeGPU, ws_computeHybrid

	bool allocated = false;
	bool constructed = false;

	void convertLagrangianPoints();
	void allocateMatricesCPU();
	void allocateMatricesGPU();
	void constructMatricesCPU();
	void constructMatricesGPU();
	void computeForces(real time);

	real maxDist;			// maximal distance between points
	real minDist;			// minimal distance between points
	int diracDeltaTypeEL=2;
	int diracDeltaTypeLL=1;

	std::vector<point_t> LL;

	using DLPVECTOR = TNL::Containers::Vector<point_t, TNL::Devices::Cuda, idx>;
	using HLPVECTOR = TNL::Containers::Vector<point_t, TNL::Devices::Host, idx>;

	HLPVECTOR hLL_lat;
	DLPVECTOR dLL_lat;

	// accessors for macroscopic quantities as a 1D vector
	hVectorView hmacroVector(int macro_idx);  // macro_idx must be less than MACRO::N
	dVectorView dmacroVector(int macro_idx);  // macro_idx must be less than MACRO::N

	real computeMinDist();		// computes min and max distance between neinghboring nodes
	real computeMaxDistFromMinDist(real mindist);		// computes min and max distance between neighboring nodes
//	void integrateForce(real &Fx, real &Fy, real &Fz, real surface_element_size) { printf("integrateForce not implemented yet."); }

	// flag to enable matrix output to .mtx files
	bool mtx_output = false;

	// constructors
	Lagrange3D(LBM &inputLBM, const std::string& state_id);

	// disable copy-constructor and copy-assignment, leave only move-constructor and move-assignment
	// (because this class has a "LBM &lbm;" member)
	Lagrange3D(const Lagrange3D&) = delete;
	Lagrange3D(Lagrange3D&&) = default;
	Lagrange3D& operator=(const Lagrange3D&) = delete;
	Lagrange3D& operator=(Lagrange3D&&) = default;
};

#include "lagrange_3D.hpp"
